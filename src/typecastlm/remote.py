"""The client: a question over HTTP, an answer in numbers.

Deliberately the only thing here. Running the model means seven gigabytes of weights and a
deep-learning stack, and the machine that has a question is rarely the machine that should carry
them — a laptop, a request handler, a lambda. So the package depends on `requests` and stops
there.

Two question types, and they are not the same thing wearing two names:

    noul    the service's own — one probability of the `true` criterion among the two that decide
            the question. Kept exactly as it is over there, so a harness written against that
            service reads what it expects.
    tfu     ours — the three probabilities as they are, plus which one leads. A document that
            decides nothing comes back as that, instead of as a hedged yes.

The interface is compatible with the hosted decision service and wider by one number:

    noul    the probability of the `true` criterion among the two that decide the question
    unknown how much of the state points at "nothing here decides it" — never asked for, always
            returned
    logits  what the model said, before any softmax: probabilities follow from them at whatever
            temperature is asked for, and the log-odds follow exactly rather than through the
            logarithm of a rounded probability

Naming the answers is not a knob here, because it is not one over there either: in `choice` the
keys of `criteria` ARE the names, and they come back as the keys of `probabilities`. When `choice`
arrives it will name them that way; until then there is nothing to rename, and a second mechanism
for it would have been ours alone.

**Version zero answers one `noul` question per call.** `choice` and `score` are refused rather than
approximated, and a bundle of several questions is refused too: over there a bundle is cheap
because the state is read once for all of it, and here each question reads it again.

Running the checkpoint yourself needs none of this: it is an ordinary three-label classifier and
`transformers` loads it directly.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

DEFAULT_ENDPOINT = "https://api.promptidote.ru/v1/typecast"
RETRY_CODES = (408, 429, 500, 502, 503, 504, 529)


@dataclass
class Ternary:
    """What `tfu` returns: the three probabilities as they are, and what leads.

    Our own shape, not the service's. `noul` answers "yes or no, and how sure"; this one answers
    "which of the three", and a document that decides nothing shows up as itself rather than as a
    hedged yes.
    """

    p: dict[str, float]
    verdict: str
    confidence: float
    logits: dict[str, float]
    ms: float
    input_tokens: int
    model: str


@dataclass
class Answer:
    """What `noul` returns.

    `prob`, `margin`, `ms`, `input_tokens` and `model` are the service's fields. The extensions
    are `unknown` — the weight of "nothing here decides it", which the criteria never ask for —
    and `logits` with `p`: what the model said, and what that comes to at the temperature asked
    for.
    """

    prob: float
    margin: float
    ms: float
    input_tokens: int
    model: str
    unknown: float = 0.0
    logits: dict[str, float] | None = None
    p: dict[str, float] | None = None


class Client:
    """One question, one state, three numbers — answered by the service.

    The key is read from `TYPECASTLM_API_KEY` unless one is passed. A connection is kept open for
    the life of the client: a fresh one per request means a name lookup per request, and a few
    dozen of those at once is how a fast service starts looking slow.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 model: str = "typecastlm-qwen3-3.5b", timeout: float = 10.0, retries: int = 5,
                 pool: int = 32, temperature: float | None = None,
                 shift: tuple[float, float] | None = None, calibrated: bool = True):
        """Two knobs, both applied HERE, because the service sends the logits.

        `temperature` divides them before the softmax. `shift` adds a constant to the second and
        third answers, relative to the first — **two** numbers for three answers, because a
        softmax cannot tell a common shift from nothing, so three would be one too many.

        What the shift is for: the third logit sits far below the other two — the model was never
        going to write that word — and the softmax crushes it to zero. Its RANKING survives, its
        probability does not, and a threshold on zero is no threshold. Lifting it puts the number
        back on a usable scale.

        What the shift is NOT for: it changes no ordering whatsoever. A reading that ranks badly
        keeps ranking badly at every shift; this is calibration, not improvement, and the numbers
        belong to your data — fit them on a few hundred of your own documents.

        It changes confidence and nothing else: the order of documents is untouched, so a ranking
        stays exactly as it was and only the probabilities move. Which makes it a calibration
        knob, to be set on your own data and no one else's — and it lives client-side because the
        service sends the logits, so the same answer can be read again at another temperature
        without asking anything twice.
        """
        import requests
        from requests.adapters import HTTPAdapter

        self.endpoint = endpoint or os.environ.get("TYPECASTLM_ENDPOINT", DEFAULT_ENDPOINT)
        self.key = api_key or os.environ.get("TYPECASTLM_API_KEY", "")
        self.model, self.timeout, self.retries = model, timeout, retries
        self._s = requests.Session()
        self._s.mount("https://", HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
        self._requests = requests
        # Калибровка приезжает с ответом сервиса — она свойство чекпойнта, а не клиента. Явно
        # заданные числа её перекрывают, `calibrated=False` выключает совсем.
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if shift is not None and len(shift) != 2:
            raise ValueError(f"three answers have two free shifts, got {len(shift)}")
        self.calibrated = calibrated
        self._own = dict(temperature=temperature,
                         shift=tuple(float(x) for x in shift) if shift else None)
        self.temperature, self.shift = temperature or 1.0, shift or (0.0, 0.0)
        self.calibration: dict = {}

    def _use(self, mode: str) -> tuple[float, tuple[float, float]]:
        """Температура и два сдвига для режима: своё, потом присланное, потом ничего."""
        c = (self.calibration or {}).get(mode, {}) if self.calibrated else {}
        t = self._own["temperature"] or float(c.get("temperature", 1.0))
        if self._own["shift"] is not None:
            return t, self._own["shift"]
        if mode == "binary":
            return t, (float(c.get("shift_false", 0.0)), 0.0)
        sh = c.get("shift", [0.0, 0.0, 0.0])
        return t, (float(sh[1]), float(sh[2]))

    def _post(self, body: dict) -> tuple[dict, float]:
        delay = 1.0
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
                r = self._s.post(self.endpoint, json=body, headers=headers, timeout=self.timeout)
                ms = (time.perf_counter() - t0) * 1000.0
                if r.status_code == 200:
                    return r.json(), ms
                if r.status_code not in RETRY_CODES or attempt == self.retries:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            except self._requests.RequestException:
                if attempt == self.retries:
                    raise
            ra = None
            time.sleep(float(ra) if ra else delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    def _three(self, logits: dict, mode: str = "ternary") -> dict[str, float]:
        """Три вероятности из трёх логитов при калибровке выбранного режима."""
        t, sh = self._use(mode)
        b = (0.0,) + tuple(sh)
        v = {n: (z + s) / t for (n, z), s in zip(logits.items(), b)}
        m = max(v.values())
        e = {n: math.exp(x - m) for n, x in v.items()}
        s = sum(e.values())
        return {n: x / s for n, x in e.items()}

    def tfu(self, state: str, instructions: str, true: str | None = None,
            false: str | None = None) -> Ternary:
        """The three answers, unreduced — our type, beside the compatible one.

        The criteria are still two: `unknown` is not something anyone states, it is what is left
        when neither of the two fits. The temperature of this client applies here as well.
        """
        q: dict = {"type": "tfu", "instructions": instructions}
        if true is not None or false is not None:
            q["criteria"] = {"true": true or "", "false": false or ""}
        data, ms = self._post({"state": state, "model": self.model, "questions": {"q": q}})
        self.calibration = data.get("calibration", self.calibration)
        a = data["answers"]["q"]
        z = a["logits"]
        p = self._three(z, "ternary")
        lead = max(p, key=p.get)
        return Ternary(p=p, verdict=lead, confidence=p[lead], logits=z, ms=ms,
                       input_tokens=int(data.get("usage", {}).get("input_tokens", 0)),
                       model=str(data.get("model", self.model)))

    def ask(self, state: str, questions: dict) -> dict:
        """A question in the body the service takes. Version zero sends one at a time."""
        if len(questions) != 1:
            raise NotImplementedError(
                f"version zero asks one question per call, got {len(questions)}")
        data, _ = self._post({"state": state, "model": self.model, "questions": questions})
        self.calibration = data.get("calibration", self.calibration)
        return {"answers": data["answers"],
                "input_tokens": int(data.get("usage", {}).get("input_tokens", 0))}

    def noul(self, state: str, instructions: str, true: str | None = None,
             false: str | None = None) -> Answer:
        """One closed question. `margin` is the log-odds, so a saturated probability still ranks."""
        q: dict = {"type": "noul", "instructions": instructions}
        if true is not None or false is not None:
            q["criteria"] = {"true": true or "", "false": false or ""}
        data, ms = self._post({"state": state, "model": self.model, "questions": {"q": q}})
        self.calibration = data.get("calibration", self.calibration)
        a = data["answers"]["q"]
        z = a.get("logits")
        if z:
            # Из логитов всё считается точно: маржа это разность, а не логарифм округлённой
            # вероятности, и на уверенных документах она не упирается в потолок зажима.
            names = list(z)
            p3 = self._three(z, "binary")
            decided = p3[names[0]] + p3[names[1]]
            prob = p3[names[0]] / decided if decided > 0 else 0.5
            t, sh = self._use("binary")
            margin = (z[names[0]] - (z[names[1]] + sh[0])) / t
            unknown = p3[names[2]] if len(names) > 2 else 0.0
        else:                                   # сервис прислал только вероятность
            prob = float(a["noul"])
            pc = min(max(prob, 1e-6), 1 - 1e-6)
            margin, unknown, p3 = math.log(pc / (1 - pc)), float(a.get("unknown", 0.0)), None
        return Answer(prob=prob, margin=margin, ms=ms,
                      input_tokens=int(data.get("usage", {}).get("input_tokens", 0)),
                      model=str(data.get("model", self.model)),
                      unknown=unknown, logits=z, p=p3)
