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

Since the first release there are two more, and they are not approximations of the first: the checkpoint
carries a row per answer mark, so a choice or a rubric level is read the same way a verdict is —
one pass, one dot product per option.

    choice  two to sixteen options, exactly one correct; marks are letters, because order among
            the options means nothing
    score   an ordinal rubric; marks are the levels\' own digits, because there order is the whole
            point — a miss lands on a neighbouring level rather than anywhere

A bundle of several questions is answered as a bundle: the material is read once for all of them
and each question costs only its own tail — the same economy the hosted service has, and the same
numbers as asking one by one.

Running the checkpoint yourself needs none of this: it is an ordinary classifier and
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
class Choice:
    """What `choice` and `scale` return: a probability per option, and which one leads.

    The keys are YOUR names for the options — they never reach the prompt, where the options are
    marked with letters or with the levels' own digits. The names travel from the request to the
    answer and nothing else, exactly as in the hosted service.
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
                 model: str = "typecastlm-qwen3.5-3.8b", timeout: float = 10.0, retries: int = 5,
                 pool: int = 32, temperature: float | None = None, calibrated: bool = True):
        """One knob, applied HERE, because the service sends the logits.

        `temperature` divides them before the softmax, and each mode has its own — the verdict,
        the three answers, a choice and a scale are calibrated separately, because a temperature
        corrects the mismatch between confidence and the difficulty of the DATA, and that differs
        per task. Passing one here overrides all of them; `calibrated=False` turns them off.

        Per-answer shifts used to live here too and are gone since 1.0.0. They were needed while
        the third row of the head was an average of vocabulary rows living on its own scale: the
        third logit sat far below the other two and a softmax crushed it to zero. The head now
        reads the third answer with a direction computed from data and scaled to the other two,
        and the perverse offset went with it — on a balanced set the model picks the third answer
        on 27% of rows where it is right on 33%, and fitting shifts on top buys 0.005 of
        calibration error and no accuracy at all.

        A temperature changes confidence and nothing else: the order of documents is untouched,
        so a ranking stays exactly as it was. Which makes it a calibration knob, to be set on your
        own data — and it lives client-side because the service sends the logits, so the same
        answer can be read again at another temperature without asking anything twice.
        """
        import requests
        from requests.adapters import HTTPAdapter

        self.endpoint = endpoint or os.environ.get("TYPECASTLM_ENDPOINT", DEFAULT_ENDPOINT)
        self.key = api_key or os.environ.get("TYPECASTLM_API_KEY", "")
        self.model, self.timeout, self.retries = model, timeout, retries
        self._s = requests.Session()
        self._s.mount("https://", HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
        self._requests = requests
        # The calibration travels with the service's answer: it belongs to the checkpoint, not to
        # the client. A temperature passed here overrides it, `calibrated=False` turns it off.
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.calibrated = calibrated
        self.temperature = temperature
        self.calibration: dict = {}

    def _temp(self, mode: str) -> float:
        """The temperature for a mode: the caller's, then the checkpoint's, then none."""
        if self.temperature is not None:
            return float(self.temperature)
        if not self.calibrated:
            return 1.0
        return float(((self.calibration or {}).get(mode) or {}).get("temperature", 1.0))

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

    def _soft(self, logits: dict, mode: str) -> dict[str, float]:
        """Softmax по выходам ОДНОГО режима, с его температурой.

        Нормировать все выходы головы разом бессмысленно: это ответы на разные вопросы, а не
        варианты одного. Температура у каждого режима своя, потому что правит она не веса, а
        несоответствие уверенности трудности данных, и на разных задачах оно разное.
        """
        v = {n: z / self._temp(mode) for n, z in logits.items()}
        m = max(v.values())
        e = {n: math.exp(x - m) for n, x in v.items()}
        s = sum(e.values())
        return {n: x / s for n, x in e.items()}

    def choice(self, state: str, instructions: str, options: dict[str, str]) -> "Choice":
        """Один верный вариант из нескольких. Имена вариантов ваши и в промпт не уходят."""
        return self._pick(state, instructions, options, "choice")

    def scale(self, state: str, instructions: str, levels: dict[str, str]) -> "Choice":
        """Порядковая шкала: уровни идут своими метками, и порядок несут они же."""
        return self._pick(state, instructions, levels, "score")

    def _pick(self, state: str, instructions: str, options: dict[str, str], kind: str) -> "Choice":
        if len(options) < 2:
            raise ValueError(f"{kind} takes at least two options")
        q = {"type": kind, "instructions": instructions, "criteria": dict(options)}
        data, ms = self._post({"state": state, "model": self.model, "questions": {"q": q}})
        self.calibration = data.get("calibration", self.calibration)
        z = data["answers"]["q"]["logits"]
        p = self._soft(z, "scale" if kind == "score" else "choice")
        lead = max(p, key=p.get)
        return Choice(p=p, verdict=lead, confidence=p[lead], logits=z, ms=ms,
                      input_tokens=int(data.get("usage", {}).get("input_tokens", 0)),
                      model=str(data.get("model", self.model)))

    def _three(self, logits: dict, mode: str = "ternary") -> dict[str, float]:
        """Three probabilities from three logits, calibrated for the mode asked for."""
        key = {"ternary": "three_answers", "binary": "verdict"}.get(mode, mode)
        v = {n: z / self._temp(key) for n, z in logits.items()}
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
        """Questions in the body the service takes — as many as you like about one state.

        A bundle is the cheap way to ask: the material is read once for all of them, and only
        each question\'s own tail is computed. On a policy of three thousand tokens that is about
        four times faster than asking one by one, and the numbers are identical — the material\'s
        state does not depend on the question, so sharing it changes nothing.
        """
        if not questions:
            raise ValueError("no questions")
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
            # From logits everything follows exactly: the margin is a difference, not the logarithm of
            # a rounded probability, so on confident documents it does not hit a clamp.
            names = list(z)
            p3 = self._three(z, "binary")
            decided = p3[names[0]] + p3[names[1]]
            prob = p3[names[0]] / decided if decided > 0 else 0.5
            margin = (z[names[0]] - z[names[1]]) / self._temp("verdict")
            unknown = p3[names[2]] if len(names) > 2 else 0.0
        else:                                   # the service sent a probability and nothing else
            prob = float(a["noul"])
            pc = min(max(prob, 1e-6), 1 - 1e-6)
            margin, unknown, p3 = math.log(pc / (1 - pc)), float(a.get("unknown", 0.0)), None
        return Answer(prob=prob, margin=margin, ms=ms,
                      input_tokens=int(data.get("usage", {}).get("input_tokens", 0)),
                      model=str(data.get("model", self.model)),
                      unknown=unknown, logits=z, p=p3)
