"""HTTP client: a question about a state, an answer in numbers.

    noul(state, instructions, true, false)   -> Answer(prob, margin, unknown, logits, p)
    tfu(state, instructions, true, false)    -> Ternary(p, verdict, confidence, logits)
    choice(state, instructions, options)     -> Choice(p, verdict, confidence, logits)
    scale(state, instructions, levels)       -> Choice, for an ordinal rubric
    ask(state, questions)                    -> the service body, any number of questions

`noul` matches the hosted service field for field. The rest are extensions: `unknown` is returned
whether or not the question asks for it, `logits` are raw, and option names in `choice` and
`scale` come back as the keys of `p`.

The service holds the weights; this package depends on `requests` only.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

DEFAULT_ENDPOINT = ""          # no hosted service; set one or run `typecastlm-serve`
RETRY_CODES = (408, 429, 500, 502, 503, 504, 529)


@dataclass
class Ternary:
    """What `tfu` returns: the three probabilities and which one leads."""

    p: dict[str, float]
    verdict: str
    confidence: float
    logits: dict[str, float]
    ms: float
    input_tokens: int
    model: str


@dataclass
class Choice:
    """What `choice` and `scale` return: a probability per option and which one leads. Keys are
    the option names from the request."""

    p: dict[str, float]
    verdict: str
    confidence: float
    logits: dict[str, float]
    ms: float
    input_tokens: int
    model: str


@dataclass
class Answer:
    """What `noul` returns. `prob`, `margin`, `ms`, `input_tokens`, `model` are the service\'s
    fields; `unknown`, `logits` and `p` are extensions."""

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
        """`temperature` overrides the checkpoint\'s per-mode temperatures; `calibrated=False`
        disables them. Temperatures are applied here because the service returns logits, so an
        answer can be re-read at another temperature without asking again. They change no
        ordering, only confidence."""
        import requests
        from requests.adapters import HTTPAdapter

        self.endpoint = endpoint or os.environ.get("TYPECASTLM_ENDPOINT", DEFAULT_ENDPOINT)
        if not self.endpoint:
            raise ValueError(
                "no endpoint. Pass Client(endpoint=...), set TYPECASTLM_ENDPOINT, or run the "
                "model yourself: `pip install \"typecastlm[server]\"` and `typecastlm-serve`, or "
                "`pip install \"typecastlm[local]\"` and use typecastlm.Reader in this process")
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
        """Softmax over one mode\'s outputs, at that mode\'s temperature."""
        v = {n: z / self._temp(mode) for n, z in logits.items()}
        m = max(v.values())
        e = {n: math.exp(x - m) for n, x in v.items()}
        s = sum(e.values())
        return {n: x / s for n, x in e.items()}

    def choice(self, state: str, instructions: str, options: dict[str, str]) -> "Choice":
        """Pick one of several options. Option names are yours and do not reach the prompt."""
        return self._pick(state, instructions, options, "choice")

    def scale(self, state: str, instructions: str, levels: dict[str, str]) -> "Choice":
        """Pick a level of an ordinal rubric."""
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
        """Any number of questions about one state. The state is read once for the bundle;
        results equal asking one by one."""
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
