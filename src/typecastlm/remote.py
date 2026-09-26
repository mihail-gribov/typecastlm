"""HTTP client: a question about a state, an answer in numbers.

    noul(state, instructions, true, false)   -> Answer(prob, margin, logits)
    tfu(state, instructions, true, false)    -> Ternary(p, verdict, confidence, logits)
    choice(state, instructions, options)     -> Choice(p, verdict, confidence, logits)
    scale(state, instructions, levels)       -> Choice, for an ordinal rubric
    ask(state, questions)                    -> the service body, any number of questions

`noul` matches the hosted service field for field: two answers, a softmax over the two, and
nothing else in it. The third answer is a mode of its own, `tfu`, fitted and calibrated
separately — reading it out of a `noul` call would mix two temperatures. `logits` are raw, and
option names in `choice` and `scale` come back as the keys of `p`.

One client, several places the model may be. `api` says which, and the address says where:

    Client("http://localhost:8000")                         # typecastlm-serve, ours
    Client("localhost:8080", api="llama")                   # llama-server with the GGUF
    Client("http://gpu-box:8000", api="openai")             # any server speaking /v1/embeddings
    Client.jev(api_key="…")                                 # TypeSafe's Jev, the API this copies
    Client()                                                # TYPECASTLM_ENDPOINT, TYPECASTLM_API

The four modes are the same calls everywhere. What differs is what comes back with them: our
service and the GGUF readers return `logits` and the calibration, so an answer can be re-read at
another temperature; Jev returns probabilities alone, and `tfu`, which Jev does not have, is
asked as a `choice` with a third option and marked `native=False`. Jev takes up to 255 options
and 10 levels, ours 26 and 36; either side refuses what it cannot read, and the answer says so.

The service holds the weights; this package depends on `requests` only.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

DEFAULT_ENDPOINT = ""          # no hosted service; set one or run `typecastlm-serve`
RETRY_CODES = (408, 429, 500, 502, 503, 504, 529)
ROUTE = "/v1/systemone"
APIS = ("auto", "typecastlm", "jev", "llama", "openai")
JEV_ENDPOINT = "https://api.typesafe.ai"
JEV_MODEL = "jev-latest"
DEFAULT_MODEL = "typecastlm-qwen3.5-3.8b"
# The wording of the third answer when it has to be asked for as an option — the same sentence
# the shipped prompt gives `unsure`, so an emulated `tfu` asks what the native one asks.
UNSURE = ("the material leaves the question unclear, unanswered or self-contradictory, or the "
          "task itself is ill-posed — there is nothing here to judge")
CRITERIA_DEFAULT = {"true": "the answer to the question is yes",
                    "false": "the answer to the question is no"}


def _base(endpoint: str) -> str:
    """An address as a base URL: `localhost:8080` gets `http://`, a trailing slash goes."""
    e = (endpoint or "").strip().rstrip("/")
    if e and "://" not in e:
        e = "http://" + e
    return e


class ApiError(RuntimeError):
    """The service refused: the HTTP status and the message it gave, whatever shape it came in
    (Jev wraps it in `detail.message`, FastAPI in `detail` as a string or a list)."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status, self.message = status, message


def _message(text: str) -> str:
    import json

    try:
        d = json.loads(text).get("detail", text)
    except Exception:
        return text[:300]
    if isinstance(d, dict):
        return str(d.get("message") or d)[:300]
    if isinstance(d, list):
        return "; ".join(f"{'.'.join(str(x) for x in e.get('loc', [])[1:])}: {e.get('msg')}"
                         for e in d if isinstance(e, dict))[:300]
    return str(d)[:300]


def _route(endpoint: str) -> str:
    """A host without a path gets the route added.

    `TYPECASTLM_ENDPOINT=http://localhost:8000` is what anyone writes after starting the service,
    and posting to the bare host answers 404. An endpoint that already carries a path is left
    alone: a gateway may mount the service anywhere.
    """
    e = _base(endpoint)
    if not e:
        return ""
    rest = e.split("://", 1)[-1]
    return e if "/" in rest else e + ROUTE


@dataclass
class Ternary:
    """What `tfu` returns: the three probabilities and which one leads."""

    p: dict[str, float]
    verdict: str
    confidence: float
    logits: dict[str, float] | None
    ms: float
    input_tokens: int
    model: str
    native: bool = True             # False: asked as a `choice` with a third option, on a server
                                    # that has no `tfu` — Jev — and read from its probabilities


@dataclass
class Choice:
    """What `choice` and `scale` return: a probability per option and which one leads. Keys are
    the option names from the request. `scale` also fills `score`, the mean level — the answer
    lands between levels, which is what an ordinal rubric is for."""

    p: dict[str, float]
    verdict: str
    confidence: float
    logits: dict[str, float] | None
    ms: float
    input_tokens: int
    model: str
    score: float | None = None      # `scale` only: the mean level, between 0 and len - 1


@dataclass
class Answer:
    """What `noul` returns: the two answers that decide the question, read against each other.

    `prob`, `margin`, `ms`, `input_tokens`, `model` are the service's fields; `logits` is the raw
    payload. There is no third number here — ask `tfu` for that reading.
    """

    prob: float
    margin: float
    ms: float
    input_tokens: int
    model: str
    logits: dict[str, float] | None = None


class Client:
    """One question about one state, answered in numbers by whichever server holds the model.

    The key is read from `TYPECASTLM_API_KEY` unless one is passed (`TYPESAFE_API_KEY` for Jev).
    A connection is kept open for the life of the client: a fresh one per request means a name
    lookup per request, and a few dozen of those at once is how a fast service starts looking
    slow.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float | None = None, retries: int = 5,
                 pool: int = 32, temperature: float | None = None, calibrated: bool = True,
                 transport=None, api: str | None = None, host: str | None = None,
                 port: int | None = None, files: str | None = None):
        """`api` names the server: `typecastlm` (ours, `typecastlm-serve`), `jev` (TypeSafe's),
        `llama` (llama-server holding the GGUF), `openai` (any server with `/v1/embeddings`
        that returns the vector raw), or `auto`, the default, which recognises Jev by its host
        and asks anything else what it is on the first call. `endpoint` is the address, or
        `host` and `port` are; `TYPECASTLM_ENDPOINT` and `TYPECASTLM_API` stand in for both.

        `temperature` overrides the checkpoint's per-mode temperatures; `calibrated=False`
        disables them. Temperatures are applied here because the service returns logits, so an
        answer can be re-read at another temperature without asking again. They change no
        ordering, only confidence. Jev returns no logits, and neither setting touches it.

        `transport` replaces the server: a callable that takes a request body and returns the
        answer body, such as an `EmbeddingReader` you built yourself. `files` is the model
        directory or Hub repository the GGUF readers take `prompt.json` and `head.json` from."""
        import requests
        from requests.adapters import HTTPAdapter

        api = (api or os.environ.get("TYPECASTLM_API") or "auto").lower()
        if api not in APIS:
            raise ValueError(f"api is one of {APIS}, not {api!r}")
        if not endpoint and (host or port):
            endpoint = f"http://{host or '127.0.0.1'}:{port or 8000}"
        endpoint = endpoint or os.environ.get("TYPECASTLM_ENDPOINT", DEFAULT_ENDPOINT)
        if api == "auto":
            if transport is not None:
                api = "typecastlm"
            elif endpoint and "typesafe.ai" in _base(endpoint):
                api = "jev"
            elif endpoint:
                api = "detect"                   # asked on the first call
        if api == "jev":
            endpoint = endpoint or JEV_ENDPOINT
            api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.api, self.transport = api, transport
        self.base = _base(endpoint)
        self.endpoint = _route(endpoint) if self.base else ""
        if transport is not None:
            self.endpoint = self.endpoint or "transport"
        if not self.endpoint:
            raise ValueError(
                "no endpoint. Pass Client(endpoint=...), set TYPECASTLM_ENDPOINT, or run the "
                "model yourself: `pip install \"typecastlm[server]\"` and `typecastlm-serve`, "
                "or a llama-server with the GGUF and Client(endpoint, api=\"llama\"); "
                "Client.jev(api_key=...) reaches TypeSafe's Jev")
        self.key = api_key or os.environ.get("TYPECASTLM_API_KEY", "")
        self.model = model or os.environ.get("TYPECASTLM_MODEL") or (
            JEV_MODEL if api == "jev" else DEFAULT_MODEL)
        self.files = files
        # Jev's median is a quarter of a second and its tail a minute; a timeout that fires on
        # the tail re-sends the work and pays for it twice.
        self.timeout = timeout if timeout is not None else (30.0 if api == "jev" else 10.0)
        self.retries = retries
        self._s = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
        for scheme in ("https://", "http://"):       # a service of your own is usually http
            self._s.mount(scheme, adapter)
        self._requests = requests
        # The calibration travels with the service's answer: it belongs to the checkpoint, not to
        # the client. A temperature passed here overrides it, `calibrated=False` turns it off.
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.calibrated = calibrated
        self.temperature = temperature
        self.calibration: dict = {}
        self.native_tfu: bool | None = (False if api == "jev" else None)   # learned on first use
        if api in ("llama", "openai"):
            self._reader(api)

    @classmethod
    def jev(cls, api_key: str | None = None, model: str = JEV_MODEL,
            endpoint: str = JEV_ENDPOINT, **kw) -> "Client":
        """TypeSafe's Jev: the API this package copies, at its own address. The key is
        `TYPESAFE_API_KEY` unless passed."""
        return cls(endpoint=endpoint, api_key=api_key, model=model, api="jev", **kw)

    def _reader(self, api: str) -> None:
        """A GGUF server becomes a transport: the reader does the prompt and the head."""
        from .embedding import EmbeddingReader

        kw = {"api": api, "api_key": self.key}
        if self.files:
            kw["model"] = self.files             # else the GGUF repository, which holds both files
        self.transport = EmbeddingReader(self.base, **kw)
        self.api = api

    def _detect(self) -> None:
        """What is at the address, asked once: our service answers `/health` with its labels,
        a llama-server answers `/embedding` with a vector, anything with `/v1/models` is
        Jev-shaped, and what is left is taken for an OpenAI-style embeddings server."""
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        try:
            r = self._s.get(f"{self.base}/health", timeout=self.timeout)
            if r.status_code == 200 and isinstance(r.json(), dict) and "labels" in r.json():
                self.api = "typecastlm"
                return
        except (self._requests.RequestException, ValueError):
            pass
        try:
            r = self._s.post(f"{self.base}/embedding", json={"content": "?", "embd_normalize": -1},
                             headers=headers, timeout=self.timeout)
            if r.status_code == 200:
                self._reader("llama")
                return
        except self._requests.RequestException:
            pass
        try:
            r = self._s.get(f"{self.base}/v1/models", headers=headers, timeout=self.timeout)
            if r.status_code in (200, 401) and "embeddings" not in r.text[:200]:
                self.api = "typecastlm"
                return
        except self._requests.RequestException:
            pass
        self._reader("openai")

    def info(self) -> dict:
        """Which server, where, and what it gives back."""
        if self.api == "detect":
            self._detect()
        return {"api": self.api, "endpoint": self.base or "transport", "model": self.model,
                "logits": bool(self.calibration) or None, "native_tfu": self.native_tfu}

    def models(self) -> list[dict]:
        """`GET /v1/models`: the names a request may carry, with descriptions and dates."""
        if self.api == "detect":
            self._detect()
        if self.transport is not None:
            return [{"name": self.model, "description": "read through a GGUF server",
                     "release_date": ""}]
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = self._s.get(f"{self.base}/v1/models", headers=headers, timeout=self.timeout)
        if r.status_code != 200:
            raise ApiError(r.status_code, _message(r.text))
        return r.json().get("models", [])

    def _temp(self, mode: str) -> float:
        """The temperature for a mode: the caller's, then the checkpoint's, then none."""
        if self.temperature is not None:
            return float(self.temperature)
        if not self.calibrated:
            return 1.0
        return float(((self.calibration or {}).get(mode) or {}).get("temperature", 1.0))

    def _post(self, body: dict) -> tuple[dict, float]:
        if self.api == "detect":
            self._detect()
        if self.transport is not None:
            t0 = time.perf_counter()
            data = self.transport(body)
            return data, (time.perf_counter() - t0) * 1000.0
        delay = 1.0
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            wait = None
            try:
                headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
                r = self._s.post(self.endpoint, json=body, headers=headers, timeout=self.timeout)
                ms = (time.perf_counter() - t0) * 1000.0
                if r.status_code == 200:
                    return r.json(), ms
                if r.status_code not in RETRY_CODES or attempt == self.retries:
                    raise ApiError(r.status_code, _message(r.text))
                # A service that says when to come back knows better than our doubling.
                ra = r.headers.get("retry-after", "")
                wait = float(ra) if ra.replace(".", "", 1).isdigit() else None
            except self._requests.RequestException:
                if attempt == self.retries:
                    raise
            time.sleep(wait if wait is not None else delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    def _soft(self, logits: dict, mode: str) -> dict[str, float]:
        """Softmax over one mode's outputs, at that mode's temperature."""
        v = {n: z / self._temp(mode) for n, z in logits.items()}
        m = max(v.values())
        e = {n: math.exp(x - m) for n, x in v.items()}
        s = sum(e.values())
        return {n: x / s for n, x in e.items()}

    @staticmethod
    def _tokens(data: dict) -> int:
        return int((data.get("usage") or {}).get("input_tokens", 0))

    def choice(self, state: str, instructions: str, options: dict[str, str]) -> "Choice":
        """Pick one of several options. Option names are yours and do not reach the prompt."""
        return self._pick(state, instructions, options, "choice")

    def scale(self, state: str, instructions: str, levels: dict[str, str] | list[str]) -> "Choice":
        """Pick a level of an ordinal rubric. Levels are ordered — a list, or a map whose order
        is the order — and the answer comes back under your names."""
        if isinstance(levels, (list, tuple)):
            levels = {str(i): d for i, d in enumerate(levels)}
        return self._pick(state, instructions, levels, "score")

    def _pick(self, state: str, instructions: str, options: dict[str, str], kind: str) -> "Choice":
        if len(options) < 2:
            raise ValueError(f"{kind} takes at least two options")
        keys = [str(k) for k in options]
        if kind == "score":
            # Levels go on the wire as a list, the form the Jev API defines and ours accepts
            # too; positions come back, and are mapped to the caller's names by order.
            q = {"type": kind, "instructions": instructions, "criteria": list(options.values())}
        else:
            q = {"type": kind, "instructions": instructions, "criteria": dict(options)}
        data, ms = self._post({"state": state, "model": self.model, "questions": {"q": q}})
        self.calibration = data.get("calibration", self.calibration)
        a = data["answers"]["q"]
        rename = ({str(i): k for i, k in enumerate(keys)} if kind == "score"
                  else {k: k for k in keys})
        z = a.get("logits")
        z = {rename.get(n, n): v for n, v in z.items()} if z else None
        # A service that sends finished probabilities and no logits — Jev itself, for one — is
        # answered from those; the temperature has then already been applied by whoever fitted it.
        p = (self._soft(z, "scale" if kind == "score" else "choice") if z
             else {rename.get(n, n): float(v) for n, v in a["probabilities"].items()})
        p = {k: p[k] for k in keys if k in p} or p
        named = a.get(kind)                     # `choice` names the leader; `score` gives a number
        named = rename.get(named, named) if isinstance(named, str) else named
        lead = named if isinstance(named, str) and named in p else max(p, key=p.get)
        mean = sum(i * v for i, v in enumerate(p.values())) if kind == "score" else None
        return Choice(p=p, verdict=lead, confidence=p[lead], logits=z, ms=ms,
                      input_tokens=self._tokens(data), model=str(data.get("model", self.model)),
                      score=mean)

    def _three(self, logits: dict) -> dict[str, float]:
        """Three probabilities from three logits, at the three-answer temperature."""
        v = {n: z / self._temp("three_answers") for n, z in logits.items()}
        m = max(v.values())
        e = {n: math.exp(x - m) for n, x in v.items()}
        s = sum(e.values())
        return {n: x / s for n, x in e.items()}

    def tfu(self, state: str, instructions: str, true: str | None = None,
            false: str | None = None) -> Ternary:
        """The three answers, unreduced — our type, beside the compatible one.

        The criteria are still two: `unsure` is not something anyone states, it is what is left
        when neither of the two fits. The temperature of this client applies here as well.

        A server without the type — Jev — is asked a `choice` among the two criteria and a
        third option worded as the shipped prompt words `unsure`, and read from its
        probabilities; the answer says `native=False`. That is the reading the model gives when
        told the third answer exists, not the calibrated third output, and the two are not the
        same number.
        """
        if self.native_tfu is False:
            return self._tfu_as_choice(state, instructions, true, false)
        q: dict = {"type": "tfu", "instructions": instructions}
        if true is not None or false is not None:
            q["criteria"] = {"true": true or "", "false": false or ""}
        try:
            data, ms = self._post({"state": state, "model": self.model, "questions": {"q": q}})
        except ApiError as e:
            if e.status in (400, 422) and self.native_tfu is None:
                self.native_tfu = False          # not this server; remembered
                return self._tfu_as_choice(state, instructions, true, false)
            raise
        self.native_tfu = True
        self.calibration = data.get("calibration", self.calibration)
        a = data["answers"]["q"]
        z = a.get("logits")
        p = self._three(z) if z else a["probabilities"]
        lead = max(p, key=p.get)
        return Ternary(p=p, verdict=lead, confidence=p[lead], logits=z, ms=ms,
                       input_tokens=self._tokens(data), model=str(data.get("model", self.model)))

    def _tfu_as_choice(self, state, instructions, true, false) -> Ternary:
        c = self.choice(state, instructions,
                        {"true": true or CRITERIA_DEFAULT["true"],
                         "false": false or CRITERIA_DEFAULT["false"], "unsure": UNSURE})
        return Ternary(p=c.p, verdict=c.verdict, confidence=c.confidence, logits=None, ms=c.ms,
                       input_tokens=c.input_tokens, model=c.model, native=False)

    def ask(self, state: str, questions: dict) -> dict:
        """Any number of questions about one state, answered in one call, the answers passed
        through as the server sent them. The state is read once for the bundle, by our service
        and by Jev alike, and `input_tokens` counts it once."""
        if not questions:
            raise ValueError("no questions")
        data, _ = self._post({"state": state, "model": self.model, "questions": questions})
        self.calibration = data.get("calibration", self.calibration)
        return {"answers": data["answers"], "input_tokens": self._tokens(data),
                "model": str(data.get("model", self.model))}

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
            # Two answers decide this question, so the softmax runs over those two and the third
            # output takes no part in it. The margin is a difference of logits rather than the
            # logarithm of a rounded probability, so a confident document does not hit a clamp.
            names = list(z)[:2]
            margin = (z[names[0]] - z[names[1]]) / self._temp("verdict")
            prob = 1.0 / (1.0 + math.exp(-margin))
        else:                                   # the service sent a probability and nothing else
            prob = float(a["noul"])
            pc = min(max(prob, 1e-6), 1 - 1e-6)
            margin = math.log(pc / (1 - pc))
        return Answer(prob=prob, margin=margin, ms=ms, input_tokens=self._tokens(data),
                      model=str(data.get("model", self.model)), logits=z)
