"""The client: a question over HTTP, an answer in numbers.

Deliberately the only thing here. Running the model means seven gigabytes of weights and a
deep-learning stack, and the machine that has a question is rarely the machine that should carry
them — a laptop, a request handler, a lambda. So the package depends on `requests` and stops
there.

The interface is compatible with the hosted decision service and wider by one number:

    noul    the probability of the `true` criterion among the two that decide the question
    unknown how much of the state points at "nothing here decides it" — never asked for, always
            returned

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
class Answer:
    """What `noul` returns. Every field but `unknown` is the service's; `unknown` is the extension."""

    prob: float
    margin: float
    ms: float
    input_tokens: int
    model: str
    unknown: float = 0.0


class Client:
    """One question, one state, three numbers — answered by the service.

    The key is read from `TYPECASTLM_API_KEY` unless one is passed. A connection is kept open for
    the life of the client: a fresh one per request means a name lookup per request, and a few
    dozen of those at once is how a fast service starts looking slow.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 model: str = "typecastlm-qwen3-3.5b", timeout: float = 10.0, retries: int = 5,
                 pool: int = 32):
        import requests
        from requests.adapters import HTTPAdapter

        self.endpoint = endpoint or os.environ.get("TYPECASTLM_ENDPOINT", DEFAULT_ENDPOINT)
        self.key = api_key or os.environ.get("TYPECASTLM_API_KEY", "")
        self.model, self.timeout, self.retries = model, timeout, retries
        self._s = requests.Session()
        self._s.mount("https://", HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
        self._requests = requests

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

    def ask(self, state: str, questions: dict) -> dict:
        """A question in the body the service takes. Version zero sends one at a time."""
        if len(questions) != 1:
            raise NotImplementedError(
                f"version zero asks one question per call, got {len(questions)}")
        data, _ = self._post({"state": state, "model": self.model, "questions": questions})
        return {"answers": data["answers"],
                "input_tokens": int(data.get("usage", {}).get("input_tokens", 0))}

    def noul(self, state: str, instructions: str, true: str | None = None,
             false: str | None = None) -> Answer:
        """One closed question. `margin` is the log-odds, so a saturated probability still ranks."""
        q: dict = {"type": "noul", "instructions": instructions}
        if true is not None or false is not None:
            q["criteria"] = {"true": true or "", "false": false or ""}
        data, ms = self._post({"state": state, "model": self.model, "questions": {"q": q}})
        a = data["answers"]["q"]
        p = float(a["noul"])
        pc = min(max(p, 1e-6), 1 - 1e-6)
        return Answer(prob=p, margin=math.log(pc / (1 - pc)), ms=ms,
                      input_tokens=int(data.get("usage", {}).get("input_tokens", 0)),
                      model=str(data.get("model", self.model)),
                      unknown=float(a.get("unknown", 0.0)))
