"""The same surface over HTTP, for machines that should not carry a model.

`Client` loads seven gigabytes and needs torch. On a laptop, in a browser backend, inside a lambda
— none of that belongs there, and the question is not big enough to justify it. `RemoteClient`
answers the same calls by asking a service, and depends on nothing but `requests`.

The two are interchangeable on purpose: the same code runs against local weights while it is being
developed and against a service in production, and the answers have the same fields either way.
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
    prob: float
    margin: float
    ms: float
    input_tokens: int
    model: str


class RemoteClient:
    """Same calls as `Client`, answered by a service.

    The key is read from `TYPECASTLM_API_KEY` unless one is passed. A connection is kept open for
    the life of the client: a fresh one per request means a name lookup per request, and a few
    dozen of those at once is how a fast service starts looking slow.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 model: str = "typecastlm-qwen3-4b", timeout: float = 10.0, retries: int = 5,
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
        data, _ = self._post({"state": state, "model": self.model, "questions": questions})
        return {"answers": data["answers"],
                "input_tokens": int(data.get("usage", {}).get("input_tokens", 0))}

    def noul(self, state: str, instructions: str, true: str | None = None,
             false: str | None = None) -> Answer:
        q: dict = {"type": "noul", "instructions": instructions}
        if true is not None or false is not None:
            q["criteria"] = {"true": true or "", "false": false or ""}
        data, ms = self._post({"state": state, "model": self.model, "questions": {"q": q}})
        p = float(data["answers"]["q"]["noul"])
        pc = min(max(p, 1e-6), 1 - 1e-6)
        return Answer(prob=p, margin=math.log(pc / (1 - pc)), ms=ms,
                      input_tokens=int(data.get("usage", {}).get("input_tokens", 0)),
                      model=str(data.get("model", self.model)))
