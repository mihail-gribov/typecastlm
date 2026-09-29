"""Jev behind our API: the service as a proxy.

The third place the model may be is not ours at all. With `--backend jev` the service forwards
each request to TypeSafe's Jev and answers in the same shape it answers from its own weights,
so a caller sees one address whichever model is behind it, and switching between them is a
setting rather than a deployment. What the proxy adds is what Jev lacks: `tfu` is asked as a
`choice` with a third option and comes back marked `native: false`, `scale` levels given as a
map travel as the list Jev wants and come back under the caller's names. What it cannot add is
`logits` and a calibration — Jev returns probabilities alone, and the answer says so by leaving
those fields out.

A bundle is forwarded as one request, which is where Jev bills the state once.
"""
import os

from .reading import Reading
from .remote import (CRITERIA_DEFAULT, JEV_ENDPOINT, JEV_MODEL, UNSURE, ApiError, _base,
                     _message)


class JevReader(Reading):
    """Answers questions by asking Jev, in the service's own shape."""

    def __init__(self, endpoint: str = JEV_ENDPOINT, api_key: str = "", model: str = JEV_MODEL,
                 timeout: float = 60.0, check: bool = True):
        import requests

        self.endpoint = _base(endpoint) or JEV_ENDPOINT
        self.key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.name, self.served, self.model = self.endpoint, model, model
        self.timeout = timeout
        self.prompt_cfg = {"max_state_tokens": 32768, "calibration": {}}
        self.max_state_tokens = 32768
        self.labels: list[str] = []
        self.prompt_source = "upstream"
        self._s = requests.Session()
        if self.key:
            self._s.headers["Authorization"] = f"Bearer {self.key}"
        if check:
            self.check()

    # -- the upstream --------------------------------------------------------------------------

    def _get(self, route: str):
        return self._s.get(f"{self.endpoint}{route}", timeout=self.timeout)

    def check(self, strict: bool = True) -> list[str]:
        bad = []
        try:
            r = self._get("/v1/models")
            if r.status_code == 401:
                bad.append("the upstream refuses the key")
            elif r.status_code != 200:
                bad.append(f"the upstream answers {r.status_code} to /v1/models")
            elif self.model not in {m.get("name") for m in r.json().get("models", [])}:
                bad.append(f"the upstream does not list model {self.model!r}")
        except Exception as e:
            bad.append(f"the upstream does not answer: {type(e).__name__}")
        if bad and strict:
            raise RuntimeError("; ".join(bad))
        return bad

    def describe(self) -> dict:
        return {"backend": "jev", "model": self.endpoint, "labels": self.labels,
                "device": self.endpoint, "dtype": "upstream", "prompt": "upstream"}

    def metadata(self) -> list[dict]:
        try:
            r = self._get("/v1/models")
            if r.status_code == 200:
                return r.json().get("models", [])
        except Exception:
            pass
        return [{"name": self.model, "description": "proxied to " + self.endpoint,
                 "release_date": "unknown"}]

    # -- what Reading asks for; the proxy never builds a prompt --------------------------------

    def fold(self, state: str) -> str:
        return state

    def render(self, system: str, user: str) -> str:
        return user

    def has_mark(self, mark: str) -> bool:
        return True

    def answer(self, state, q: dict) -> tuple[dict, int]:
        answers, tokens = self.answer_many(state, {"q": q})
        return answers["q"], tokens

    # -- the translation ----------------------------------------------------------------------

    def translate(self, questions: dict) -> tuple[dict, dict]:
        """The questions as Jev takes them, and how to read each answer back: `tfu` becomes a
        `choice` over the two criteria and the shipped wording of the third answer; `scale` is
        `score`; a map of levels becomes a list, its keys kept for the way back."""
        out, back = {}, {}
        for key, q in questions.items():
            kind = self.kind_of(q)
            crit = q.get("criteria")
            if kind == "tfu":
                two = crit if isinstance(crit, dict) and len(crit) == 2 else {}
                vals = [v for _, v in self.two_criteria(two)] if two else []
                out[key] = {"type": "choice", "instructions": q.get("instructions"),
                            "criteria": {"true": vals[0] if len(vals) > 0 and vals[0] else CRITERIA_DEFAULT["true"],
                                         "false": vals[1] if len(vals) > 1 and vals[1] else CRITERIA_DEFAULT["false"],
                                         "unsure": UNSURE}}
                back[key] = ("tfu", None)
            elif kind == "score":
                keys = None
                if isinstance(crit, dict):
                    keys = [str(k) for k in crit]
                    crit = list(crit.values())
                out[key] = {"type": "score", "instructions": q.get("instructions"),
                            "criteria": crit}
                back[key] = ("score", keys)
            else:
                out[key] = {"type": kind, "instructions": q.get("instructions"),
                            "criteria": crit}
                back[key] = (kind, None)
        return out, back

    @staticmethod
    def read_back(a: dict, how: tuple) -> dict:
        kind, keys = how
        if kind == "tfu":
            p = {k: float(v) for k, v in a.get("probabilities", {}).items()}
            lead = max(p, key=p.get) if p else a.get("choice", "unsure")
            return {"type": "tfu", "tfu": lead, "probabilities": p,
                    "confidence": float(a.get("confidence", p.get(lead, 0.0))), "native": False}
        if kind == "score" and keys:
            rename = {str(i): k for i, k in enumerate(keys)}
            a = dict(a)
            for field in ("probabilities", "legend"):
                if isinstance(a.get(field), dict):
                    a[field] = {rename.get(k, k): v for k, v in a[field].items()}
        return a

    def answer_many(self, state, questions: dict) -> tuple[dict, int]:
        sent, back = self.translate(questions)
        r = self._s.post(f"{self.endpoint}/v1/systemone",
                         json={"state": state, "model": self.model, "questions": sent},
                         timeout=self.timeout)
        if r.status_code != 200:
            raise ApiError(r.status_code, _message(r.text))
        data = r.json()
        answers = {k: self.read_back(a, back[k]) for k, a in data.get("answers", {}).items()}
        self.upstream_model = data.get("model", self.model)
        return answers, int((data.get("usage") or {}).get("input_tokens", 0))
