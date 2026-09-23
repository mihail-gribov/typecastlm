"""The decision interface: compatible with the hosted one, and wider by one number.

Compatible means a harness written against the service runs here by swapping the client — same
call, same body, same fields:

    client.ask(state, questions)   -> {"answers": {...}, "input_tokens": int}
    client.noul(state, instructions, true=..., false=...) -> Answer(prob, margin, ms, ...)

**Version zero: one question per call, and only `noul`.** A question of type `choice` or `score` is refused, not approximated:
this reader has three fixed labels, and folding named options or a described scale onto them would
return numbers that look like answers to the question that was asked while answering a different
one. When those modes arrive they will be measured first.

**And the answer carries one number the service does not.** `unknown` is never asked for — the
prompt offers two criteria and nothing else — but the third label is read anyway, and it reports
how much of the state points at "nothing here decides it". It rides beside `noul` as an extension:
anything reading the compatible fields ignores it, and anything that wants it does not have to ask
a second question to get it.

`noul` itself stays exactly what it is over there: the probability of the `true` criterion among
the two that decide the question, so a state that decides nothing does not drag it to a half.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .model import DEFAULT_MODEL, TypecastLM

IMPLEMENTED = ("noul",)


@dataclass
class Answer:
    """What `noul` returns. Every field but `unknown` is the service's; `unknown` is ours."""

    prob: float
    margin: float
    ms: float
    input_tokens: int
    model: str
    unknown: float = 0.0


class Client:
    """Holds the reader and answers in the shape of the service.

    `model` is a repo id or a local directory; the weights load once and stay resident, so the
    first call is slow and the rest are not.
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto", dtype: str = "bfloat16",
                 prompt: dict | None = None, **_ignored):
        self.model = str(model)
        self.reader = TypecastLM(model, device=device, dtype=dtype, prompt=prompt)

    # --- criteria -----------------------------------------------------------------------------
    def _criteria(self, crit) -> tuple[str, str]:
        """The wording of the two criteria. `unknown` is not among them and never is: it is not
        something the caller states, it is what the reader has left over."""
        if crit is None:
            d = self.reader.prompt_cfg["criteria_default"]
            return d["true"], d["false"]
        if not isinstance(crit, dict) or len(crit) != 2:
            raise ValueError(
                "a question takes exactly two criteria — the two sides of the closed question. "
                f"got {crit!r}. The third answer is read without being asked for.")
        a, b = crit.values()
        return a, b

    def _one(self, state: str, q: dict) -> dict:
        kind = q.get("type", "noul")
        if kind not in IMPLEMENTED:
            raise NotImplementedError(
                f"question type {kind!r} is not implemented here; this reader answers {IMPLEMENTED}"
                " — a binary question, with `unknown` coming back as an extra field")
        t, f = self._criteria(q.get("criteria"))
        p = self.reader.ask(state, q["instructions"], (t, f))[0]
        v = list(p.values())
        decided = v[0] + v[1]
        return {"noul": v[0] / decided if decided > 0 else 0.5,
                "unknown": v[2] if len(v) > 2 else 0.0}

    # --- the service surface ------------------------------------------------------------------
    def ask(self, state: str, questions: dict) -> dict:
        """One question about one state, in the shape a bundle would take.

        The body is the service's, so a harness that sends a bundle of one runs unchanged. More
        than one is refused rather than served: over there a bundle is cheap because the state is
        read once for all of it, and here it would be one reading per question — the same shape
        with none of the saving. Version zero answers one question at a time and says so.
        """
        if len(questions) != 1:
            raise NotImplementedError(
                f"version zero answers one question per call, got {len(questions)}. A bundle is "
                "cheap where the state is read once for all of it; here each question reads the "
                "state again, so the shape would promise a saving that does not exist.")
        key, q = next(iter(questions.items()))
        return {"answers": {key: self._one(state, q)},
                "input_tokens": self.reader.last_tokens}

    def noul(self, state: str, instructions: str, true: str | None = None,
             false: str | None = None) -> Answer:
        """One closed question. `margin` is the log-odds, so a saturated probability still ranks."""
        q: dict = {"type": "noul", "instructions": instructions}
        if true is not None or false is not None:
            q["criteria"] = {"true": true or "", "false": false or ""}
        t0 = time.perf_counter()
        a = self._one(state, q)
        ms = (time.perf_counter() - t0) * 1000.0
        p = float(a["noul"])
        pc = min(max(p, 1e-6), 1 - 1e-6)
        return Answer(prob=p, margin=math.log(pc / (1 - pc)), ms=ms,
                      input_tokens=self.reader.last_tokens, model=self.model,
                      unknown=float(a["unknown"]))

    # --- what the hosted shape has no room for ------------------------------------------------
    def batch(self, states: list[str], instructions: str, true: str | None = None,
              false: str | None = None, batch_size: int = 4) -> list[Answer]:
        """The same question over many states — the direction this reader is cheap in.

        Not part of the compatible surface: over there the state is the billed unit and a sweep
        over states is just many calls. Here each state is read once either way, so the sweep is
        the natural shape.
        """
        t, f = self._criteria({"true": true, "false": false} if true or false else None)
        out = self.reader.ask(states, instructions, (t, f), batch_size=batch_size)
        res = []
        for p in out:
            v = list(p.values())
            decided = v[0] + v[1]
            prob = v[0] / decided if decided > 0 else 0.5
            pc = min(max(prob, 1e-6), 1 - 1e-6)
            res.append(Answer(prob=prob, margin=math.log(pc / (1 - pc)), ms=0.0,
                              input_tokens=self.reader.last_tokens // max(1, len(states)),
                              model=self.model, unknown=v[2] if len(v) > 2 else 0.0))
        return res
