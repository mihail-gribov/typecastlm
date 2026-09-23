"""The decision interface, field for field the one this reader stands in for.

Same call, same body, same answer shape — so a harness written against the hosted service runs
against this one by swapping the client and nothing else:

    client.ask(state, questions)   -> {"answers": {...}, "input_tokens": int}
    client.noul(state, instructions, true=..., false=...) -> Answer(prob, margin, ms, ...)

Question types and their fields:

    noul    {"noul": probability of the `true` criterion}
    choice  {"probabilities": {name: p}, "confidence": p of the leading one, "choice": its name}
    score   {"score": expected level, "confidence": ..., "probabilities": {"0": p, "1": p, ...}}

**The one real limit.** The carried head has three rows: a criterion, its opposite, and the
outcome the prompt never offers. A question here is therefore binary with `unknown` as a possible
answer — `choice` takes two or three criteria and `score` a scale of two or three levels. A longer
list is refused rather than quietly folded, because folding it would return numbers that look fine
and mean nothing.

**The bill differs even though the shape does not.** Every question is its own forward pass: the
state is read again for each one, since nothing about it is kept between them. A hosted service
bills the state once for a whole bundle. What is cheap here is the other direction — the same
question over many states, which is what `batch` is for.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .model import DEFAULT_MODEL, TypecastLM


@dataclass
class Answer:
    """What `noul` returns: the probability, its log-odds, and what the call cost."""

    prob: float
    margin: float
    ms: float
    input_tokens: int
    model: str


class Client:
    """Holds the reader and answers in the shape of the service.

    `model` is a repo id or a local directory; the weights are loaded once and stay resident, so
    the first call is slow and the rest are not.
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto", dtype: str = "bfloat16",
                 prompt: dict | None = None, **_ignored):
        self.model = str(model)
        self.reader = TypecastLM(model, device=device, dtype=dtype, prompt=prompt)

    # --- criteria -----------------------------------------------------------------------------
    def _pair(self, crit) -> tuple[tuple[str, str], list[str]]:
        if crit is None:
            d = self.reader.prompt_cfg["criteria_default"]
            return (d["true"], d["false"]), list(d)
        if not isinstance(crit, dict) or not 2 <= len(crit) <= 3:
            raise ValueError(
                "this reader answers a binary question with a possible unknown, so a question "
                f"takes two or three criteria; got {crit!r}")
        names = list(crit)
        return (crit[names[0]], crit[names[1]]), names

    # --- one question -------------------------------------------------------------------------
    def _one(self, state: str, q: dict) -> dict:
        kind = q.get("type", "noul")
        if kind == "score":
            levels = q["criteria"]
            if not isinstance(levels, (list, tuple)) or not 2 <= len(levels) <= 3:
                raise ValueError("a scale here takes two or three levels, lowest first")
            # A scale runs from denial to assertion, and the head's rows do not: the middle level
            # of a three-step scale is the row the prompt never offers.
            order = [1, 2, 0] if len(levels) == 3 else [1, 0]
            v = list(self.reader.ask(state, q["instructions"], (levels[-1], levels[0]))[0]
                     .values())
            p = {str(i): v[order[i]] for i in range(len(levels))}
            s = sum(p.values()) or 1.0
            p = {k: x / s for k, x in p.items()}
            lead = max(p, key=p.get)
            return {"score": sum(int(k) * x for k, x in p.items()), "confidence": p[lead],
                    "probabilities": p}

        (t, f), names = self._pair(q.get("criteria"))
        a = self.reader.ask(state, q["instructions"], (t, f),
                            labels=names if len(names) == 3 else None)[0]
        v = list(a.values())
        if kind == "noul":
            decided = v[0] + v[1]
            return {"noul": v[0] / decided if decided > 0 else 0.5}
        if kind == "choice":
            p = dict(a) if len(names) == 3 else dict(zip(names, v[:2]))
            s = sum(p.values()) or 1.0
            p = {k: x / s for k, x in p.items()}
            lead = max(p, key=p.get)
            return {"probabilities": p, "confidence": p[lead], "choice": lead}
        raise ValueError(f"unknown question type {kind!r}; known: noul, choice, score")

    # --- the service surface ------------------------------------------------------------------
    def ask(self, state: str, questions: dict) -> dict:
        """Several questions about one state. Here each one is read separately; see the module."""
        answers, tokens = {}, 0
        for key, q in questions.items():
            answers[key] = self._one(state, q)
            tokens += self.reader.last_tokens
        return {"answers": answers, "input_tokens": tokens}

    def noul(self, state: str, instructions: str, true: str | None = None,
             false: str | None = None) -> Answer:
        """One closed question. `margin` is the log-odds, so a saturated probability still ranks."""
        q: dict = {"type": "noul", "instructions": instructions}
        if true is not None or false is not None:
            q["criteria"] = {"true": true or "", "false": false or ""}
        t0 = time.perf_counter()
        p = float(self._one(state, q)["noul"])
        ms = (time.perf_counter() - t0) * 1000.0
        eps = 1e-6
        pc = min(max(p, eps), 1 - eps)
        return Answer(prob=p, margin=math.log(pc / (1 - pc)), ms=ms,
                      input_tokens=self.reader.last_tokens, model=self.model)

    # --- what the hosted shape has no room for ------------------------------------------------
    def batch(self, states: list[str], instructions: str, true: str | None = None,
              false: str | None = None, batch_size: int = 4) -> list[Answer]:
        """The same question over many states — the direction this reader is cheap in.

        Not part of the mirrored surface: a hosted service has no reason to offer it, and here it
        is the whole point, since the cost is one reading of each state either way.
        """
        (t, f), _ = self._pair({"true": true, "false": false} if true or false else None)
        out = self.reader.ask(states, instructions, (t, f), batch_size=batch_size)
        res = []
        for a in out:
            v = list(a.values())
            decided = v[0] + v[1]
            p = v[0] / decided if decided > 0 else 0.5
            pc = min(max(p, 1e-6), 1 - 1e-6)
            res.append(Answer(prob=p, margin=math.log(pc / (1 - pc)), ms=0.0,
                              input_tokens=self.reader.last_tokens // max(1, len(states)),
                              model=self.model))
        return res
