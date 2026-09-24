"""Fitting a temperature on your own labelled rows — one per mode.

A temperature adds no knowledge: it changes no ordering at all, so a ranking and every AUC stay
exactly as they were. It changes one thing — whether the probability means what it says. Which is
why the number has to be yours: it corrects the mismatch between how confident the reader sounds
and how hard YOUR data is, and that is a property of the data, not of the weights. The same
checkpoint needs none on an easy pool and a strong one on a hard one.

Per-answer shifts were removed in 1.0.0. They were needed while the third row of the head lived on
its own scale; it is now a direction computed from data and scaled to the other two, and fitting
shifts on top buys 0.005 of calibration error and no accuracy.

    from typecastlm import Client, calibrate

    rows = [(c.noul(t, q, true=T, false=F).logits, gold) for t, gold in my_labelled]
    T = calibrate(rows, keys=("true", "false"))["temperature"]      # the two `noul` decides over
    c = Client(temperature=T)

The same function serves every mode: pass the logits that mode returned and the answer that was
right. `keys` says which of them the mode reads — for `noul` that is the two criteria, since the
third answer belongs to `tfu` and carries a temperature of its own.
"""
from __future__ import annotations

import math


def _softmax(v: list[float]) -> list[float]:
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def _ece(conf: list[float], ok: list[bool], bins: int = 10) -> float:
    n = len(conf)
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j in range(n) if lo <= conf[j] < hi]
        if idx:
            c = sum(conf[j] for j in idx) / len(idx)
            a = sum(ok[j] for j in idx) / len(idx)
            total += abs(c - a) * len(idx) / n
    return total


def calibrate(rows: list[tuple[dict, str]], grid: tuple[float, float, float] = (0.6, 4.0, 0.05),
              keys: tuple[str, ...] | list[str] | None = None) -> dict:
    """One temperature from labelled rows, each a pair `(logits, correct answer)`.

    `logits` is the dictionary an answer carries (`Answer.logits`, `Choice.logits`); the correct
    answer is one of its keys. `keys` restricts the softmax to the answers the mode is read over,
    which matters for `noul`: its logits carry the third answer too, and fitting over all three
    would fit `tfu`'s temperature instead. The temperature is chosen to minimise calibration
    error — NOT cross-entropy, which pulls towards numbers that read worse: on our own
    three-answer mode it picked 5.63 where 2.85 is right, and made the error larger, not smaller.

    Two or three hundred rows are enough, because one number is free.
    """
    if len(rows) < 30:
        raise ValueError(f"thirty rows is the floor, got {len(rows)}")
    if keys is not None:
        missing = [k for k in keys if k not in rows[0][0]]
        if missing:
            raise ValueError(f"{missing} not among the answers {list(rows[0][0])}")
        rows = [({k: z[k] for k in keys}, g) for z, g in rows]
        if any(g not in keys for _, g in rows):
            raise ValueError(f"every correct answer must be one of {list(keys)}")
    keys = list(rows[0][0])
    for z, g in rows:
        if list(z) != keys:
            raise ValueError("every row must carry the same answers in the same order")
        if g not in z:
            raise ValueError(f"{g!r} is not one of {keys}")
    Z = [[float(z[k]) for k in keys] for z, _ in rows]
    Y = [keys.index(g) for _, g in rows]

    def at(T: float) -> float:
        conf, ok = [], []
        for z, y in zip(Z, Y):
            p = _softmax([x / T for x in z])
            top = max(range(len(p)), key=lambda i: p[i])
            conf.append(p[top])
            ok.append(top == y)
        return _ece(conf, ok)

    lo, hi, step = grid
    T = min((lo + i * step for i in range(int((hi - lo) / step) + 1)), key=at)
    return {"temperature": float(T), "calibration_error": float(at(T)),
            "calibration_error_before": float(at(1.0)), "rows": len(rows),
            "note": "changes no ordering — only what the probability means"}
