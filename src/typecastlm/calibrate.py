"""Fitting the calibration on your own labelled rows.

Calibration adds no knowledge: it changes no ordering at all (macro AUC 0.817 before, 0.820
after). It changes two other things — where the threshold falls, and how confident the answer
sounds. Which is why the numbers have to be yours: they encode how often the answer is yes in
YOUR data and how often nothing decides it, and that belongs to the task, not to the model.

Three knobs, one job each:

    temperature       divides the logits; fixes confidence and calibration error, moves no decision
    shift on `false`  moves the threshold between yes and no
    shift on `unsure` decides how often the reader declines to answer

A softmax cannot tell a common shift from nothing, so there are two shifts for three answers.

    from typecastlm import Client, calibrate

    rows = [(c.noul(t, q, true=T, false=F).logits, gold) for t, gold in my_labelled]
    cal = calibrate(rows)              # {'temperature': ..., 'shift': [...]}
    c = Client(temperature=cal["temperature"], shift=tuple(cal["shift"][1:]))
"""
from __future__ import annotations

import math

LABELS = ("true", "false", "unsure")


def _softmax(v: list[float]) -> list[float]:
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def calibrate(rows: list[tuple[dict, str]], steps: int = 4000, lr: float = 0.05,
              fit_temperature: bool = True) -> dict:
    """Three numbers from labelled rows, each a pair `(logits, correct answer)`.

    `logits` is the dictionary an answer carries (`Answer.logits`); the correct answer is one of
    `true`, `false`, `unsure`. Cross-entropy is minimised. Two or three hundred rows are usually
    enough, because only three numbers are free.
    """
    if len(rows) < 30:
        raise ValueError(f"thirty rows is the floor, got {len(rows)}")
    Z = [[float(z[k]) for k in LABELS] for z, _ in rows]
    try:
        Y = [LABELS.index(g) for _, g in rows]
    except ValueError as e:
        raise ValueError(f"every label must be one of {LABELS}") from e
    lt, b, n = 0.0, [0.0, 0.0, 0.0], len(Z)
    for _ in range(steps):
        T = math.exp(lt)
        gb, glt = [0.0, 0.0, 0.0], 0.0
        for z, y in zip(Z, Y):
            p = _softmax([(x + s) / T for x, s in zip(z, b)])
            for k in range(3):
                d = p[k] - (1.0 if y == k else 0.0)
                gb[k] += d / n
                glt += d * (-(z[k] + b[k]) / T) / n
        for k in (1, 2):
            b[k] -= lr * gb[k]
        if fit_temperature:
            lt -= lr * glt
    return {"temperature": math.exp(lt), "shift": b, "rows": n,
            "note": "this changes no ordering — only the threshold and the confidence"}
