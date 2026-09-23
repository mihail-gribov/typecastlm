"""Подгонка калибровки по своим размеченным строкам.

Калибровка не добавляет знания: порядок документов она не меняет вовсе (макро-AUC 0.817 против
0.820 до и после). Она меняет ДВЕ вещи — куда попадает порог и насколько уверенно звучит ответ.
Поэтому числа нужны свои: они кодируют, как часто в ВАШИХ данных ответ «да» и как часто решать
нечем, а это свойство задачи, а не модели.

Ручек ровно три, и каждая делает своё:

    температура   делит логиты; правит уверенность и ECE, решение не двигает совсем
    сдвиг «нет»   двигает порог между «да» и «нет»
    сдвиг «не знаю»  решает, как часто читатель отказывается отвечать

Общий сдвиг софтмакс не различает, поэтому сдвигов именно два, а не три.

    from typecastlm import Client, calibrate
    rows = [(client.noul(t, q, true=T, false=F).logits, gold) for t, gold in my_labelled]
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
    """Три числа по размеченным строкам: `(логиты, правильный ответ)`.

    `логиты` — словарь из ответа (`Answer.logits`), `правильный ответ` — одно из `true`, `false`,
    `unsure`. Минимизируется кросс-энтропия; двести-триста строк обычно хватает, потому что
    свободных чисел всего три.
    """
    if len(rows) < 30:
        raise ValueError(f"тридцать строк — нижний предел, дано {len(rows)}")
    Z = [[float(z[k]) for k in LABELS] for z, _ in rows]
    Y = [LABELS.index(g) for _, g in rows]
    if any(y is None for y in Y):
        raise ValueError(f"метка должна быть одной из {LABELS}")
    lt, b = 0.0, [0.0, 0.0, 0.0]
    n = len(Z)
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
    return {"temperature": math.exp(lt), "shift": b,
            "rows": n, "note": "порядок документов это не меняет — только порог и уверенность"}
