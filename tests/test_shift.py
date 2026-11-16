"""The optional `shift` of prompt.json calibration: absent = the reading of every earlier checkpoint, present = true against false moved by it."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from typecastlm.reading import Reading


def reader(cal):
    r = Reading.__new__(Reading)
    r.prompt_cfg = {"calibration": cal}
    return r


def test_absent_shift_is_the_old_reading():
    r = reader({"verdict": {"temperature": 0.5}, "three_answers": {"temperature": 1.0}})
    z = {"true": 2.0, "false": 0.5}
    old = math.exp(2.0 / 0.5) / (math.exp(2.0 / 0.5) + math.exp(0.5 / 0.5))
    assert abs(r.read("noul", {**z, "unsure": 0.0}, {})["noul"] - old) < 1e-12
    p = r.read("tfu", {"true": 1.0, "false": 0.0, "unsure": -1.0}, {})["probabilities"]
    assert abs(sum(p.values()) - 1) < 1e-12


def test_shift_moves_the_odds_of_true_against_false():
    base, shifted = reader({"verdict": {"temperature": 1.0}}), reader({"verdict": {"temperature": 1.0, "shift": 0.8}})
    z = {"true": 1.0, "false": 0.3, "unsure": -2.0}
    p0, p1 = base.read("noul", z, {})["noul"], shifted.read("noul", z, {})["noul"]
    assert abs(math.log(p1 / (1 - p1)) - math.log(p0 / (1 - p0)) - 0.8) < 1e-9


def test_shift_leaves_unsure_alone():
    r = reader({"three_answers": {"temperature": 1.0, "shift": 1.0}}); b = reader({"three_answers": {"temperature": 1.0}})
    z = {"true": 1.0, "false": 0.0, "unsure": 0.5}
    p1, p0 = r.read("tfu", z, {})["probabilities"], b.read("tfu", z, {})["probabilities"]
    assert abs(math.log(p1["true"] / p1["false"]) - math.log(p0["true"] / p0["false"]) - 1.0) < 1e-9
    assert abs(math.log(p1["unsure"] / p1["false"]) - math.log(p0["unsure"] / p0["false"]) - 0.5) < 1e-9     # false lost s/2, so unsure gains s/2 against it


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
