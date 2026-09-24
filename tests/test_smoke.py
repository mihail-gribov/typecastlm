"""Tests that need no weights: the shapes, the readings, and the weight of an import."""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT.parent / "model"          # the model repository, when it is checked out beside us

# Temperatures as inputs, not as truth: every expectation below is computed from these, so the
# tests say what the code does with a calibration rather than what this checkpoint's happens to be.
CAL = {"verdict": {"temperature": 1.3}, "three_answers": {"temperature": 2.85},
       "choice": {"temperature": 1.75}, "scale": {"temperature": 3.9}}


def test_import_costs_nothing_heavy():
    """A client must not drag in a deep-learning stack. This is the whole point of the package."""
    for mod in ("torch", "transformers"):
        sys.modules.pop(mod, None)
    import typecastlm

    assert {"Client", "Answer", "calibrate"} <= set(typecastlm.__all__)
    assert not {"torch", "transformers"} & set(sys.modules)


def _client(logits, kind="noul"):
    from typecastlm import Client

    c = Client(endpoint="http://x")
    c.calibration = CAL
    c._post = lambda body: ({"model": "m",
                             "answers": {"q": {"type": kind, "logits": logits}},
                             "usage": {"input_tokens": 9, "output_tokens": 0},
                             "calibration": CAL}, 1.0)
    return c


def test_a_bare_host_gets_the_route():
    """`TYPECASTLM_ENDPOINT=http://localhost:8000` is what anyone writes after starting the
    service. Posting to the bare host answers 404, so the client adds the route itself."""
    from typecastlm.remote import ROUTE, _route

    assert _route("http://localhost:8000") == "http://localhost:8000" + ROUTE
    assert _route("http://localhost:8000/") == "http://localhost:8000" + ROUTE
    assert _route("https://gw.example.com/typecast" + ROUTE) == \
        "https://gw.example.com/typecast" + ROUTE      # a path of its own is left alone
    assert _route("") == ""


def test_noul_is_a_softmax_over_two_answers():
    """The third output is another mode's, read at another temperature. It must not leak in."""
    z = {"true": 2.2, "false": 0.7, "unsure": 9.9}     # a third answer loud enough to be noticed
    a = _client(z).noul("s", "q?", true="t", false="f")

    t = CAL["verdict"]["temperature"]
    assert a.prob == pytest.approx(1 / (1 + math.exp(-(2.2 - 0.7) / t)))
    assert a.margin == pytest.approx((2.2 - 0.7) / t)
    assert not hasattr(a, "unknown")


def test_tfu_reads_all_three_at_its_own_temperature():
    z = {"true": 2.2, "false": 0.7, "unsure": 1.4}
    p = _client(z, "tfu").tfu("s", "q?", true="t", false="f").p

    t = CAL["three_answers"]["temperature"]
    e = {k: math.exp(v / t) for k, v in z.items()}
    s = sum(e.values())
    assert p == pytest.approx({k: v / s for k, v in e.items()})
    assert sum(p.values()) == pytest.approx(1.0)


def test_the_service_answers_in_the_shape_of_the_api_it_copies():
    from typecastlm.server import Reader

    r = object.__new__(Reader)
    r.prompt_cfg = {"calibration": CAL}

    noul = r.read("noul", {"true": 3.0, "false": 0.0, "unsure": -1.0}, {})
    assert set(noul) == {"type", "noul"} and 0 < noul["noul"] < 1

    tfu = r.read("tfu", {"true": 3.0, "false": 0.0, "unsure": -1.0}, {})
    assert set(tfu) == {"type", "tfu", "probabilities", "confidence"}

    choice = r.read("choice", {"a": 1.0, "b": 2.0}, {})
    assert choice["choice"] == "b" and choice["confidence"] == choice["probabilities"]["b"]

    legend = {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    score = r.read("score", {"0": -3.0, "1": 4.0, "2": 1.0}, legend)
    assert set(score) == {"type", "score", "legend", "probabilities", "confidence"}
    # The level is a mean over positions, so it lands between levels and can pass 1.
    p = score["probabilities"]
    assert score["score"] == pytest.approx(p["1"] + 2 * p["2"])
    assert 0 <= score["score"] <= len(legend) - 1


def test_noul_and_the_verdict_of_the_local_reader_are_the_same_number():
    """Two implementations of one mode: the client over HTTP and the reader in-process."""
    torch = pytest.importorskip("torch")                                     # noqa: F841
    from typecastlm.local import Reader

    r = object.__new__(Reader)
    r.cal, r.names = CAL, ["true", "false", "unsure"]
    z = {"true": 2.2, "false": 0.7, "unsure": 1.4}

    assert (_client(z).noul("s", "q?", true="t", false="f").prob
            == pytest.approx(r._verdict(list(z.values()))["p"]["yes"]))


@pytest.mark.skipif(not (pathlib.Path(__file__).resolve().parents[2] / "model").is_dir(),
                    reason="the model repository is not checked out beside this one")
def test_the_shipped_reader_has_not_drifted_from_the_package():
    """The weights ship a copy of the reader, and a copy nobody checks goes stale."""
    assert subprocess.run([sys.executable, str(ROOT / "scripts/sync_model.py"), "--check"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not (pathlib.Path(__file__).resolve().parents[2] / "model").is_dir(),
                    reason="the model repository is not checked out beside this one")
def test_the_shipped_checkpoint_is_calibrated_for_every_mode():
    """A partial calibration reads one mode at another's temperature, silently."""
    cal = json.loads((MODEL / "prompt.json").read_text(encoding="utf-8"))["calibration"]
    assert {"verdict", "three_answers", "choice", "scale"} <= set(cal)
    assert all(cal[m]["temperature"] > 0 for m in ("verdict", "three_answers", "choice", "scale"))


def test_the_shape_check_is_written_down():
    """The server must refuse a checkpoint that does not have the three outputs the clients read."""
    import ast

    tree = ast.parse((ROOT / "src/typecastlm/server.py").read_text(encoding="utf-8"))
    reader = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and n.name == "Reader")
    assert any(isinstance(n, ast.FunctionDef) and n.name == "check" for n in reader.body)
    expected = next(n for n in reader.body
                    if isinstance(n, ast.Assign) and n.targets[0].id == "EXPECTED")
    assert [c.value for c in expected.value.elts] == ["true", "false", "unsure"]
