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

    choice = r.read("choice", {"a": 1.0, "b": 2.0}, {}, ["A", "B"])
    assert choice["choice"] == "b" and choice["confidence"] == choice["probabilities"]["b"]
    # The mark is the position, so the answer has to say which option got which: reordering the
    # options changes the reading, and a caller cannot see that from names alone.
    assert choice["marks"] == {"a": "A", "b": "B"}

    legend = {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    score = r.read("score", {"0": -3.0, "1": 4.0, "2": 1.0}, legend, ["0", "1", "2"])
    assert set(score) == {"type", "score", "legend", "probabilities", "confidence", "marks"}
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
    # The three outputs are named once, in the reading both readers share.
    tree = ast.parse((ROOT / "src/typecastlm/reading.py").read_text(encoding="utf-8"))
    reading = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == "Reading")
    expected = next(n for n in reading.body
                    if isinstance(n, ast.Assign) and n.targets[0].id == "EXPECTED")
    assert [c.value for c in expected.value.elts] == ["true", "false", "unsure"]


def test_every_field_the_api_allows_becomes_text():
    """The Jev API takes a string, an object, an array or nothing wherever prose goes. An object
    is read as JSON, nothing as an empty string — never a 500."""
    from typecastlm.server import Reader

    assert Reader.as_text("plain") == "plain"
    assert Reader.as_text(None) == ""
    assert Reader.as_text({"task": "spam?"}) == '{"task": "spam?"}'
    assert Reader.as_text(["a", 1]) == '["a", 1]'
    assert Reader.as_text({"é": "ü"}) == '{"é": "ü"}'      # not escaped: the model reads it


def test_the_answer_names_the_checkpoint_never_a_path():
    from typecastlm.server import short_name

    assert short_name("mihailgribov/typecastlm-qwen3.5-3.8b") == "typecastlm-qwen3.5-3.8b"
    assert short_name("/models/typecastlm/") == "typecastlm"
    assert short_name("typecastlm") == "typecastlm"


def test_the_line_for_the_model_has_an_end():
    """One request holds the model; `queue` may wait; the next is told to come back, with a
    Retry-After of the line's length times the average request."""
    import threading

    from typecastlm.server import Busy, Gate

    g = Gate(queue=1)
    with g:                                     # 1 running
        assert g.load()["busy"] and g.load()["waiting"] == 0
        started, release = threading.Event(), threading.Event()

        def waiter():
            with g:
                pass

        with g._mu:                             # a second one, counted as waiting
            g.inflight += 1
        with pytest.raises(Busy) as e:          # the third is one too many
            with g:
                pass
        assert e.value.wait >= 1
        with g._mu:
            g.inflight -= 1
    assert not g.load()["busy"] and g.load()["served"] == 1
    with g:                                     # the lock was released
        pass
    assert g.load()["served"] == 2 and g.load()["avg_ms"] >= 0


def test_flags_read_the_environment(monkeypatch):
    """The container runs `typecastlm-serve` with no arguments: everything comes from
    TYPECASTLM_*. An empty variable counts as unset, so a compose file may pass them all."""
    from typecastlm.server import env

    monkeypatch.setenv("TYPECASTLM_PORT", "8077")
    monkeypatch.setenv("TYPECASTLM_API_KEY", "")
    assert env("PORT", "8000") == "8077"
    assert env("API_KEY", "") == ""
    assert env("QUEUE", "32") == "32"


def test_a_refused_question_carries_its_key():
    """A bundle of twenty is refused with the one that is wrong, not with 'a question'."""
    from typecastlm.server import QuestionError, Reader

    r = object.__new__(Reader)
    r.answer = lambda state, q: (_ for _ in ()).throw(ValueError("bad"))
    with pytest.raises(QuestionError) as e:
        r.answer_many("s", {"fine": {"type": "noul"}})
    assert e.value.key == "fine" and str(e.value) == "bad"
    with pytest.raises(QuestionError) as e:
        r.answer_many("s", {"broken": "not an object"})
    assert e.value.key == "broken"


def test_the_committed_schema_is_the_one_the_service_renders():
    """docs/openapi.json is read by people and generators who never start the service, so it
    must be what the service would say. Regenerate with scripts/export_openapi.py."""
    pytest.importorskip("fastapi")
    sys.path.insert(0, str(ROOT / "scripts"))
    import export_openapi

    want = json.dumps(export_openapi.schema(), indent=1, ensure_ascii=False) + "\n"
    have = (ROOT / "docs" / "openapi.json").read_text(encoding="utf-8")
    assert have == want, "docs/openapi.json is stale: run scripts/export_openapi.py"


def test_the_request_schema_takes_what_the_api_it_copies_takes():
    """A question of each type, in the loosest form the Jev API allows: object instructions, a
    missing description, a bare yes/no question, `scale` for `score`."""
    pytest.importorskip("pydantic")
    from typecastlm.schema import SystemOneRequest

    req = SystemOneRequest(state={"subject": "Duplicate charge"}, questions={
        "spam": {"type": "noul", "instructions": "Is this spam?"},
        "sure": {"type": "tfu", "instructions": {"task": "Is the claim supported?"},
                 "criteria": {"true": "supported", "false": "contradicted"}},
        "topic": {"type": "choice", "criteria": {"billing": None, "shipping": "delivery"}},
        "mood": {"type": "scale", "instructions": "How angry?", "criteria": ["calm", "angry"]},
    })
    dumped = {k: q.model_dump() for k, q in req.questions.items()}
    assert dumped["spam"] == {"type": "noul", "instructions": "Is this spam?", "criteria": None}
    assert dumped["topic"]["criteria"] == {"billing": None, "shipping": "delivery"}
    assert dumped["mood"]["type"] == "scale" and dumped["mood"]["criteria"] == ["calm", "angry"]
    with pytest.raises(ValueError):
        SystemOneRequest(state="x", questions={})                       # nothing to ask
    with pytest.raises(ValueError):
        SystemOneRequest(state="x", questions={"q": {"type": "guess"}})  # not a type
    with pytest.raises(ValueError):
        SystemOneRequest(state="x", questions={"q": {"type": "choice"}})  # options missing


def _embedding_reader(vector):
    """An EmbeddingReader with no server: the prompt and head from the model checkout beside
    us, and a vector supplied by the test."""
    from typecastlm.embedding import EmbeddingReader

    r = object.__new__(EmbeddingReader)
    r.prompt_cfg = json.loads((MODEL / "prompt.json").read_text(encoding="utf-8"))
    r.rows = {"true": [[1.0, 0.0, 0.0]], "false": [[0.0, 1.0, 0.0]], "unsure": [[0.0, 0.0, 1.0]],
              "mark_A": [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], "mark_B": [[0.0, 1.0, 0.0]],
              "mark_C": [[0.0, 0.0, 1.0]]}
    r.hidden, r.frame, r.max_state_tokens = 3, r.prompt_cfg.get("chat_frame", ""), 100
    from typecastlm.embedding import CHAT_FRAME

    r.frame = r.frame or CHAT_FRAME
    r.served, r.name, r.endpoint = "m", "m", "http://x"
    r._tokens, r._can_tokenize = None, True
    r.vector = lambda text: vector
    r.tokenize = lambda text: list(range(len(text.split())))
    return r


@pytest.mark.skipif(not (pathlib.Path(__file__).resolve().parents[2] / "model").is_dir(),
                    reason="the model repository is not checked out beside the package")
def test_the_launcher_reader_renders_the_measured_prompt():
    """The chat frame is the tokenizer's rendering, verbatim: system, user, assistant with
    thinking off and the generation prompt — then the tail. A prompt that differs by a character
    is another model, so this is the string exactly."""
    r = _embedding_reader([0.0, 0.0, 0.0])
    text, keys, marks, _ = r.prepare("The pipe burst.", {
        "type": "noul", "instructions": "Sudden?", "criteria": {"true": "sudden", "false": "slow"}})
    assert text.startswith("<|im_start|>system\n" + r.prompt_cfg["system"] + "<|im_end|>\n"
                           "<|im_start|>user\n" + r.prompt_cfg["format_two"])
    assert text.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                         + r.prompt_cfg["tail"])
    assert "<state>\nThe pipe burst.\n</state>" in text and marks is None
    assert keys == ["true", "false", "unsure"]


@pytest.mark.skipif(not (pathlib.Path(__file__).resolve().parents[2] / "model").is_dir(),
                    reason="the model repository is not checked out beside the package")
def test_the_launcher_reader_reads_the_head_like_the_service():
    """39 dot products, the stronger surface form of a mark, the mode's temperature — and the
    same answer body the service returns, so `Client(transport=…)` sees no difference."""
    from typecastlm import Client

    r = _embedding_reader([2.0, 0.5, -1.0])
    cal = r.prompt_cfg["calibration"]
    a, n = r.answer("s", {"type": "noul", "instructions": "?",
                          "criteria": {"true": "a", "false": "b"}})
    t = cal["verdict"]["temperature"]
    assert a["logits"] == {"true": 2.0, "false": 0.5, "unsure": -1.0}
    assert a["noul"] == pytest.approx(1 / (1 + math.exp(-(2.0 - 0.5) / t))) and n > 0
    c, _ = r.answer("s", {"type": "choice", "instructions": "?",
                          "criteria": {"x": "1", "y": "2", "z": "3"}})
    assert c["choice"] == "x" and c["marks"] == {"x": "A", "y": "B", "z": "C"}
    assert c["logits"]["x"] == 2.0                       # the stronger of A's two forms
    body = r({"state": "s", "questions": {"q": {"type": "tfu", "instructions": "?"}}})
    assert set(body) == {"model", "answers", "usage", "calibration"}
    assert body["answers"]["q"]["type"] == "tfu" and body["usage"]["output_tokens"] == 0
    v = Client(transport=r).noul("s", "?", true="a", false="b")
    assert v.prob == pytest.approx(a["noul"]) and v.margin == pytest.approx((2.0 - 0.5) / t)


def _jev(answers, usage=None):
    """A client against a Jev-shaped server: probabilities, no logits, no calibration."""
    from typecastlm import Client

    c = Client.jev(api_key="k")
    sent = []

    def post(body):
        sent.append(body)
        return ({"model": "jev-1.13.0", "answers": answers,
                 "usage": usage or {"input_tokens": 120, "output_tokens": 12}}, 1.0)

    c._post = post
    return c, sent


def test_jev_is_a_backend_with_its_own_address_key_and_model(monkeypatch):
    from typecastlm import Client
    from typecastlm.remote import JEV_ENDPOINT, JEV_MODEL

    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    c = Client.jev()
    assert c.api == "jev" and c.base == JEV_ENDPOINT and c.model == JEV_MODEL
    assert c.key == "from-env" and c.timeout == 30.0 and c.native_tfu is False
    assert Client("https://api.typesafe.ai", api_key="k").api == "jev"     # by its host
    assert Client(host="gpu", port=8090, api="typecastlm").base == "http://gpu:8090"
    assert Client("localhost:8000", api="typecastlm").endpoint == "http://localhost:8000/v1/systemone"
    with pytest.raises(ValueError):
        Client("http://x", api="grpc")


def test_jev_answers_from_probabilities_alone():
    c, sent = _jev({"q": {"type": "noul", "noul": 0.95}})
    a = c.noul("s", "covered?", true="yes", false="no")
    assert a.prob == 0.95 and a.logits is None and a.model == "jev-1.13.0"
    assert a.margin == pytest.approx(math.log(0.95 / 0.05))
    assert sent[0]["model"] == "jev-latest" and sent[0]["questions"]["q"]["type"] == "noul"


def test_tfu_on_jev_is_a_choice_with_a_third_option():
    from typecastlm.remote import UNSURE

    c, sent = _jev({"q": {"type": "choice", "choice": "unsure", "confidence": 0.9,
                          "probabilities": {"true": 0.05, "false": 0.05, "unsure": 0.9}}})
    t = c.tfu("s", "supported?", true="supported", false="contradicted")
    q = sent[0]["questions"]["q"]
    assert q["type"] == "choice" and list(q["criteria"]) == ["true", "false", "unsure"]
    assert q["criteria"]["unsure"] == UNSURE
    assert t.native is False and t.verdict == "unsure" and t.p["unsure"] == 0.9
    assert t.logits is None


def test_levels_go_as_a_list_and_come_back_under_your_names():
    c, sent = _jev({"q": {"type": "score", "score": 1.03, "confidence": 0.94,
                          "legend": {"0": "calm", "1": "annoyed", "2": "angry"},
                          "probabilities": {"0": 0.03, "1": 0.91, "2": 0.06}}})
    s = c.scale("s", "how angry?", {"calm": "calm", "annoyed": "annoyed", "angry": "angry"})
    assert sent[0]["questions"]["q"]["criteria"] == ["calm", "annoyed", "angry"]
    assert list(s.p) == ["calm", "annoyed", "angry"] and s.verdict == "annoyed"
    assert s.score == pytest.approx(0.91 + 2 * 0.06)
    s2 = c.scale("s", "how angry?", ["calm", "annoyed", "angry"])          # a list works too
    assert list(s2.p) == ["0", "1", "2"]


def test_the_error_carries_the_message_whatever_its_shape():
    from typecastlm.remote import _message

    assert _message('{"detail": {"error_type": "api_usage_error", "message": "Unknown model: x"}}') \
        == "Unknown model: x"
    assert _message('{"detail": [{"loc": ["body", "questions", "q", "criteria"], '
                    '"msg": "Input should be a valid list", "type": "list_type"}]}') \
        == "questions.q.criteria: Input should be a valid list"
    assert _message('{"detail": "no questions"}') == "no questions"
    assert _message("<html>gateway</html>") == "<html>gateway</html>"
