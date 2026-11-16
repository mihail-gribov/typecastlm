"""The `decision` task of transformers' pipeline: registration and the shape of an input. No weights are loaded here; the reading itself is
checked against the reader with `python3 tests/live_pipeline.py /path/to/checkpoint`.

The checks run in a process of their own: `test_smoke` drops torch from `sys.modules` to prove the client does not need it, and torch cannot
be imported a second time into the same process."""
import importlib.util
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

HEAVY = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))   # the client installs without them, and so do its tests


def test_the_task_in_a_fresh_process():
    if not HEAVY:
        return
    done = subprocess.run([sys.executable, __file__], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def check_import_registers_the_task_once():
    from transformers.pipelines import PIPELINE_REGISTRY

    from typecastlm import pipeline as tp

    assert tp.TASK in PIPELINE_REGISTRY.get_supported_tasks()
    tp.register()                         # a second call changes nothing
    assert PIPELINE_REGISTRY.get_supported_tasks().count(tp.TASK) == 1


def check_a_string_question_is_asked_yes_or_no():
    from typecastlm.pipeline import DecisionPipeline

    state, q = DecisionPipeline._question({"state": "S", "question": "Paid?"})
    assert (state, q) == ("S", {"type": "noul", "instructions": "Paid?"})
    obj = {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}}
    assert DecisionPipeline._question({"state": "S", "question": obj})[1] is obj


def check_an_input_without_a_question_is_refused():
    from typecastlm.pipeline import DecisionPipeline

    for bad in ("just text", {"state": "S"}, {"question": "Q"}):
        try:
            DecisionPipeline._question(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("check_"):
            fn()
            print("ok  ", name)
