"""The batcher without weights: who is pooled, how passes are cut, and that every caller hears back."""
from __future__ import annotations

import threading

import pytest

from typecastlm.batching import Batcher
from typecastlm.reading import QuestionError, Reading

CAL = {"verdict": {"temperature": 1.0}, "three_answers": {"temperature": 1.0},
       "choice": {"temperature": 1.0}, "scale": {"temperature": 1.0}}


class Fake(Reading):
    """A prompt is its state; a token is a character; a row's logits say how long it was."""

    def __init__(self):
        self.prompt_cfg = {"calibration": CAL, "system": "", "format_two": "", "means_two": "",
                           "body": "{state}", "tail": "", "max_state_tokens": 10 ** 6}
        self.max_state_tokens = 10 ** 6
        self.passes: list[list[int]] = []
        self.bundles = 0
        self.gate = threading.Event()
        self.gate.set()

    def fold(self, state):
        return state

    def render(self, system, user):
        return user

    def has_mark(self, mark):
        return True

    def tok(self, texts, add_special_tokens=False):
        return {"input_ids": [[1] * len(t) for t in texts]}

    def read_rows(self, ids, marks):
        self.gate.wait()
        self.passes.append([len(x) for x in ids])
        if any(len(x) == 13 for x in ids):
            raise RuntimeError("out of memory")
        return [[float(len(x)), 0.0, -9.0] if m is None else [float(len(x))] + [0.0] * (len(m) - 1)
                for x, m in zip(ids, marks)]

    def answer_many(self, state, questions):
        self.bundles += 1
        return {k: {"bundle": True} for k in questions}, 0


def noul(state):
    return state, {"q": {"type": "noul", "instructions": "", "criteria": {"true": "y", "false": "n"}}}


def test_a_crowd_shares_passes_and_each_caller_gets_its_own_answer():
    r = Fake()
    r.gate.clear()                               # hold the worker so that a crowd queues up
    b = Batcher(r, max_rows=4, batch_tokens=10 ** 6)
    states = ["x" * n for n in (5, 1, 9, 3, 7, 2)]
    got = {}

    def ask(s):
        got[s] = b(*noul(s))

    first = threading.Thread(target=ask, args=("x" * 11,))
    first.start()
    while not b._q.empty():                      # the worker took the first and waits on the gate
        pass
    threads = [threading.Thread(target=ask, args=(s,)) for s in states]
    for t in threads:
        t.start()
    while b._q.qsize() < len(states):
        pass
    r.gate.set()
    for t in [first, *threads]:
        t.join(5)
    assert r.passes[0] == [11]
    # ascending length, at most max_rows a pass
    assert r.passes[1:] == [[1, 2, 3, 5], [7, 9]]
    for s, (answers, tokens) in got.items():
        assert tokens == len(s)
        assert answers["q"]["logits"]["true"] == len(s)


def test_the_token_budget_cuts_a_pass_and_a_long_row_runs_alone():
    r = Fake()
    b = Batcher(r, max_rows=64, batch_tokens=10)
    r.gate.clear()
    jobs = [noul("x" * n) for n in (2, 2, 3, 30)]
    out = []
    ts = [threading.Thread(target=lambda j=j: out.append(b(*j))) for j in jobs]
    for t in ts:
        t.start()
    while b._q.qsize() + len(r.passes) < 1:
        pass
    r.gate.set()
    for t in ts:
        t.join(5)
    assert len(out) == 4
    assert [30] in r.passes                     # past the budget on its own, never refused
    assert all(len(p) * max(p) <= 10 or len(p) == 1 for p in r.passes)


def test_a_bundle_keeps_its_own_path_and_a_bad_question_names_its_key():
    r = Fake()
    b = Batcher(r)
    answers, _ = b("s", {"a": {"type": "noul"}, "b": {"type": "noul"}})
    assert r.bundles == 1 and answers == {"a": {"bundle": True}, "b": {"bundle": True}}
    with pytest.raises(QuestionError) as e:
        b("s", {"bad": {"type": "poem"}})
    assert e.value.key == "bad"


def test_a_failed_pass_fails_its_rows_only():
    r = Fake()
    b = Batcher(r)
    with pytest.raises(RuntimeError):
        b(*noul("x" * 13))
    answers, _ = b(*noul("x" * 4))               # the worker is still alive
    assert answers["q"]["logits"]["true"] == 4


def test_marks_questions_are_pooled_with_yes_no_ones():
    r = Fake()
    b = Batcher(r)
    q = {"type": "choice", "instructions": "", "criteria": {"one": "1", "two": "2", "three": "3"}}
    answers, _ = b("abc", {"c": q})
    assert answers["c"]["type"] == "choice" and set(answers["c"]["logits"]) == {"one", "two", "three"}


def test_a_closed_batcher_lets_go_of_its_reader():
    r = Fake()
    b = Batcher(r)
    b(*noul("abc"))
    b.close()
    for _ in range(500):
        if b.reader is None:
            break
        threading.Event().wait(0.01)
    assert b.reader is None
