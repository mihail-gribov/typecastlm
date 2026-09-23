"""Smoke tests. The ones that need weights are skipped unless ASKSTATE_MODEL points at them."""
from __future__ import annotations

import os

import pytest

MODEL = os.environ.get("ASKSTATE_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set ASKSTATE_MODEL to a model directory")


def test_imports():
    import askstate

    assert {"Reader", "noul", "choice", "score"} <= set(askstate.__all__)


@needs_model
def test_prompt_shows_words_in_order():
    from askstate import Reader

    r = Reader(MODEL)
    p = r.prompt("a document", "is it so?", {"True": "it is so", "False": "it is not"})
    assert "either True or False" in p
    assert p.index("True means:") < p.index("False means:")
    assert p.rstrip().endswith("Answer:")


@needs_model
def test_third_answer_is_read_without_being_offered():
    from askstate import Reader, noul

    r = Reader(MODEL)
    v = noul(r, "The ferry leaves at dawn.", "Does the material mention a ferry?",
             true="a ferry is mentioned", false="no ferry is mentioned")[0]
    assert 0.0 <= v.p_yes <= 1.0 and 0.0 <= v.p_unknown <= 1.0
    assert "Unknown" in v.raw.p                      # read although never shown


@needs_model
def test_clashing_answer_words_are_refused():
    from askstate import Reader, choice

    r = Reader(MODEL)
    with pytest.raises(ValueError):
        choice(r, "any text", "which one?", {"Truth": "one", "Trusty": "other"})
