"""Smoke tests. The ones that need weights are skipped unless TYPECASTLM_MODEL points at them."""
from __future__ import annotations

import os

import pytest

MODEL = os.environ.get("TYPECASTLM_MODEL")
needs_model = pytest.mark.skipif(not MODEL, reason="set TYPECASTLM_MODEL to a model directory")


def test_import_is_light():
    """Importing the package must not pull in a deep-learning stack: the remote client runs
    on machines that have none."""
    import sys

    sys.modules.pop("torch", None)
    import typecastlm

    assert {"Client", "RemoteClient"} <= set(typecastlm.__all__)
    assert "torch" not in sys.modules


@needs_model
def test_labels_come_from_the_model():
    from typecastlm import Client

    c = Client(MODEL)
    assert len(c.reader.labels) == 3


@needs_model
def test_noul_returns_a_probability_and_its_log_odds():
    import math

    from typecastlm import Client

    c = Client(MODEL)
    a = c.noul("The ferry leaves at dawn.", "Does the material mention a ferry?",
               true="a ferry is mentioned", false="no ferry is mentioned")
    assert 0.0 <= a.prob <= 1.0
    assert math.isfinite(a.margin)
    assert a.input_tokens > 0


@needs_model
def test_a_longer_option_list_is_refused():
    from typecastlm import Client

    c = Client(MODEL)
    with pytest.raises(ValueError):
        c.ask("any text", {"q": {"type": "choice", "instructions": "which one?",
                                 "criteria": {"a": "1", "b": "2", "c": "3", "d": "4"}}})
