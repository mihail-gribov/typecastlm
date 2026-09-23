"""Tests that need no service: the shapes, the refusals, and the weight of an import."""
from __future__ import annotations

import sys

import pytest


def test_import_costs_nothing_heavy():
    """A client must not drag in a deep-learning stack. This is the whole point of the package."""
    for mod in ("torch", "transformers"):
        sys.modules.pop(mod, None)
    import typecastlm

    assert set(typecastlm.__all__) == {"Client", "Answer"}
    assert not {"torch", "transformers"} & set(sys.modules)


def test_answer_carries_the_extension():
    from typecastlm import Answer

    a = Answer(prob=0.9, margin=2.2, ms=1.0, input_tokens=10, model="m")
    assert a.unknown == 0.0            # an extension with a default, so the service fields stand


def test_version_zero_takes_one_question():
    from typecastlm import Client

    c = Client(endpoint="http://127.0.0.1:1/never", api_key="x")
    with pytest.raises(NotImplementedError):
        c.ask("state", {"a": {"type": "noul", "instructions": "?"},
                        "b": {"type": "noul", "instructions": "?"}})
