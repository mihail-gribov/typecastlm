"""`pipeline("decision")` must read a question exactly as the reader does.

The pipeline owns no reading of its own, so this guards the plumbing between transformers and the reader: all four question types, the
three input forms, and the count of tokens. Not part of `pytest`: it needs weights.

    python3 tests/live_pipeline.py /path/to/checkpoint [cuda|cpu]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from transformers import pipeline

import typecastlm  # noqa: F401  (transformers is imported: this registers the task)

MODEL = sys.argv[1]
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda"
STATE = "The invoice was paid on 3 May."
QUESTIONS = [
    "Was the invoice paid?",
    {"type": "tfu", "instructions": "Was the invoice paid?"},
    {"type": "choice", "instructions": "Which month?", "criteria": {"may": "May", "june": "June", "july": "July"}},
    {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "mid", "high"]},
]


def main() -> int:
    decide = pipeline("decision", model=MODEL, device_map=DEVICE,
                      dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
    bad = 0
    for q in QUESTIONS:
        got = decide(STATE, q)
        ref, tokens = decide.reader.answer(STATE, q if isinstance(q, dict) else {"type": "noul", "instructions": q})
        diff = max(abs(got["logits"][k] - ref["logits"][k]) for k in ref["logits"])
        ok = diff < 1e-3 and got["tokens"] == tokens and got[got["type"]] == ref[ref["type"]]
        bad += not ok
        print(("ok   " if ok else "FAIL ") + f"{got['type']}: {got[got['type']]!r}, max logit difference {diff:.4f}", flush=True)
    one = decide({"state": STATE, "question": QUESTIONS[0]})
    many = decide([{"state": STATE, "question": QUESTIONS[0]}] * 2)
    ok = isinstance(many, list) and len(many) == 2 and all(a["noul"] == one["noul"] for a in many)
    bad += not ok
    print(("ok   " if ok else "FAIL ") + "a dict and a list of dicts answer like the two-argument call")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
