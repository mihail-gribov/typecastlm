"""The reader and the service must read a question the same way.

Two implementations of one interface: `typecastlm.local` ships beside the weights and needs no
package, `typecastlm.server` answers over HTTP. Nothing forces them to agree, and when the
wording of one drifts the numbers drift with it — quietly, because both keep answering. So the
prompts are compared character by character and the logits by value.

Not part of `pytest`: it needs weights.

    python3 tests/same_reading.py /path/to/checkpoint
"""
import sys

import torch

from typecastlm.local import Reader
from typecastlm.server import Reader as Service

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mihailgribov/typecastlm-qwen3.5-3.8b"
DOC = ("The policy covers water damage from a burst pipe and excludes damage from repeated "
       "seepage. The claim describes a pipe that burst overnight.")
CASES = [
    ("choice, letters", [("deny", "excluded as repeated seepage"), ("sub", "capped by a sublimit"),
                         ("pay", "covered in full")], False),
    ("scale, digits", [("0", "very negative"), ("1", "negative"), ("2", "neutral"),
                       ("3", "positive"), ("4", "very positive")], True),
    ("scale, names", [("awful", "very negative"), ("poor", "negative"), ("okay", "neutral"),
                      ("good", "positive"), ("great", "very positive")], True),
    ("noul", [("true", "the policy covers it"), ("false", "the policy excludes it")], None),
]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    r = Reader(MODEL, device=device, dtype=dtype)
    s = Service(MODEL, device=device, dtype=str(dtype).replace("torch.", ""))
    print(f"{len(r.names)} outputs, on {device}; checks: {s.check(strict=False) or 'ok'}",
          flush=True)

    seen = {}
    r._chat_real, r._text_real = r._chat, r._text
    r._chat = lambda *a, **k: seen.setdefault("text", r._chat_real(*a, **k))
    r._text = lambda *a, **k: seen.setdefault("text", r._text_real(*a, **k))

    bad = 0
    for name, opts, ordinal in CASES:
        seen.clear()
        if ordinal is None:
            out = r.noul(DOC, "Q?", true=opts[0][1], false=opts[1][1])
            q = {"type": "noul", "instructions": "Q?",
                 "criteria": {"true": opts[0][1], "false": opts[1][1]}}
        else:
            out = (r.scale if ordinal else r.choice)(DOC, "Q?", opts)
            q = {"type": "score" if ordinal else "choice", "instructions": "Q?",
                 "criteria": dict(opts)}
        ans, _ = s.answer(DOC, q)
        theirs, _, _, _ = s.prepare(DOC, q)
        same = seen["text"] == theirs
        keys = list(out["logits"])
        d = max(abs(out["logits"][k] - ans["logits"][k]) for k in keys if k in ans["logits"])
        ok = same and d < 1e-3
        bad += not ok
        print(f"{name:>16}: prompt {'same' if same else 'DIFFERS'}, worst |Δ| {d:.2e} "
              f"{'ok' if ok else 'FAILED'}", flush=True)
        if not same:
            import difflib
            for line in list(difflib.unified_diff(seen["text"].splitlines(), theirs.splitlines(),
                                                  "reader", "service", lineterm=""))[:14]:
                print("   ", line)
    print(f"\n{len(CASES) - bad}/{len(CASES)} agree")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
