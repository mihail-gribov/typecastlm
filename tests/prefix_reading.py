"""A bundle read from one pass over the state must say what the questions say one by one.

Not part of `pytest`: it needs weights. The service reads a bundle by running the common prefix
once and the tails as a batch; this compares that against the same questions asked separately,
for every mode, and reports the largest move in probability.

    python3 tests/prefix_reading.py /path/to/checkpoint [cpu|cuda]
"""
import sys
import time

from typecastlm.server import Reader

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mihailgribov/typecastlm-qwen3.5-3.8b"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "auto"
STATE = ("Section 4.2 Water damage. The policy covers water damage from a burst pipe and excludes "
         "damage from repeated seepage over time. The claim describes a pipe that burst "
         "overnight, flooding the kitchen; the tenant reported it the same morning. ") * 8
CRIT = {"true": "the policy covers it", "false": "the policy excludes it"}
QS = {
    "covered": {"type": "noul", "instructions": "Is the claim covered by this policy?", "criteria": CRIT},
    "sudden": {"type": "noul", "instructions": "Was the damage sudden rather than gradual?", "criteria": CRIT},
    "reported": {"type": "tfu", "instructions": "Was the damage reported promptly?", "criteria": CRIT},
    "settle": {"type": "choice", "instructions": "How should this claim be settled?",
               "criteria": {"deny": "excluded as repeated seepage", "sublimit": "capped by a sublimit",
                            "pay": "covered in full"}},
    "room": {"type": "choice", "instructions": "Which room was affected?",
             "criteria": {"kitchen": None, "bathroom": None, "bedroom": None}},
    "severity": {"type": "score", "instructions": "How severe is the damage?",
                 "criteria": ["minor", "moderate", "severe"]},
}


def probs(a: dict) -> dict:
    if a["type"] == "noul":
        return {"true": a["noul"]}
    return a["probabilities"]


r = Reader(MODEL, device=DEVICE)
t0 = time.time()
single = {k: r.answer(STATE, q)[0] for k, q in QS.items()}
t_single = time.time() - t0
t0 = time.time()
bundle, tokens = r.answer_many(STATE, QS)
t_bundle = time.time() - t0
worst, rows = 0.0, []
for k in QS:
    ps, pb = probs(single[k]), probs(bundle[k])
    d = max(abs(ps[n] - pb[n]) for n in ps)
    worst = max(worst, d)
    lead_s = max(ps, key=ps.get) if len(ps) > 1 else ("true" if ps["true"] >= 0.5 else "false")
    lead_b = max(pb, key=pb.get) if len(pb) > 1 else ("true" if pb["true"] >= 0.5 else "false")
    rows.append(f"  {k:9s} {QS[k]['type']:6s} max|Δp| {d:.4f}  {'same' if lead_s == lead_b else 'FLIPPED'}")
print("\n".join(rows))
print(f"one by one {t_single:.1f} s, as a bundle {t_bundle:.1f} s; bundle usage {tokens} tokens; "
      f"worst max|Δp| = {worst:.4f}")
sys.exit(0 if worst < 0.05 else 1)
