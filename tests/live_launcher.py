"""The launcher path against a running llama-server: the same questions as the service test.

Not part of `pytest`: it needs the GGUF served. Start llama-server on it and point this at it —
`prompt.json` and `head.json` come from the model directory or the Hub.

    llama-server -m typecastlm-qwen3.5-3.8b-q8_0.gguf --embeddings --port 8080
    python3 tests/live_launcher.py http://127.0.0.1:8080 [model dir or repo id]
"""
import os
import sys

from typecastlm import Client, EmbeddingReader

BASE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLAMA_ENDPOINT", "http://127.0.0.1:8080")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "mihailgribov/typecastlm-qwen3.5-3.8b"
DOC = ("The policy covers water damage from a burst pipe and excludes damage from repeated "
       "seepage over time. The claim describes a pipe that burst overnight.")
ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name} {detail}", flush=True)
    else:
        fail += 1
        print(f"  FAIL {name} {detail}", flush=True)


r = EmbeddingReader(BASE, model=MODEL)
check("the server hands over an unnormalised vector", True, f"({r.hidden} dims)")
c = Client(transport=r)

a = c.noul(DOC, "Is the claim covered by this policy?",
           true="the policy covers it", false="the policy excludes it")
check("noul", 0.5 < a.prob <= 1.0, f"p={a.prob:.4f} margin={a.margin:+.2f} tokens={a.input_tokens}")
check("noul logits carry all three", len(a.logits) == 3)
t = c.tfu(DOC, "Is the claim covered by this policy?",
          true="the policy covers it", false="the policy excludes it")
check("tfu sums to 1", abs(sum(t.p.values()) - 1) < 1e-9, str({k: round(v, 3) for k, v in t.p.items()}))
ch = c.choice(DOC, "How should this claim be settled?",
              {"deny": "excluded as repeated seepage", "sublimit": "capped by a sublimit",
               "pay": "covered in full"})
check("choice", ch.verdict == "pay", f"{ch.verdict} p={ch.confidence:.3f}")
s = c.scale("The food was cold and the waiter rude, but the bill was small.",
            "How positive is this review overall?",
            {"0": "very negative", "1": "negative", "2": "neutral", "3": "positive",
             "4": "very positive"})
check("scale", s.verdict in s.p, f"{s.verdict} score={s.score:.2f}")
opts = {f"o{i}": w for i, w in enumerate(
    ["hammer", "sock", "lamp", "brick", "rope", "kettle", "nail", "chair", "broom", "candle",
     "mirror", "ladder", "spoon", "wrench", "bucket", "pillow", "razor", "stamp", "hinge",
     "wheel", "anvil", "crate", "drill", "screw", "tile", "apple"])}
w = c.choice("", "Which of the listed items is food?", opts)
check("26 options", w.verdict == "o25", f"picked {w.verdict} p={w.confidence:.3f}")
b = c.ask(DOC, {"covered": {"type": "noul", "instructions": "Is the claim covered?"},
                "settle": {"type": "choice", "instructions": "How should it be settled?",
                           "criteria": {"deny": None, "pay": None}}})
check("bundle keeps the keys", set(b["answers"]) == {"covered", "settle"})
short = EmbeddingReader(BASE, model=MODEL, max_state_tokens=64, check=False)
text, *_ = short.prepare("word " * 500, {"type": "noul", "instructions": "?"})
check("a long state is folded in the middle", "[…]" in text and len(short.tokenize(text)) < 300,
      f"{len(short.tokenize(text))} tokens")
for name, q in [("unknown type", {"type": "guess"}),
                ("three criteria", {"type": "noul", "criteria": {"a": 1, "b": 2, "c": 3}}),
                ("one option", {"type": "choice", "criteria": {"only": "a"}})]:
    try:
        r.answer("x", q)
        check(f"refuses {name}", False)
    except ValueError as e:
        check(f"refuses {name}", True, str(e)[:50])

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
