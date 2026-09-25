"""The client against TypeSafe's Jev: the same calls as against our service, read from what
Jev returns. Not part of `pytest`: it spends a few cents. `TYPESAFE_API_KEY` must be set.

    python3 tests/live_jev.py
"""
import sys

from typecastlm import ApiError, Client

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


c = Client.jev()
print("info:", c.info(), flush=True)
m = c.models()
check("models lists jev-latest", any(x["name"] == "jev-latest" for x in m),
      str([x["name"] for x in m]))
a = c.noul(DOC, "Is the claim covered by this policy?",
           true="the policy covers it", false="the policy excludes it")
check("noul from a probability", 0.5 < a.prob <= 1.0 and a.logits is None,
      f"p={a.prob:.3f} margin={a.margin:+.2f} model={a.model} {a.ms:.0f} ms")
t = c.tfu(DOC, "Is the claim covered by this policy?",
          true="the policy covers it", false="the policy excludes it")
check("tfu emulated as a choice", t.native is False and set(t.p) == {"true", "false", "unsure"},
      f"{t.verdict} {dict((k, round(v, 3)) for k, v in t.p.items())}")
ch = c.choice(DOC, "How should this claim be settled?",
              {"deny": "excluded as repeated seepage", "sublimit": "capped by a sublimit",
               "pay": "covered in full"})
check("choice", ch.verdict in ch.p, f"{ch.verdict} p={ch.confidence:.3f}")
s = c.scale("The food was cold and the waiter rude, but the bill was small.",
            "How positive is this review overall?",
            {"very_negative": "very negative", "negative": "negative", "neutral": "neutral",
             "positive": "positive", "very_positive": "very positive"})
check("scale under your names", s.verdict in s.p and list(s.p)[0] == "very_negative",
      f"{s.verdict} score={s.score:.2f}")
b = c.ask(DOC, {"covered": {"type": "noul", "instructions": "Is the claim covered?"},
                "tone": {"type": "score", "instructions": "How formal is the text?",
                         "criteria": ["casual", "neutral", "formal"]}})
check("bundle: the state billed once", set(b["answers"]) == {"covered", "tone"}
      and b["input_tokens"] < 2 * a.input_tokens, f"tokens={b['input_tokens']} vs {a.input_tokens} for one")
try:
    Client.jev(model="no-such-model").noul("x", "?")
    check("unknown model refused", False)
except ApiError as e:
    check("unknown model refused", e.status == 400, e.message)
print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
