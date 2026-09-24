"""The client against a running service: four modes, a bundle, the refusals and the key.

Not part of `pytest`: it needs weights, and on a CPU it takes a couple of minutes. Start a service
and point this at it — that is the only way the HTTP path gets exercised at all.

    typecastlm-serve --model mihailgribov/typecastlm-qwen3.5-3.8b --port 8077 --api-key testkey
    python3 tests/live_service.py                       # or: ... http://host:port key
"""
import os
import sys

import requests

from typecastlm import Client

BASE = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("TYPECASTLM_ENDPOINT", "http://127.0.0.1:8077")).rstrip("/")
KEY = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TYPECASTLM_API_KEY", "testkey")
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


h = requests.get(f"{BASE}/health").json()
print("health:", {k: h[k] for k in ("model", "device", "prompt", "auth", "checks")}, flush=True)
check("health lists the head", len(h["labels"]) >= 29, f"({len(h['labels'])} labels)")
check("health says whether auth is on", isinstance(h["auth"], bool))

c = Client(endpoint=BASE, api_key=KEY)      # the bare host: the client adds the route

a = c.noul(DOC, "Is the claim covered by this policy?",
           true="the policy covers it", false="the policy excludes it")
check("noul", 0.5 < a.prob <= 1.0, f"p={a.prob:.4f} margin={a.margin:+.2f} logits={len(a.logits)}")
check("noul logits carry all three", len(a.logits) == 3)
check("noul has no third number", not hasattr(a, "unknown"))

t = c.tfu(DOC, "Is the claim covered by this policy?",
          true="the policy covers it", false="the policy excludes it")
check("tfu sums to 1", abs(sum(t.p.values()) - 1) < 1e-9,
      f"{ {k: round(v,3) for k,v in t.p.items()} }")

r = c.choice(DOC, "How should this claim be settled?",
             {"deny": "excluded as repeated seepage", "sublimit": "capped by a sublimit",
              "pay": "covered in full"})
check("choice", r.verdict in r.p, f"{r.verdict} p={r.confidence:.3f}")

s = c.scale("The food was cold and the waiter rude, but the bill was small.",
            "How positive is this review overall?",
            {"0": "very negative", "1": "negative", "2": "neutral", "3": "positive",
             "4": "very positive"})
check("scale", s.verdict in s.p, f"{s.verdict} score={s.score:.2f} p={ {k: round(v,2) for k,v in s.p.items()} }")

# 26 options: on a head that stops at mark_P the rest come from the embedding
opts = {f"o{i}": w for i, w in enumerate(
    ["hammer", "sock", "lamp", "brick", "rope", "kettle", "nail", "chair", "broom", "candle",
     "mirror", "ladder", "spoon", "wrench", "bucket", "pillow", "razor", "stamp", "hinge",
     "wheel", "anvil", "crate", "drill", "screw", "tile", "apple"])}
w = c.choice("", "Which of the listed items is food?", opts)
check("26 options", w.verdict == "o25", f"picked {w.verdict} p={w.confidence:.3f}")

b = c.ask(DOC, {"covered": {"type": "noul", "instructions": "Is the claim covered?",
                            "criteria": {"true": "covered", "false": "excluded"}},
                "settle": {"type": "choice", "instructions": "How should it be settled?",
                           "criteria": {"deny": "…", "pay": "…"}}})
check("bundle keeps the keys", set(b["answers"]) == {"covered", "settle"},
      f"types={[b['answers'][k]['type'] for k in b['answers']]}")

# the shape of the API it copies
raw = requests.post(f"{BASE}/v1/systemone", headers={"Authorization": f"Bearer {KEY}"},
                    json={"state": "Help! My payouts have been failing for 3 days.",
                          "model": "jev-latest",
                          "questions": {"frustration": {
                              "type": "score", "instructions": "How frustrated is the customer?",
                              "criteria": ["Calm", "Frustrated", "Very angry"]}}}).json()
f = raw["answers"]["frustration"]
check("score from a list of levels", set(f) >= {"type", "score", "legend", "probabilities",
                                                "confidence"},
      f"score={f['score']:.2f} legend={list(f['legend'].values())}")
check("usage carries both counts", set(raw["usage"]) == {"input_tokens", "output_tokens"},
      str(raw["usage"]))
check("old route still answers",
      requests.post(f"{BASE}/v1/typecast", headers={"Authorization": f"Bearer {KEY}"},
                    json={"state": "x", "questions": {"q": {"type": "noul",
                          "instructions": "?", "criteria": {"true": "a", "false": "b"}}}}
                    ).status_code == 200)

# refusals
check("401 without the key",
      requests.post(f"{BASE}/v1/systemone", json={"state": "x", "questions": {}}).status_code == 401)
for name, body, want in [
    ("no questions", {"state": "x", "questions": {}}, 422),
    ("unknown type", {"state": "x", "questions": {"q": {"type": "guess", "instructions": "?"}}}, 422),
    ("three criteria", {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?",
      "criteria": {"true": "a", "false": "b", "maybe": "c"}}}}, 422),
    ("one option", {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?",
      "criteria": {"only": "a"}}}}, 422),
    ("27 options", {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?",
      "criteria": {f"o{i}": "a" for i in range(27)}}}}, 422),
    ("not a body", {"state": "x"}, 422),
]:
    got = requests.post(f"{BASE}/v1/systemone", headers={"Authorization": f"Bearer {KEY}"},
                        json=body).status_code
    check(f"{want} on {name}", got == want, f"got {got}")

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
