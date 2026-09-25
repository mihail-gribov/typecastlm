"""The service as a proxy to Jev, and the switch between backends through /admin.

Not part of `pytest`: it spends a few cents on Jev and needs a llama-server to switch to.

    typecastlm-serve --backend jev --port 8077 --api-key testkey       # TYPESAFE_API_KEY set
    python3 tests/live_proxy.py http://127.0.0.1:8077 testkey http://127.0.0.1:8091 <files repo>
"""
import sys
import time

import requests

from typecastlm import Client

BASE, KEY = sys.argv[1].rstrip("/"), sys.argv[2]
LLAMA = sys.argv[3] if len(sys.argv) > 3 else ""
FILES = sys.argv[4] if len(sys.argv) > 4 else "mihailgribov/typecastlm-qwen3.5-3.8b-gguf"
H = {"Authorization": f"Bearer {KEY}"}
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


def wait_ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = requests.get(f"{BASE}/admin/config", headers=H).json()
        if s["status"] != "loading":
            return s
        time.sleep(2)
    return {"status": "timeout"}


h = requests.get(f"{BASE}/health").json()
check("health names the proxy", h["backend"] == "jev" and h["status"] == "ready", f"{h['backend']} {h['status']}")
page = requests.get(f"{BASE}/admin")
check("the admin page is served", page.status_code == 200 and "Weights here" in page.text)
check("admin config needs the key", requests.get(f"{BASE}/admin/config").status_code == 401)

c = Client(BASE, api_key=KEY)
a = c.noul(DOC, "Is the claim covered by this policy?", true="the policy covers it", false="the policy excludes it")
check("noul through the proxy", 0.5 < a.prob <= 1.0 and a.logits is None, f"p={a.prob:.3f} model={a.model}")
t = c.tfu(DOC, "Is the claim covered by this policy?", true="the policy covers it", false="the policy excludes it")
check("tfu translated, marked native=false", set(t.p) == {"true", "false", "unsure"},
      f"{t.verdict} {dict((k, round(v, 3)) for k, v in t.p.items())}")
raw = requests.post(f"{BASE}/v1/systemone", headers=H, json={"state": DOC, "questions": {
    "q": {"type": "tfu", "instructions": "Is the claim covered?"}}}).json()
check("the wire says native: false", raw["answers"]["q"].get("native") is False, str(raw["answers"]["q"])[:100])
s = c.scale("The food was cold and the waiter rude.", "How positive is this review?",
            {"neg": "negative", "neu": "neutral", "pos": "positive"})
check("scale under your names", list(s.p) == ["neg", "neu", "pos"], f"{s.verdict} {s.score:.2f}")
b = c.ask(DOC, {"covered": {"type": "noul", "instructions": "Is the claim covered?"},
                "tone": {"type": "score", "instructions": "How formal?", "criteria": ["casual", "formal"]}})
check("bundle forwarded as one request", set(b["answers"]) == {"covered", "tone"} and b["model"].startswith("jev"), b["model"])
m = c.models()
check("models come from upstream", any(x["name"] == "jev-latest" for x in m), str([x["name"] for x in m]))
bad = requests.post(f"{BASE}/v1/systemone", headers=H, json={"state": "x", "model": "x",
                    "questions": {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(30)}}}})
check("30 options pass through to Jev (its limit is 255)", bad.status_code == 200, f"{bad.status_code}")

if LLAMA:
    r = requests.post(f"{BASE}/admin/config", headers=H, json={"backend": "llama", "backend_endpoint": LLAMA, "model": FILES})
    check("switch accepted", r.status_code == 200 and r.json()["status"] == "loading", r.text[:120])
    st = wait_ready()
    check("switched to llama", st["status"] == "ready" and st["settings"]["backend"] == "llama", st.get("detail", ""))
    a2 = c.noul(DOC, "Is the claim covered by this policy?", true="the policy covers it", false="the policy excludes it")
    check("logits are back", a2.logits is not None and 0.5 < a2.prob, f"p={a2.prob:.4f} model={a2.model}")
    h2 = requests.get(f"{BASE}/health").json()
    check("health follows", h2["backend"] == "llama" and len(h2["labels"]) == 39)
    r = requests.post(f"{BASE}/admin/config", headers=H, json={"backend": "llama", "backend_endpoint": "http://127.0.0.1:1", "model": FILES})
    st = wait_ready()
    check("a failed switch keeps the working backend", st["status"] == "ready" and "switch failed" in st["detail"], st["detail"][:80])
    r = requests.post(f"{BASE}/admin/config", headers=H, json={"backend": "jev", "model": "jev-latest", "backend_endpoint": ""})
    st = wait_ready()
    check("back to the proxy", st["status"] == "ready" and st["settings"]["backend"] == "jev", st.get("detail", ""))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
