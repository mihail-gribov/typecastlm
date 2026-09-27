#!/usr/bin/env bash
# Everything from scratch, the way the README says: the package from PyPI, the model from the Hub,
# the image from the registry, each of the README's ways of arranging them, then the checks.
# All of it lands on a RAM disk and is thrown away afterwards.
#
#   tests/fresh_install.sh              run what is not done yet, stage by stage
#   tests/fresh_install.sh --status     what is done, by what is on disk; starts nothing
#   tests/fresh_install.sh --only s6    one stage again
#   tests/fresh_install.sh --with-jev   also the client against TypeSafe's Jev (a few cents)
#   tests/fresh_install.sh --clean      remove the RAM-disk directory
#
# Ctrl-C stops between steps or inside one; whatever this script started on the card is stopped,
# and running it again continues. Downloads resume. Made for tmux: the log is the console.
set -u
cd "$(dirname "$0")/.."
REPO_DIR=$(pwd)
ROOT=${ROOT:-/dev/shm/typecastlm-fresh}
case "$ROOT" in /dev/shm/?*) ;; *) echo "ROOT must be a directory under /dev/shm, not '$ROOT'" >&2; exit 2;; esac
MODEL=${MODEL:-mihailgribov/typecastlm-qwen3.5-3.8b}
IMAGE=${IMAGE:-ghcr.io/mihail-gribov/typecastlm:latest}
LLAMA_IMAGE=${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}
P_SERVE=${P_SERVE:-8110}; P_DOCKER=${P_DOCKER:-8111}; P_LLAMA=${P_LLAMA:-8112}; P_OVER=${P_OVER:-8113}
KEY=fresh-test
RES=$ROOT/results
export UV_CACHE_DIR=$ROOT/uvcache HF_HOME=$ROOT/hf TYPECASTLM_CACHE=$ROOT/tcache
STAGES="s1 s2 s3 s4 s5 s6 s7 s8 s9"
declare -A TITLE=(
  [s1]="client from PyPI            (README: Somewhere else)"
  [s2]="pip install typecastlm[local] + kernels"
  [s3]="the checkpoint from the Hub, 7.5 GB"
  [s4]="reader in this process, GPU (README: In this process)"
  [s5]="pip install typecastlm[server] + kernels"
  [s6]="typecastlm-serve, GPU        (README: Behind HTTP on this machine)"
  [s7]="the container, GPU          (README: In a container)"
  [s8]="llama-server + the reader   (README: Under a launcher)"
  [s9]="typecastlm-serve over llama (--backend llama)"
  [s10]="the client against Jev"
)
WITH_JEV=0; ONLY=""
while [ $# -gt 0 ]; do case "$1" in
  --status) MODE=status;; --clean) MODE=clean;; --with-jev) WITH_JEV=1;;
  --only) ONLY=$2; shift;; *) echo "unknown option $1" >&2; exit 2;; esac; shift; done
MODE=${MODE:-run}
[ "$WITH_JEV" = 1 ] && STAGES="$STAGES s10"

say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
status() {
    local done_n=0 total=0
    for s in $STAGES; do
        total=$((total+1))
        if [ -f "$RES/$s.ok" ]; then done_n=$((done_n+1)); printf '  done  %-4s %-58s %s\n' "$s" "${TITLE[$s]}" "$(cat "$RES/$s.ok")"
        elif [ -f "$RES/$s.fail" ]; then printf '  FAIL  %-4s %-58s %s\n' "$s" "${TITLE[$s]}" "$(head -c 160 "$RES/$s.fail")"
        else printf '  --    %-4s %s\n' "$s" "${TITLE[$s]}"; fi
    done
    echo "$done_n/$total stages done; root $ROOT ($(du -sh "$ROOT" 2>/dev/null | cut -f1))"
}
# What a stage starts is written down in files, not variables: stages run in a subshell, and
# the stop has to find their processes from outside it.
track_pid() { echo "$1" >> "$ROOT/started.pids"; }
track_container() { echo "$1" >> "$ROOT/started.containers"; }
stop_all() {
    [ -f "$ROOT/started.pids" ] && while read -r p; do kill "$p" 2>/dev/null; done < "$ROOT/started.pids"
    [ -f "$ROOT/started.containers" ] && while read -r c; do docker rm -f "$c" >/dev/null 2>&1; done < "$ROOT/started.containers"
    : > "$ROOT/started.pids"; : > "$ROOT/started.containers"
}
trap 'echo; say "stopped"; stop_all; status; echo "continue: tests/fresh_install.sh"; exit 130' INT TERM

if [ "$MODE" = status ]; then status; exit 0; fi
if [ "$MODE" = clean ]; then
    docker rm -f fresh-typecastlm fresh-llama >/dev/null 2>&1
    rm -rf "${ROOT:?}" && echo "removed $ROOT"; exit 0
fi

# -- before anything is started ----------------------------------------------------------------
mkdir -p "$RES" "$HF_HOME" "$ROOT/llama"
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 30 ] || { echo "only ${free_gb} GB free under $ROOT; 30 are needed" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is not installed" >&2; exit 1; }
docker info 2>/dev/null | grep -qi nvidia || { echo "docker has no nvidia runtime: install nvidia-container-toolkit" >&2; exit 1; }
for p in $P_SERVE $P_DOCKER $P_LLAMA $P_OVER; do
    if ss -tln 2>/dev/null | grep -q ":$p "; then echo "port $p is taken; set P_SERVE/P_DOCKER/P_LLAMA/P_OVER" >&2; exit 1; fi
done
gpu_free() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
need_gpu() { local f; f=$(gpu_free); [ "$f" -ge "$1" ] || { echo "the card has $f MiB free, $1 are needed" ; return 1; }; }
wait_http() {   # url, seconds, command printing progress, text the answer must contain
    local t=0
    until curl -sf "$1" 2>/dev/null | grep -q "${4:-.}"; do
        [ $t -ge "$2" ] && return 1
        [ $((t % 30)) -eq 0 ] && say "  waiting for $1 (${t}s) ${3:+$(eval "$3")}"
        sleep 5; t=$((t+5))
    done
}
status

run_stage() {
    local s=$1
    [ -n "$ONLY" ] && [ "$ONLY" != "$s" ] && return 0
    [ -z "$ONLY" ] && [ -f "$RES/$s.ok" ] && return 0
    rm -f "${RES:?}/${s:?}.ok" "${RES:?}/${s:?}.fail"
    say "== $s  ${TITLE[$s]}"
    local t0 rc out; t0=$(date +%s)
    # A subshell outside any `if`: inside a condition bash ignores `set -e`, and a step that
    # failed half way would pass for done.
    ( set -e; "stage_$s" ) 2>&1 | tee "$ROOT/$s.log"; rc=${PIPESTATUS[0]}
    out=$(tail -1 "$ROOT/$s.log")
    if [ "$rc" -eq 0 ]; then
        echo "$out ($(( $(date +%s) - t0 )) s)" > "$RES/$s.ok"; say "   done: $out"
    else
        echo "$out" > "$RES/$s.fail"; say "   FAILED (exit $rc): $out"
    fi
    stop_all
}
fresh_venv() { rm -rf "${ROOT:?}/${1:?}"; uv venv -q "$ROOT/$1"; }

stage_s1() {
    fresh_venv venv-client
    uv pip install -q --refresh --python "$ROOT/venv-client/bin/python" typecastlm
    "$ROOT/venv-client/bin/python" - <<'PY'
import sys, importlib.metadata as md, typecastlm
from typecastlm import Client
deps = sorted(d.metadata["Name"].lower() for d in md.distributions())
heavy = sorted({"torch", "transformers", "numpy", "fastapi"} & set(deps))
assert not heavy, heavy
Client("http://localhost:8000"); Client.jev(api_key="k")
print(f"typecastlm {typecastlm.__version__} from PyPI, {len(deps)} packages, none heavy")
PY
}
install_heavy() {   # venv name, extra
    fresh_venv "$1"
    say "  installing typecastlm[$2] (torch comes with it: gigabytes)"
    uv pip install --python "$ROOT/$1/bin/python" "typecastlm[$2]" 2>&1 | tail -2
    uv pip install -q --python "$ROOT/$1/bin/python" flash-linear-attention fla-core
    "$ROOT/$1/bin/python" -c "import torch, transformers, fla, typecastlm; print(f'typecastlm {typecastlm.__version__}, torch {torch.__version__}, transformers {transformers.__version__}, fla {fla.__version__}, cuda {torch.cuda.is_available()}')"
}
stage_s2() { install_heavy venv-local local; }
stage_s5() { install_heavy venv-server server; }
stage_s3() {
    local py=$ROOT/venv-local/bin/python blobs="$HF_HOME/hub/models--${MODEL//\//--}/blobs"
    [ -x "$py" ] || { echo "s2 first: the downloader comes with typecastlm[local]"; return 1; }
    size() { du -sm "$blobs" 2>/dev/null | cut -f1; }
    say "  the Hub's own downloader first, watched for a stall"
    "$ROOT/venv-local/bin/hf" download "$MODEL" --exclude "*.gguf" >"$ROOT/hf_download.log" 2>&1 &
    local pid=$! last=-1 still=0; track_pid $pid
    while kill -0 $pid 2>/dev/null; do
        sleep 30; local now; now=$(size); now=${now:-0}
        say "  $now MB of ~7540"
        if [ "$now" = "$last" ]; then still=$((still+30)); else still=0; fi
        last=$now
        if [ $still -ge 180 ]; then say "  no progress for 3 minutes: the downloader stalled, falling back to curl"; kill $pid; break; fi
    done
    wait $pid 2>/dev/null && how="the Hub's downloader" || {
        how="curl, after the Hub's downloader stalled"
        "$py" - "$MODEL" "$HF_HOME" <<'PY'
import json, os, subprocess, sys, urllib.request
repo, home = sys.argv[1], sys.argv[2]
info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true"))
cache = f"{home}/hub/models--{repo.replace('/', '--')}"
snap = f"{cache}/snapshots/{info['sha']}"
os.makedirs(f"{cache}/blobs", exist_ok=True); os.makedirs(snap, exist_ok=True); os.makedirs(f"{cache}/refs", exist_ok=True)
open(f"{cache}/refs/main", "w").write(info["sha"])
for s in info["siblings"]:
    name = s["rfilename"]
    if name.endswith(".gguf"):
        continue
    etag = s["lfs"]["sha256"] if "lfs" in s else s["blobId"]
    blob = f"{cache}/blobs/{etag}"
    if not (os.path.exists(blob) and os.path.getsize(blob) == s["size"]):
        print(f"  fetching {name} ({s['size'] / 1e6:.0f} MB)", flush=True)
        subprocess.run(["curl", "-L", "-C", "-", "--retry", "20", "--retry-delay", "5", "-s", "-S", "-o", blob,
                        f"https://huggingface.co/{repo}/resolve/{info['sha']}/{name}"], check=True)
    link = f"{snap}/{name}"
    if not os.path.lexists(link):
        os.symlink(os.path.relpath(blob, os.path.dirname(link)), link)
PY
    }
    echo "$(size) MB in the cache, by $how"
}
stage_s4() {
    need_gpu 8500
    "$ROOT/venv-local/bin/python" - "$MODEL" <<'PY' 2>&1 | grep -v -i "warning\|Loading weights\|Fetching"
import sys, time
from typecastlm import Reader
doc = ("The policy covers water damage from a burst pipe and excludes damage from repeated seepage "
       "over time. The claim describes a pipe that burst overnight.")
r = Reader(sys.argv[1])
ask = lambda: r.noul(doc, "Is the claim covered?", true="the policy covers it", false="the policy excludes it")
ask(); ask()
t0 = time.time(); a = ask(); ms = (time.time() - t0) * 1000
many = r.noul_many(doc, [("Is the claim covered?", "covered", "excluded"), ("Was the damage sudden?", "sudden", "gradual")])
assert a["p"]["yes"] > 0.9 and len(many) == 2
print(f"README snippet: p(yes) {a['p']['yes']:.3f} in {ms:.0f} ms on the card; a bundle of two read from one pass")
PY
}
live() {   # endpoint, key
    "$ROOT/venv-client/bin/python" "$REPO_DIR/tests/live_service.py" "$1" "$2" 2>&1 | grep -E "FAIL|passed" | tr '\n' ' '
}
stage_s6() {
    need_gpu 8500
    "$ROOT/venv-server/bin/typecastlm-serve" --model "$MODEL" --port $P_SERVE --api-key $KEY >"$ROOT/serve.log" 2>&1 &
    track_pid $!
    wait_http "http://127.0.0.1:$P_SERVE/health" 600 "" '"status":"ready"' || { tail -3 "$ROOT/serve.log"; echo "the service did not come up"; return 1; }
    curl -s "http://127.0.0.1:$P_SERVE/v1/systemone" -H "Authorization: Bearer $KEY" -H 'content-type: application/json' -d '{"state": "Reply by Friday noon or the offer lapses.", "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?", "criteria": {"true": "explicitly time-sensitive", "false": "no urgency expressed"}}}}' | grep -q '"noul"' || { echo "the README curl example got no answer"; return 1; }
    out=$(live "http://127.0.0.1:$P_SERVE" $KEY); echo "$out" | grep -q " 0 failed" || { echo "live suite: $out"; return 1; }
    echo "typecastlm-serve from PyPI on the card: the README curl answered; $out"
}
stage_s7() {
    need_gpu 8500
    say "  pulling $IMAGE"
    docker pull -q "$IMAGE" >/dev/null || { echo "cannot pull $IMAGE: private? docker login ghcr.io"; return 1; }
    docker rm -f fresh-typecastlm >/dev/null 2>&1 || true
    docker run -d --gpus all --name fresh-typecastlm --user "$(id -u):$(id -g)" -e HOME=/tmp -p 127.0.0.1:$P_DOCKER:8000 \
        -v "$HF_HOME:/data/hf" -e TYPECASTLM_API_KEY=$KEY -e TYPECASTLM_CONFIG=/tmp/cfg.json "$IMAGE" >/dev/null
    track_container fresh-typecastlm
    wait_http "http://127.0.0.1:$P_DOCKER/health" 900 "" '"status":"ready"' || { docker logs fresh-typecastlm 2>&1 | tail -3; echo "the container did not come up"; return 1; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$P_DOCKER/admin")" = 200 ] || { echo "/admin does not answer"; return 1; }
    out=$(live "http://127.0.0.1:$P_DOCKER" $KEY); echo "$out" | grep -q " 0 failed" || { echo "live suite: $out"; return 1; }
    echo "the container on the card, /admin served; $out"
}
start_llama() {
    docker rm -f fresh-llama >/dev/null 2>&1 || true
    docker run -d --gpus all --name fresh-llama --user "$(id -u):$(id -g)" -e HOME=/tmp -p 127.0.0.1:$P_LLAMA:8080 \
        -v "$ROOT/llama:/data/cache" -e LLAMA_CACHE=/data/cache "$LLAMA_IMAGE" \
        -hf "$MODEL:Q8_0" --embeddings -c 8192 -b 2048 -ub 2048 --host 0.0.0.0 --port 8080 >/dev/null
    track_container fresh-llama
    wait_http "http://127.0.0.1:$P_LLAMA/health" 5400 'echo "$(du -sm '"$ROOT"'/llama 2>/dev/null | cut -f1) MB of ~4010"' ok \
        || { docker logs fresh-llama 2>&1 | tail -3; echo "llama-server did not come up"; return 1; }
}
stage_s8() {
    need_gpu 6500
    uv pip install -q --python "$ROOT/venv-client/bin/python" "typecastlm[embed]"
    start_llama
    "$ROOT/venv-client/bin/python" -c "
from typecastlm import Client, EmbeddingReader
c = Client(transport=EmbeddingReader('http://127.0.0.1:$P_LLAMA'))
a = c.noul('The pipe burst overnight.', 'Is this a sudden event?', true='sudden', false='gradual')
assert a.prob > 0.9 and a.logits, a
print('README snippet over the launcher:', round(a.prob, 3))"
    out=$("$ROOT/venv-client/bin/python" "$REPO_DIR/tests/live_launcher.py" "http://127.0.0.1:$P_LLAMA" 2>&1 | grep -E "FAIL|passed" | tr '\n' ' ')
    echo "$out" | grep -q " 0 failed" || { echo "launcher suite: $out"; return 1; }
    echo "llama-server fetched $MODEL:Q8_0 by name; the PyPI client read it; $out"
}
stage_s9() {
    need_gpu 6500
    start_llama
    fresh_venv venv-thin
    uv pip install -q --python "$ROOT/venv-thin/bin/python" "typecastlm[embed]" fastapi uvicorn
    "$ROOT/venv-thin/bin/typecastlm-serve" --backend llama --backend-endpoint "http://127.0.0.1:$P_LLAMA" --model "$MODEL:Q8_0" \
        --port $P_OVER --api-key $KEY >"$ROOT/serve_over.log" 2>&1 &
    track_pid $!
    wait_http "http://127.0.0.1:$P_OVER/health" 120 "" '"status":"ready"' || { tail -3 "$ROOT/serve_over.log"; echo "the service over llama did not come up"; return 1; }
    "$ROOT/venv-thin/bin/python" -c "import importlib.util as u; assert u.find_spec('torch') is None"
    out=$(live "http://127.0.0.1:$P_OVER" $KEY); echo "$out" | grep -q " 0 failed" || { echo "live suite: $out"; return 1; }
    echo "the service with no torch installed, over llama-server; $out"
}
stage_s10() {
    [ -n "${TYPESAFE_API_KEY:-}" ] || export TYPESAFE_API_KEY=$(grep -m1 '^TYPESAFE_API_KEY' "$REPO_DIR/../../.env" | cut -d= -f2- | tr -d "\"'")
    out=$("$ROOT/venv-client/bin/python" "$REPO_DIR/tests/live_jev.py" 2>&1 | grep -E "FAIL|passed" | tr '\n' ' ')
    echo "$out" | grep -q " 0 failed" || { echo "jev suite: $out"; return 1; }
    echo "the PyPI client against Jev; $out"
}

for s in $STAGES; do run_stage "$s"; done
echo; status
