#!/bin/sh
# One container, two processes: llama.cpp's server holding the GGUF, ours in front of it.
#
# llama-server fetches the file by name into LLAMA_CACHE on first start and takes minutes to do
# so; our service is started only once it answers, so it never begins in the "no model" state.
# Ours is exec'd as the main process: the container's signals and exit code are its.
set -eu
: "${LLAMA_GGUF:=mihailgribov/typecastlm-qwen3.5-3.8b-gguf:Q8_0}"
: "${LLAMA_CTX:=8192}"
: "${LLAMA_UBATCH:=2048}"
: "${LLAMA_PARALLEL:=1}"
: "${LLAMA_PORT:=8080}"
: "${LLAMA_WAIT:=1800}"
: "${LLAMA_ARGS:=}"

echo "llama-server: ${LLAMA_GGUF}, context ${LLAMA_CTX}, ${LLAMA_PARALLEL} slot(s) …"
# shellcheck disable=SC2086
/app/llama-server -hf "${LLAMA_GGUF}" --embeddings -c "${LLAMA_CTX}" -b "${LLAMA_UBATCH}" \
    -ub "${LLAMA_UBATCH}" --parallel "${LLAMA_PARALLEL}" --kv-unified \
    --host 127.0.0.1 --port "${LLAMA_PORT}" ${LLAMA_ARGS} &
LLAMA_PID=$!

waited=0
until curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
        echo "llama-server exited before it was ready" >&2
        exit 1
    fi
    if [ "${waited}" -ge "${LLAMA_WAIT}" ]; then
        echo "llama-server not ready after ${LLAMA_WAIT} s" >&2
        exit 1
    fi
    [ $((waited % 30)) -eq 0 ] && echo "waiting for llama-server (${waited} s; a first start downloads the GGUF)"
    sleep 2
    waited=$((waited + 2))
done
echo "llama-server is up on 127.0.0.1:${LLAMA_PORT}"

export TYPECASTLM_BACKEND="${TYPECASTLM_BACKEND:-llama}"
export TYPECASTLM_BACKEND_ENDPOINT="${TYPECASTLM_BACKEND_ENDPOINT:-http://127.0.0.1:${LLAMA_PORT}}"
exec /opt/typecastlm/bin/typecastlm-serve "$@"
