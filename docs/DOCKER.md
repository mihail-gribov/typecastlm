# The service in Docker

One image, one command, and the Jev API is on port 8000 of your own machine. The image holds the
code and the libraries at the versions the numbers were measured with; the weights are not in it —
they arrive once, into a volume, and stay there across rebuilds.

```
docker run -d --name typecastlm --gpus all -p 127.0.0.1:8000:8000 \
    -v typecastlm-hf:/data/hf -e TYPECASTLM_API_KEY=… ghcr.io/mihail-gribov/typecastlm:latest
curl localhost:8000/health
```

The image is published at `ghcr.io/mihail-gribov/typecastlm`, tagged with the package version
(`1.2.0`) and `latest`. With the repository checked out, compose reads the settings from `.env`:

```
git clone https://github.com/mihail-gribov/typecastlm && cd typecastlm
cp .env.example .env            # set TYPECASTLM_API_KEY at least
docker compose pull && docker compose up -d     # or: docker compose up -d --build
docker compose logs -f          # "ready on 0.0.0.0:8000 as typecastlm-qwen3.5-3.8b …"
```

The first start downloads the checkpoint (7.5 GB) and takes minutes; every later start takes
seconds. Then, from anywhere that reaches the port:

```
pip install typecastlm
export TYPECASTLM_ENDPOINT=http://localhost:8000 TYPECASTLM_API_KEY=…
python -c "from typecastlm import Client; print(Client().noul('The pipe burst overnight.', 'Is this a sudden event?', true='sudden', false='gradual').prob)"
```

## What the host needs

| | GPU | CPU |
|---|---|---|
| Docker | Engine 24+ with Compose v2 | the same |
| driver | an NVIDIA driver, and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so containers see it | nothing |
| memory | 16 GB of VRAM holds the model and a 32k-token state | 16 GB of RAM |
| disk | 8 GB for the image, 8 GB for the weights | the same |
| a decision | 50 ms on short material, half a second at 4k tokens | about ten seconds on short material, on 24 cores |

The toolkit is one package and one command on Ubuntu:

```
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi     # should list the card
```

Without it `docker compose up` fails with `could not select device driver "nvidia"`. A host with
no GPU at all runs the CPU layer instead, which drops the device reservation and pins the model to
the processor:

```
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

## Settings

All of them in `.env` beside the compose file; `.env.example` lists them with their defaults.
The container itself reads `TYPECASTLM_*` from its environment, so the same names work with
plain `docker run -e …`.

| variable | default | |
|---|---|---|
| `TYPECASTLM_API_KEY` | empty | bearer token(s) the `/v1` routes require, comma-separated for several. Empty means the service answers anyone who reaches the port |
| `TYPECASTLM_MODEL` | `mihailgribov/typecastlm-qwen3.5-3.8b` | a Hub id, or a directory under `/models` |
| `TYPECASTLM_QUEUE` | 32 | requests allowed to wait for the model; the next one gets `529` with `Retry-After` |
| `TYPECASTLM_CONCURRENCY` | 1 local, 8 remote | requests on the backend at once; a llama-server with `--parallel` and Jev take several |
| `TYPECASTLM_DEVICE` | `auto` | `auto` takes CUDA when the container sees a GPU |
| `TYPECASTLM_DTYPE` | `bfloat16` | what the numbers were measured in |
| `TYPECASTLM_MAX_STATE_TOKENS` | the checkpoint's 32768 | states past it are folded in the middle |
| `TYPECASTLM_BACKEND` | `local` | `llama` or `openai` puts the API in front of an embedding server, `jev` in front of TypeSafe's Jev; no GPU is then needed in this container |
| `TYPECASTLM_BACKEND_ENDPOINT` | empty | that server's address; empty takes the backend's own default (`http://127.0.0.1:8080`, or Jev's `https://api.typesafe.ai`) |
| `TYPECASTLM_BACKEND_KEY` | empty | the token that server asks for, if any; for `jev`, their API key |
| `TYPECASTLM_CONFIG` | `/data/hf/typecastlm-config.json` | where `/admin` keeps what was chosen; fills in whatever the environment leaves empty |
| `HF_TOKEN` | empty | only for a private Hub checkpoint |
| `BIND`, `PORT` | `127.0.0.1`, `8000` | where the host publishes the port. `BIND=0.0.0.0` opens it to the network — set a key first |
| `MODELS_DIR` | `./models` | host directory mounted read-only at `/models` |
| `IMAGE_TAG` | `latest` | tag of the built image |

**Weights you already have.** Put the checkpoint directory (the files of the model repository:
`config.json`, `*.safetensors`, the tokenizer, `prompt.json`) under `MODELS_DIR` and point
`TYPECASTLM_MODEL` at it inside the container:

```
MODELS_DIR=/srv/models
TYPECASTLM_MODEL=/models/typecastlm-qwen3.5-3.8b
```

Nothing is downloaded then, and the volume stays empty. `prompt.json` must be there: the wording
is part of what was measured, and the service refuses a checkpoint without it rather than guess.

**Your own wording.** Mount the file and set `TYPECASTLM_PROMPT=/models/prompt.json`; `/health`
then reports `override: …` so a changed wording is visible rather than assumed.

**Choosing in the browser.** `http://localhost:8000/admin` shows the three ways the model can
be behind this address — the weights here, an embedding server, Jev proxied — with a form for
each, and switches at runtime: the old backend answers until the new one is loaded, a failed
switch changes nothing, and the settings that worked are written to
`/data/hf/typecastlm-config.json` in the `hf-cache` volume, so the container comes back with
what was chosen. A setting written in `.env` wins over the page — that is how the compose
overlays pin a backend — and one left empty there is the page's to decide. The page asks for the service's API key when
one is set. A container whose model cannot be loaded still starts and serves `/admin`, which is
the point: set `TYPECASTLM_API_KEY`, `docker compose up -d`, then choose in the page.

**The embedder scheme in one command.** `docker-compose.llama.yml` adds llama.cpp's own server
as a second container (a sidecar) that fetches the GGUF by name and takes the GPU, and points ours at it:

```
docker compose -f docker-compose.yml -f docker-compose.llama.yml up -d
docker compose -f docker-compose.yml -f docker-compose.llama-cpu.yml up -d     # no GPU
```

Our container then runs no model — 4 GB for the GGUF instead of 7.5 for the weights, a start in
seconds, no torch at work — and `/admin` shows the second scheme already chosen. `LLAMA_PARALLEL`
gives the sidecar slots and our service lets eight requests through at once; on a CPU that is
slower than one at a time (measured: 106 s against 132 s for the same 16 documents), on a GPU
it is where a stream of short documents gains, and that number is still to be taken. `LLAMA_CTX` in
`.env` is the longest prompt the sidecar takes (8192 by default; 32768 for the model's full
states, at more memory), `LLAMA_GGUF` the file (`…-gguf:Q8_0`, or `:BF16` for the exact one),
`LLAMA_CACHE_DIR` a host directory with an already-fetched file (`~/.cache/huggingface/hub`). Ollama is not an option here:
it normalises every embedding and the head cannot be applied to the result.

**In front of a llama-server.** With `TYPECASTLM_BACKEND=llama` the container holds the wording
and the head and reads the trunk from a llama-server, so the GPU reservation belongs to that
server, not to this one: run the two side by side, or this one on a CPU host with
`docker-compose.cpu.yml`, and point `TYPECASTLM_BACKEND_ENDPOINT` at the llama-server
(`http://host.docker.internal:8080` for one on the same machine outside Docker). `TYPECASTLM_MODEL`
then names where `prompt.json` and `head.json` come from, the GGUF repository:
`mihailgribov/typecastlm-qwen3.5-3.8b-gguf`. The image is the same, torch simply stays unused.

## What it serves

| route | |
|---|---|
| `POST /v1/systemone` | the questions; the Jev body, field for field — [API.md](API.md) |
| `GET /v1/models` | the checkpoint under its name and the alias `typecastlm-latest` |
| `GET /health` | model, device, dtype, prompt source, calibration, queue load; no key needed |
| `GET /docs`, `/openapi.json` | the schema: every question and answer type, every error. The same file is committed as [openapi.json](openapi.json) |

Every response carries `X-Request-Id` (yours if you sent one) and `X-Process-Time-Ms`, the whole
request as the server saw it, queue included.

The model answers one request at a time — two on the same GPU do not finish sooner than one
after the other — and the rest wait in a line of `TYPECASTLM_QUEUE`. Past that the service says
`529` with a `Retry-After` of the line's length times the average request, which is what the
caller would have waited; the bundled client retries on it, with the header honoured.

## Checking a deployment

`tests/live_service.py` is the client against a running service: four modes, a bundle, the
refusals and the key — the checks a release is gated on. Point it at the container:

```
pip install typecastlm requests
python tests/live_service.py http://localhost:8000 "$TYPECASTLM_API_KEY"
```

On a GPU it finishes in seconds; on a CPU in a few minutes, and `TYPECASTLM_TIMEOUT` (seconds,
120 by default) is the client's patience per call.

`docker compose ps` shows the container `healthy` once the weights are loaded and `/health`
answers; until then it is `starting`, for up to ten minutes on a cold download.

## Updating

```
docker compose pull && docker compose up -d       # the published image
git pull && docker compose up -d --build          # or rebuilt here; the library layer is cached
```

The weights are in the `hf-cache` volume and are not touched by a rebuild. To change the pinned
libraries edit `docker/requirements.txt`, and rerun `tests/live_service.py` against the result
before shipping it: torch and transformers moving under a fixed checkpoint is how numbers change
without anyone changing the model.

## Building the image yourself

```
docker build -t typecastlm .
docker run -d --name typecastlm --gpus all -p 127.0.0.1:8000:8000 \
    -v typecastlm-hf:/data/hf -e TYPECASTLM_API_KEY=… typecastlm
```

Drop `--gpus all` and add `-e TYPECASTLM_DEVICE=cpu` for a host without a GPU. The published
image is built the same way by `.github/workflows/docker.yml` on every release tag.
