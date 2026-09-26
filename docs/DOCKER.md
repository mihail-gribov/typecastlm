# The container

**Your own decision model, up in one command and set up in the browser.**
`ghcr.io/mihail-gribov/typecastlm` brings the model up on your machine with the Jev API on one
port: ask it a closed question about a document, get a probability back, from weights that are
yours. A page at `/admin` shows where the model can be — the weights in this container, an
embedding server elsewhere, or TypeSafe's Jev proxied — and switches between them while the
container runs; the same choice can be made in `.env`. One image serves all three. It holds the
code and the libraries at the versions the numbers were measured with; the weights are not in
it — they arrive once, into a volume, and stay there across rebuilds and restarts.

```
docker run -d --name typecastlm --gpus all -p 127.0.0.1:8000:8000 \
    -v typecastlm-hf:/data/hf -e TYPECASTLM_API_KEY=… ghcr.io/mihail-gribov/typecastlm:latest
curl localhost:8000/health
```

Tags: the package version (`1.2.0`) and `latest`. With the repository checked out, compose
reads the settings from `.env`:

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

## Three places for the model

| scheme | `TYPECASTLM_BACKEND` | what runs in this container | what it needs | what comes back |
|---|---|---|---|---|
| **weights here** | `local` (default) | the whole model, torch on CUDA | a GPU with 16 GB, `nvidia-container-toolkit` on the host | probabilities, `logits`, the calibration |
| **an embedding server** | `llama` or `openai` | the wording and the head only; the trunk is read as a vector from llama-server (the GGUF) or any server speaking `/v1/embeddings` unnormalised | no GPU in this container; the server's address in `TYPECASTLM_BACKEND_ENDPOINT` | the same, within 0.014 in probability for the Q8_0 file |
| **Jev, proxied** | `jev` | a translator: every request goes to `api.typesafe.ai` under their key | `TYPECASTLM_BACKEND_KEY` = their API key | probabilities only, no `logits`; `tfu` asked as a choice with a third option, marked `native: false` |

The API in front is the same in all three, so a caller does not know which is behind the
address, and `/health` says which. The reading — prompt, head, temperatures — is the package's
own code in the first two; the third has no model of ours at all. The second scheme is how the
model runs on a machine without CUDA in Docker, and how one GPU serves several containers.

## What the host needs

| | GPU | CPU |
|---|---|---|
| Docker | Engine 24+ with Compose v2 | the same |
| driver | an NVIDIA driver, and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so containers see it | nothing |
| memory | 16 GB of VRAM holds the model and a 32k-token state | 16 GB of RAM |
| disk | 8 GB for the image, 8 GB for the weights | the same |
| a decision | 50 ms on short material, half a second at 4k tokens | about ten seconds on short material, on 24 cores |

The toolkit is a repository and one package on Ubuntu:

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi     # should list the card
```

Without it `docker compose up` fails with `could not select device driver "nvidia"`. A host with
no GPU at all runs the CPU layer instead, which drops the device reservation and pins the model
to the processor — slow, but the same numbers:

```
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Settings

All of them in `.env` beside the compose file; `.env.example` lists them with their defaults.
The container itself reads `TYPECASTLM_*` from its environment, so the same names work with
plain `docker run -e …`. Three tiers decide each setting: a value given in the environment wins,
what `/admin` chose last time fills in what the environment leaves empty, then the built-in
default. A deployment that says nothing keeps its `/admin` choices across restarts; one that
names a backend in `.env` gets that backend.

| variable | default | |
|---|---|---|
| `TYPECASTLM_API_KEY` | empty | bearer token(s) the `/v1` and `/admin` routes require, comma-separated for several. Empty means the service answers anyone who reaches the port |
| `TYPECASTLM_BACKEND` | `local` | `local`, `llama`, `openai` or `jev` — the table above |
| `TYPECASTLM_BACKEND_ENDPOINT` | the backend's own | the embedding server's address, or the upstream API's; `http://127.0.0.1:8080` and `https://api.typesafe.ai` when empty. A llama-server on the same machine outside Docker is `http://host.docker.internal:8080` |
| `TYPECASTLM_BACKEND_KEY` | empty | the token that server asks for, if any; for `jev`, their API key |
| `TYPECASTLM_MODEL` | `mihailgribov/typecastlm-qwen3.5-3.8b` | a Hub id or a directory under `/models`: the weights for `local`; `prompt.json` and `head.json` for an embedding backend (the GGUF repository `mihailgribov/typecastlm-qwen3.5-3.8b-gguf` holds both, and is the default there); the upstream model name for `jev` |
| `TYPECASTLM_DEVICE` | `auto` | `auto` takes CUDA when the container sees a GPU; `cpu` otherwise |
| `TYPECASTLM_DTYPE` | `bfloat16` | what the numbers were measured in |
| `TYPECASTLM_MAX_STATE_TOKENS` | the checkpoint's 32768 | states past it are folded in the middle |
| `TYPECASTLM_QUEUE` | 32 | requests allowed to wait for the model; the next one gets `529` with `Retry-After` |
| `TYPECASTLM_CONCURRENCY` | 1 local, 8 remote | requests on the backend at once; weights held here take one, a server with slots of its own or Jev take several |
| `TYPECASTLM_CONFIG` | `/data/hf/typecastlm-config.json` | where `/admin` keeps what was chosen — in the `hf-cache` volume, beside the weights |
| `HF_TOKEN` | empty | for a private Hub repository (the GGUF repository, while it is) |
| `BIND`, `PORT` | `127.0.0.1`, `8000` | where the host publishes the port. `BIND=0.0.0.0` opens it to the network — set a key first |
| `MODELS_DIR` | `./models` | host directory mounted read-only at `/models` |
| `IMAGE`, `IMAGE_TAG` | the published image, `latest` | `IMAGE=typecastlm` for a local build |

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

## Choosing in the browser

`http://localhost:8000/admin` draws the three schemes, with a form for each, and switches
between them while the container runs: the old backend keeps answering until the new one is
loaded, a switch that fails leaves the working one in place and says why, and the settings that
worked are written to the config file in the volume, so the container comes back with what was
chosen. The page asks for the service's API key when one is set; `/health` and the page itself
never do. A container whose model cannot be loaded — no GPU, a wrong name, an upstream that does
not answer — still starts, answers `503` on `/v1`, and shows in `/admin` what to fix; that is the
point of it: `docker compose up -d`, then choose.

## The compose files

| file | what it adds |
|---|---|
| `docker-compose.yml` | the service on one GPU, the `hf-cache` volume, `/models` from `MODELS_DIR` |
| `docker-compose.cpu.yml` | layered over it for a host without a GPU: no device reservation, the model on the processor |
| `docker-compose.llama.yml` | the second scheme in one command: llama.cpp's own image as a second container that fetches the GGUF by name and takes the GPU, ours pointed at it. Nothing of theirs is built into our image; the same setting reaches any embedding server, in a container or not |
| `docker-compose.llama-cpu.yml` | the same pair on a host without a GPU |

```
docker compose -f docker-compose.yml -f docker-compose.llama.yml up -d
```

For the sidecar, `.env` takes `LLAMA_GGUF` (`…-gguf:Q8_0`, or `:BF16` for the exact file),
`LLAMA_CTX` (the longest prompt it takes, 8192; 32768 for the model's full states, at more
memory), `LLAMA_UBATCH` (the slice processed at once, 2048; the compute buffer grows with it,
nine gigabytes at 8192), `LLAMA_PARALLEL` (its slots) and `LLAMA_CACHE_DIR` (a host directory
in the Hub cache's layout, `~/.cache/huggingface/hub`, when the file is already there). Ollama
is not an option: it normalises every embedding and the head cannot be applied to the result.

## What it serves

| route | |
|---|---|
| `POST /v1/systemone` | the questions; the Jev body, field for field — [API.md](API.md) |
| `GET /v1/models` | the model under its name and the alias `typecastlm-latest`; Jev's own list when proxied |
| `GET /health` | backend, model, device, dtype, prompt source, calibration, queue load, status; no key needed |
| `GET /admin`, `/admin/config` | the page, and what it reads and writes |
| `GET /docs`, `/openapi.json` | the schema: every question and answer type, every error. The same file is committed as [openapi.json](openapi.json) |

Every response carries `X-Request-Id` (yours if you sent one) and `X-Process-Time-Ms`, the whole
request as the server saw it, queue included. Weights held here answer one request at a time —
two on one GPU do not finish sooner than one after the other — and the rest wait in a line of
`TYPECASTLM_QUEUE`; past that the service says `529` with a `Retry-After` of the line's length
times the average request, and the bundled client retries on it. A bundle of questions about
one state reads the state once: the prompts' common prefix runs once and the tails continue
from it, so twenty questions cost one state and twenty tails.

## Checking a deployment

`tests/live_service.py` is the client against a running service: four modes, a bundle, the
refusals and the key — the checks a release is gated on. Point it at the container:

```
pip install typecastlm requests
python tests/live_service.py http://localhost:8000 "$TYPECASTLM_API_KEY"
```

On a GPU it finishes in seconds; on a CPU in a few minutes, and `TYPECASTLM_TIMEOUT` (seconds,
120 by default) is the client's patience per call. `docker compose ps` shows the container
`healthy` once `/health` answers with a loaded model; until then it is `starting`, for up to ten
minutes on a cold download.

## Updating and building

```
docker compose pull && docker compose up -d       # the published image
git pull && docker compose up -d --build          # or rebuilt here; the library layer is cached
docker build -t typecastlm .                      # the image alone
```

The weights and the `/admin` config are in the `hf-cache` volume and are not touched by a
rebuild. The pinned libraries are in `docker/requirements.txt` (torch's own pin is in the
`Dockerfile`, so a change to that file does not rebuild the 900 MB layer); after changing them,
rerun `tests/live_service.py` against the result before shipping it — torch and transformers
moving under a fixed checkpoint is how numbers change without anyone changing the model. The
published image is built the same way at each release ([RELEASING.md](RELEASING.md)).
