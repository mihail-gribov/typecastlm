# The service in Docker

One image, one command, and the Jev API is on port 8000 of your own machine. The image holds the
code and the libraries at the versions the numbers were measured with; the weights are not in it —
they arrive once, into a volume, and stay there across rebuilds.

```
git clone https://github.com/mihail-gribov/typecastlm && cd typecastlm
cp .env.example .env            # set TYPECASTLM_API_KEY at least
docker compose up -d --build
docker compose logs -f          # "ready on 0.0.0.0:8000 as typecastlm-qwen3.5-3.8b …"
curl localhost:8000/health
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
| `TYPECASTLM_DEVICE` | `auto` | `auto` takes CUDA when the container sees a GPU |
| `TYPECASTLM_DTYPE` | `bfloat16` | what the numbers were measured in |
| `TYPECASTLM_MAX_STATE_TOKENS` | the checkpoint's 32768 | states past it are folded in the middle |
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

## What it serves

| route | |
|---|---|
| `POST /v1/systemone` | the questions; the Jev body, field for field — [API.md](API.md) |
| `GET /v1/models` | the checkpoint under its name and the alias `typecastlm-latest` |
| `GET /health` | model, device, dtype, prompt source, calibration, queue load; no key needed |
| `GET /docs`, `/openapi.json` | the schema, as FastAPI renders it |

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
git pull
docker compose up -d --build        # the library layer is cached; the package layer rebuilds in seconds
```

The weights are in the `hf-cache` volume and are not touched by a rebuild. To change the pinned
libraries edit `docker/requirements.txt`, and rerun `tests/live_service.py` against the result
before shipping it: torch and transformers moving under a fixed checkpoint is how numbers change
without anyone changing the model.

## Without compose

```
docker build -t typecastlm .
docker run -d --name typecastlm --gpus all -p 127.0.0.1:8000:8000 \
    -v typecastlm-hf:/data/hf -e TYPECASTLM_API_KEY=… typecastlm
```

Drop `--gpus all` and add `-e TYPECASTLM_DEVICE=cpu` for a host without a GPU.
