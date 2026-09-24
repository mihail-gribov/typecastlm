# Under a launcher: llama-server and the GGUF

The checkpoint is a trunk and a head. The trunk is the Qwen3.5 stack cut after block 27, an
ordinary causal model any launcher that serves embeddings can run from a GGUF; the head is 39
directions — three verdict rows and the embedding rows of the 36 marks — and applying it is 39
dot products. So the model installs like any other GGUF, and the reading happens in the client:

```
llama-server -m typecastlm-qwen3.5-3.8b-q8_0.gguf --embeddings --port 8080
```

```python
from typecastlm import Client, EmbeddingReader

c = Client(transport=EmbeddingReader("http://127.0.0.1:8080"))
c.noul(doc, "Is the claim covered?", true="the policy covers it", false="it excludes it").prob
```

`EmbeddingReader` needs `requests` and nothing else: it builds the prompt from `prompt.json`,
asks the server for the trunk's last hidden state, applies `head.json` and reads the logits at
the mode's temperature. Both files come from the model directory when the GGUF sits beside the
checkpoint, or from the Hub with a plain GET, cached under `~/.cache/typecastlm`. The `Client`
on top is the same object with the same methods as against the service, so code written for one
runs against the other.

## The files

| file | size | what |
|---|---|---|
| `typecastlm-qwen3.5-3.8b-bf16.gguf` | 7.5 GB | the trunk as measured |
| `typecastlm-qwen3.5-3.8b-q8_0.gguf` | 4.0 GB | the trunk in 8-bit — close enough to keep the calibration |
| `head.json` | 2.8 MB | the 39 directions, each mark in its two surface forms |
| `prompt.json` | | the wording and the temperatures, the same file the service reads |

How close is close: on six prompts covering every mode, against transformers in bf16 on the same
CPU (`experiments/71_gguf_launchers/compare.py`):

| file | cos(h) | max \|Δlogit\| | max \|Δp\| | verdicts |
|---|---|---|---|---|
| bf16 | 0.99982–0.99991 | 0.09 | 0.006 | 6/6 the same |
| Q8_0 | 0.99859–0.99935 | 0.29 | 0.014 | 6/6 the same |

Both stay inside the calibration error the card states (0.011–0.052), so the temperatures are
not refitted for Q8_0. A 4-bit file would need its own measurement before it is trusted, and
none is published.

## What the launcher must do

Two things, and the reader checks both at start:

- **Pool the last token.** The GGUF says so in its metadata (`pooling_type = last`), and
  llama-server honours it; `--pooling last` on the command line says the same. Mean pooling gives
  a vector the head was never built against.
- **Hand the vector over unnormalised.** The native `/embedding` route does when asked with
  `embd_normalize: -1`, which is what the reader sends. The OpenAI-style `/v1/embeddings`
  always normalises, and a unit vector keeps the winner but loses the probabilities — the reader
  refuses one.

**Ollama** loads the GGUF but normalises every embedding before returning it (`server/routes.go`,
marked as a TODO for the model to do), so it is not a backend: nothing calibrated survives the
division. The day that changes, `EmbeddingReader` needs only another route.

## On a GPU

llama.cpp's own image serves the GGUF on any CUDA host without a build:

```
docker run -d --gpus all -p 127.0.0.1:8080:8080 -v /srv/models:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda \
    -m /models/typecastlm-qwen3.5-3.8b-q8_0.gguf --embeddings --host 0.0.0.0 --port 8080 -c 32768
```

`-c` is the context the server reserves; the model reads states up to 32768 tokens and folds
longer ones in the middle, and a server started with less refuses the prompt rather than
truncating it. On a CPU the Q8_0 file answers a 165-token question in about 2.5 s on 16 threads
and a 2400-token one in 30 s.

## Checking a deployment

```
python3 tests/live_launcher.py http://127.0.0.1:8080
```

The same questions as the service test — four modes, 26 options, a bundle, a folded state, the
refusals — through the launcher path.

## Making the files

```
LLAMA_CPP=~/llama.cpp scripts/convert_gguf.py <checkpoint> typecastlm-qwen3.5-3.8b-bf16.gguf bf16
~/llama.cpp/build/bin/llama-quantize typecastlm-qwen3.5-3.8b-bf16.gguf typecastlm-qwen3.5-3.8b-q8_0.gguf Q8_0
scripts/export_head.py <checkpoint> head.json
```

The stock converter does the work; the wrapper registers the classifier's architecture name,
drops the head tensor and writes the pooling metadata. `--no-mtp` is passed for it: Qwen3.5's
converter otherwise counts a next-token-prediction block this checkpoint does not carry, and the
server refuses the file for the missing block.
