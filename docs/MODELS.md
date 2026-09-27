# How a model is presented

One repository on the Hub holds a model in every form it is used in, and a tag picks the form.
This page is the convention, written when there was one model and a second was planned, so
that the second is laid out the same way and every reader — the local one, the service, the
container, a launcher — finds what it needs by the same names.

## The repository

`mihailgribov/typecastlm-<base>-<size>`, for instance `mihailgribov/typecastlm-qwen3.5-3.8b`.

| file | what | who reads it |
|---|---|---|
| `model-0000N-of-0000M.safetensors`, `model.safetensors.index.json`, `config.json` | the weights for transformers: the trunk, and the head as a small shard of its own so that a change to the head costs megabytes | `Reader`, `typecastlm-serve --backend local` |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | the tokenizer | the same |
| `<name>-bf16.gguf`, `<name>-q8_0.gguf` | the trunk alone, for launchers; one file per tag, the tag in the name | llama-server and whatever else loads GGUF |
| `head.json` | the head as plain numbers: the verdict rows and every mark in its surface forms | `EmbeddingReader`, `--backend llama` / `openai` |
| `prompt.json` | the wording, the calibration temperatures, `chat_frame`, `max_state_tokens`, `base`, `cut_layer` | every reader |
| `reader.py` | the local reader with no package around it, a copy of `typecastlm/local.py` | someone who wants the weights and nothing else |
| `README.md`, `LICENSE`, `NOTICE` | the card; the licence of the base; what was changed in the base and how | people |

## Tags

`repository[:tag]` wherever a model is named — `--model`, `TYPECASTLM_MODEL`, `EmbeddingReader`,
`llama-server -hf`.

| name | form |
|---|---|
| `mihailgribov/typecastlm-qwen3.5-3.8b` | the safetensors weights, run by transformers in this process |
| `…:Q8_0` | the 8-bit GGUF, run by an embedding server; head and wording fetched from the same repository |
| `…:BF16` | the GGUF at the measured precision |

llama.cpp resolves a tag through the Hub's manifest for the repository, which picks the GGUF
whose name carries the tag; that is why there is exactly one file per tag and the names are
fixed. The service refuses a tagged name for weights held here — it cannot run a GGUF — and says
which backend can.

## What must agree

The same `prompt.json` and the same head in every form, or two paths give two readings and
nothing fails. One source, the checkpoint; the rest is written from it:

| written | by | checked by |
|---|---|---|
| `reader.py` | `scripts/sync_model.py` from `typecastlm/local.py` | the same script with `--check`, and a test |
| `head.json` | `scripts/export_head.py` from the checkpoint | max \|published head − head.json\| = 0 on the verdict rows |
| the GGUF files | `scripts/convert_gguf.py`, then `llama-quantize` | `experiments/71_gguf_launchers/compare.py` against transformers, per file |
| `chat_frame` in `prompt.json` | the tokenizer's own rendering of a system line and a user turn | equality with what the tokenizer renders |
| the readers | — | `tests/same_reading.py` (service against `reader.py`), `tests/prefix_reading.py` (a bundle against one by one), `tests/live_launcher.py`, `tests/live_service.py` |

A new form is published only with its own measurement beside it: a 4-bit file, for one, needs
its temperatures checked before it is trusted.

## A second model

What is the model's own, and so has to be supplied with it, against what is already general.

| | today | for a model on another base |
|---|---|---|
| chat frame | in `prompt.json` as `chat_frame`; the built-in copy is Qwen's and only a fallback | the base's own frame in its `prompt.json`; nothing in the code changes |
| marks | checked against the tokenizer at start: a mark must be one token | the same check; which letters and digits qualify is the tokenizer's answer |
| head | 3 verdict rows and the marks' embedding rows, `hidden_size` wide | the same shape at the base's width |
| conversion to GGUF | the wrapper registers `Qwen3_5ForSequenceClassification` on the converter's Qwen3.5 class and passes `--no-mtp` | the base's classifier name on the base's converter class; whether llama.cpp knows the architecture is the first thing to find out |
| reading a bundle from one pass | continues the hybrid trunk's linear layers from a cache with a patch transformers lacks | a plain transformer needs no patch: the standard cache continues it; the patch is installed only where the trunk has those layers |
| linear-attention kernels | `flash-linear-attention`, hidden off the GPU | not needed by a plain transformer |
| libraries | pinned in `docker/requirements.txt` to what the numbers were measured with | a base that needs a newer transformers moves the pin for both models, and the first is measured again before the image ships |
| serving | one checkpoint per process | two models are two containers, each with its own `/admin`; routing by the request's `model` field in one process is possible and not built |

For a base with towers the model does not use — vision, audio — they are dropped when the
checkpoint is cut, and the card says so, the way it says which blocks were removed.
