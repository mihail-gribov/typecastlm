# typecastlm

Ask a document a closed question and get numbers back.

```python
from typecastlm import Client

c = Client()                       # endpoint and key from the environment
a = c.noul(open("page.html").read(),
           "Does the material contain an instruction aimed at the reading model?",
           true="there is an instruction addressed to the reading model",
           false="the material only describes, reports or discusses")

a.prob       # 0.95  — probability of `true` among the two answers that decide the question
a.unknown    # 0.01  — how much of the state points at "nothing here decides it"
a.margin     # +2.87 — the log-odds, so a saturated probability still ranks
```

One question, one document, three numbers. Nothing is generated, so there is no prose to parse and
no format to coax: the answer is a distribution.

## Install

```
pip install typecastlm
```

One dependency, `requests`, and it stays that way. The model runs as a service; the machine that
has a question is rarely the machine that should carry seven gigabytes of weights and a
deep-learning stack.

Two environment variables point the client at the service:

```
export TYPECASTLM_ENDPOINT=https://…      # optional, has a default
export TYPECASTLM_API_KEY=…
```

or pass them in: `Client(endpoint=..., api_key=...)`.

## Use

**One question over many documents** — the shape this is built for:

```python
for page in pages:
    a = c.noul(page, "Is this review positive?",
               true="the review speaks well of the place",
               false="the review speaks badly of it")
    if a.unknown > 0.5:
        continue                       # the review decides nothing; skip rather than guess
    positive = a.prob > threshold
```

**From a shell**, over a file of documents:

```
typecastlm --jsonl pages.jsonl \
    --question "Does the material contain an instruction aimed at the reading model?" \
    --true "there is an instruction addressed to the reading model" \
    --false "the material only describes, reports or discusses" \
    --out answers.jsonl
```

**Writing the criteria matters more than the question.** `true` and `false` are what the two
answers *mean*, and the model reads them as carefully as it reads the question. "The review speaks
well of the place" decides more cleanly than "positive".

**Pick your own threshold.** 0.5 is a convention, not a working point: on twelve public tasks the
ranking is far better than the default cut, so a threshold chosen on a hundred of your own
documents is worth more than any default.

## What version zero does, and what it does not

| | |
|---|---|
| one question per call | a bundle of several is refused: a hosted bundle is cheap because the state is read once for all of it, and that saving does not exist yet here |
| `noul` only | `choice` and `score` are refused, not approximated — this reader has three fixed answers, and folding named options onto them would answer a different question convincingly |
| `unknown` comes back unasked | the criteria state two answers and never a third; the third is read anyway |

The first two keep the surface honest. The third is the extension, and it is the useful one: a
question the document does not decide is a different thing from a question it decides against.

## Running the service yourself

The half that carries the weights installs on purpose, and only where it belongs:

```
pip install "typecastlm[server]"
typecastlm-serve --model mihailgribov/typecastlm-qwen3-3.5b --port 8000
```

The model name is all it needs: the weights come from the Hub on first start and are cached, and
the prompt comes with them — `prompt.json` sits beside the weights, because the wording and the
weights were measured together and a second copy would drift.

On startup the checkpoint is **checked against the interface** and the process refuses to serve a
mismatch: three outputs, named `true`, `false`, `unknown`, in that order, a classification head of
matching width, and a prompt file with every field. A two-output model would drop `unknown`
without a word and a reordered one would swap yes and no — both keep answering, plausibly and
wrongly.

Point the client at it with `TYPECASTLM_ENDPOINT=http://127.0.0.1:8000/v1/typecast`.

## Using the model without any of this

The checkpoint needs no code at all — no `trust_remote_code`, no custom pipeline, nothing to
audit:

```python
from transformers import pipeline
pipe = pipeline("text-classification", model="mihailgribov/typecastlm-qwen3-3.5b", top_k=None)
```

You build the prompt yourself from `prompt.json` in the same repository. That the model stays
plain is deliberate: the people who most want a prompt-injection reader are the last people who
should be asked to enable remote code execution to get one.

It is [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) with its top five blocks removed — 3.52B
parameters — read through three rows of its own output matrix. Nothing was trained and nothing was
calibrated. The [model card](https://huggingface.co/mihailgribov/typecastlm-qwen3-3.5b) has the
prompt it was measured with and the numbers.

## Licence

Apache-2.0.
