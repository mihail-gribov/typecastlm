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

## Calibration

Two things are true at once, and keeping them apart is the whole of this section.

**Calibration adds no knowledge.** It changes no ordering: macro AUC is 0.817 before and 0.820
after. A document that ranked above another still does.

**Calibration decides whether the numbers are usable.** Without it this reader is badly
overconfident — it says 0.99 where it is right 85% of the time — and its third answer never wins:
`argmax` picks `unsure` on 4% of rows where a third of the material is undecidable.

Three knobs, one job each:

| knob | what it moves | what it leaves alone |
|---|---|---|
| temperature | confidence, calibration error | every decision and every ordering |
| shift on `false` | the threshold between yes and no | orderings |
| shift on `unsure` | how often the reader declines | orderings |

A softmax cannot tell a common shift from nothing, so there are **two** shifts for three answers,
not three.

### What ships, and what it was fitted on

Defaults travel with the checkpoint (`prompt.json`) and the service passes them to the client,
which applies them unless told otherwise. Both were fitted on FEVER's dev split, half for fitting
and half for the numbers below:

| mode | numbers | effect, held out |
|---|---|---|
| binary (`noul`) | temperature 3.48, shift on `false` −0.54 | calibration error 0.127 → 0.069, accuracy 0.852 → 0.858 |
| three answers (`tfu`) | temperature 4.33, shifts +0.18 and +1.64 | three-class accuracy 0.604 → 0.666, calibration error 0.352 → 0.085, `unsure` wins argmax on 25% of rows instead of 4% |

**The temperature should travel; the shifts should not.** Overconfidence is a property of the
model. The shifts encode how often the answer is yes and how often nothing decides it — properties
of FEVER, where a third of the input is undecidable by construction. On data where little is
undecidable they will abstain far too often.

### Fitting your own

Two hundred labelled documents are enough, because there are only three free numbers:

```python
from typecastlm import Client, calibrate

c = Client(calibrated=False)                  # raw logits, nothing applied
rows = [(c.noul(text, question, true=T, false=F).logits, gold)   # gold: true / false / unsure
        for text, gold in my_labelled_sample]

cal = calibrate(rows)
c = Client(temperature=cal["temperature"], shift=tuple(cal["shift"][1:]))
```

`Client(calibrated=False)` switches the shipped defaults off — the right thing when you threshold
`p(unsure)` yourself rather than reading `argmax`, since a threshold you choose on your own data
already does what the shifts do.

## What version zero does, and what it does not

| | |
|---|---|
| one question per call | a bundle of several is refused: a hosted bundle is cheap because the state is read once for all of it, and that saving does not exist yet here |
| `noul` only | `choice` and `score` are refused, not approximated — this reader has three fixed answers, and folding named options onto them would answer a different question convincingly |
| `unknown` comes back unasked | the criteria state two answers and never a third; the third is read anyway |
| the answers are not renamed here | naming them is what `choice` does, through the keys of its `criteria` — so it waits for `choice` rather than arriving as a second mechanism of ours |

The first two keep the surface honest. The third is the extension, and it is the useful one: a
question the document does not decide is a different thing from a question it decides against.

## Running the service yourself

The half that carries the weights installs on purpose, and only where it belongs:

```
pip install "typecastlm[server]"
typecastlm-serve --model mihailgribov/typecastlm-qwen3-3.5b --port 8000
```

The model name is all it needs: the weights come from the Hub on first start and are cached, and
the prompt comes with them — `prompt.json` sits beside the weights.

### Changing the wording

```
typecastlm-serve --model … --prompt my_prompt.json
```

Two sources, and which one is in play is never assumed — `/health` says so:

| source | when | what it means |
|---|---|---|
| `model repository` | default | the wording the checkpoint was measured with; the numbers in the model card are true of this one |
| `override: …` | `--prompt` | your wording — a different question style, another language, a shorter frame. From here the card's numbers describe something else |

There is no third. A checkpoint that ships no `prompt.json` is an error and the server says so,
rather than reaching for a copy lying around: substituting a wording breaks nothing and
invalidates everything that was measured. `examples/prompt.json` is there to be copied and passed
on purpose — it is never picked up on its own.

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
