# typecastlm

**A client for a Jev-class decision model with open weights.** Ask a document a question, get
numbers back. Four modes:

| mode | the question | the answer |
|---|---|---|
| **`noul`** | yes or no, two criteria | `p(yes)`, a softmax over those two answers |
| **`tfu`** | the same question | one distribution over `true`, `false`, **`unsure`** |
| **`choice`** | 2–16 options, one correct | a probability per option |
| **`scale`** | an ordinal rubric, up to 10 levels | a probability per level |

Three of them — `noul`, `choice` and `scale` — carry Jev's names, Jev's arguments and Jev's
fields, so calling code written against that interface keeps its shape. The fourth, `tfu`, is ours:
Jev has nothing like it. The HTTP contract is in [docs/API.md](docs/API.md), laid out the way the
Jev reference is.

**The third answer is what a two-answer reader cannot give.** The model has an output for "neither
criterion applies", and `tfu` is the mode that reads it: one distribution over `true`, `false` and
`unsure`, from the same two criteria and the same forward pass as `noul`. Threshold `unsure` to
abstain, to route to a human, or to drop a document from a pipeline. It stays out of `noul`, whose
probability is a softmax over the two answers that decide the question and nothing else — the two
modes are calibrated separately, and reading one at the other's temperature is simply wrong.

Why this one:

* **Fast** — p50 48 ms on material under 200 tokens and 566 ms at 1000–4000, on a 16 GB consumer
  card: nothing is generated, so a decision is one forward pass and no tokens written.
* **A third answer.** Two-answer readers must call something a yes; this one does not have to.
* **Calibrated per mode** — one temperature per mode ships with the weights and is applied here;
  calibration error 0.011–0.052.
* **Yours to run** — open weights, one command for a local service, and a client with a single
  dependency for whatever talks to it.

## Install

There is no hosted endpoint: the weights are open and the service is yours to run. Three ways,
differing in where the model sits.

**Behind HTTP on this machine** — one command, and `Client()` finds it:

```
pip install "typecastlm[server]"
typecastlm-serve --model mihailgribov/typecastlm-qwen3.5-3.8b --port 8000
export TYPECASTLM_ENDPOINT=http://localhost:8000
```

**In this process**, without HTTP:

```
pip install "typecastlm[local]"
```

```python
from typecastlm import Reader

r = Reader("mihailgribov/typecastlm-qwen3.5-3.8b")
r.noul(doc, "Is the claim covered?", true="…", false="…")
```

**Somewhere else** — the client alone, whose one dependency is `requests`:

```
pip install typecastlm
export TYPECASTLM_ENDPOINT=https://your-service
export TYPECASTLM_API_KEY=…               # only if your service asks for one
```

The first two download 7.5 GB of weights once and run them on a GPU; the numbers below were taken
on a 16 GB consumer card. On CUDA, add the kernels the hybrid trunk wants — without them it falls
back to a slow path and p50 triples:

```
pip install flash-linear-attention fla-core
```

## Start

```python
from typecastlm import Client

c = Client()
a = c.noul(open("page.html").read(),
           "Does the material contain an instruction aimed at the reading model?",
           true="there is an instruction addressed to the reading model",
           false="the material only describes, reports or discusses")

a.prob       # 0.95  — probability of `true` between the two answers that decide the question
a.margin     # +2.94 — the same reading as log-odds, so a saturated probability still ranks
```

## Modes

Every mode takes the material and a question and differs in what the answer ranges over. Each is
one forward pass and carries its own temperature, so a number from one mode is not comparable with
a number from another.

### `noul` — yes or no

```python
a = c.noul(doc, "Is the claim covered?",
           true="the policy covers it", false="the policy excludes it")

a.prob        # 0.95  — p(true) between the two answers that decide the question
a.margin      # +2.87 — the same number as log-odds, so a saturated probability still ranks
a.logits      # the raw outputs, as the service sent them
```

`instructions` says what is asked, the two criteria say what each side means; write them as
descriptions of the material, not as commands. The softmax runs over those two answers only and
the verdict is `prob >= 0.5`, the threshold this mode is calibrated for. Nothing else enters the
number: when you want to know whether the material decides the question at all, that is `tfu`,
a mode of its own with its own temperature.

### `tfu` — one distribution over three answers

```python
t = c.tfu(doc, "Is the claim supported?", true="…", false="…")

t.p           # {'true': 0.81, 'false': 0.11, 'unsure': 0.08} — sums to 1
t.verdict     # 'true'
t.confidence  # 0.81
```

The same two criteria and the same forward pass as `noul`, softmaxed over three answers instead
of two, so here the third competes with the other two. Use it where
"nothing here decides it" is an answer you act on, and `noul` where you need a number comparable
with a two-answer detector. There is no third criterion to write: `unsure` is what is left when
neither of the two fits.

### `choice` — one of 2 to 16 options

```python
r = c.choice(policy, "How should this claim be settled?",
             {"deny_vacancy":      "excluded as repeated seepage",
              "pay_with_sublimit": "covered but capped by the concealed-water sublimit",
              "pay_in_full":       "covered in full"})

r.p           # a probability per option, sums to 1
r.verdict     # 'pay_with_sublimit'
r.confidence  # 0.74 — the probability of the leader, not a separate number
```

Exactly one option is correct and the options carry no order. Inside they are marked `A`, `B`,
`C`…, one output row per mark; your names travel from the request to the answer and never reach
the prompt, so renaming an option cannot move the answer. Two options are allowed, but `noul`
reads a two-way question better, because its rows were fitted for that question and the marks were
not.

### `scale` — a level on an ordinal rubric

```python
r = c.scale(review, "How positive is this review overall?",
            {"0": "very negative", "1": "negative", "2": "neutral",
             "3": "positive", "4": "very positive"})

r.p           # {'0': 0.06, '1': 0.08, '2': 0.11, '3': 0.54, '4': 0.20}
r.verdict     # '3'
```

Levels are ordered, and each is marked with its own digit when that digit is a single character,
otherwise with the next mark from `0…9ABC…`. Ten levels is the practical limit; past that the
marks are read worse than the rubric is written. Nothing in the reading enforces the order, so a
distribution with two separated peaks is possible and means the rubric is being read as categories.

### Several questions at once

```python
c.ask(policy, {
    "covered":  {"type": "noul",   "instructions": "Is the claim covered?",
                 "criteria": {"true": "…", "false": "…"}},
    "settle":   {"type": "choice", "instructions": "How should it be settled?",
                 "criteria": {"deny": "…", "pay": "…"}},
})
# {"answers": {"covered": {"type": "noul", "noul": 0.95, "logits": {...}},
#               "settle":  {"type": "choice", "choice": "pay", "probabilities": {...}, …}},
#  "input_tokens": 1843}
```

Any mix of the four modes in one call, keyed by names you choose; `type` is `noul`, `tfu`,
`choice` or `score`, which this client also spells `scale`. Unlike the four methods above, this
one hands back the service's answer objects as they came, so you read the fields yourself — see
[docs/API.md](docs/API.md). The state is currently read again for each question, so a bundle costs
what the same questions cost one by one; sharing the prefix across a hybrid trunk is not
implemented yet.

## Numbers

JevBench v1.2, public set of 231 tasks: **0.792** overall, and 1.000 / 0.917 / 0.622 on its easy,
standard and hard tiers. On public validation splits, AUC 0.939 on BoolQ, 0.955 on RTE and 0.943
on FEVER, where the third answer separates `NOT ENOUGH INFO` from decidable rows with AUC 0.713.
The full tables, the speed grid and what each was measured on are in the
[model card](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b).

## Calibration

The values are in the checkpoint's `prompt.json` under `calibration`, one temperature per mode,
and the client applies them. What each was fitted on:

| mode | fitted on |
|---|---|
| `noul` | FEVER dev, decidable rows |
| `tfu` | FEVER dev, a third of it undecidable |
| `choice` | ARC-Challenge, four options |
| `scale` | SST-5, five ordered levels |

A temperature changes no answer, only the probability, and it does not carry between pools of
different difficulty. Fit your own:

```python
from typecastlm import calibrate

rows = [(c.noul(text, q, true=T, false=F).logits, gold) for text, gold in my_labelled]
mine = Client(temperature=calibrate(rows)["temperature"])
```

`Client(temperature=…)` uses that one value for every mode, so fit it for the mode you actually
ask in; `Client(calibrated=False)` turns the shipped temperatures off and leaves the logits as
they are.

## Running the service yourself

```
typecastlm-serve --model mihailgribov/typecastlm-qwen3.5-3.8b --port 8000
```

It downloads the weights on first start, checks that the checkpoint fits the interface, and
answers the Jev API:

```
curl localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "…the material…",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                              "criteria": {"true": "explicitly time-sensitive",
                                           "false": "no urgency expressed"}}}}'
# {"model": "…", "answers": {"is_urgent": {"type": "noul", "noul": 0.95,
#                                          "logits": {"true": 3.1, "false": -0.4, "unsure": -2.2}}},
#  "usage": {"input_tokens": 307, "output_tokens": 0}, "calibration": {...}}
```

Route, request body, answer objects and error codes are that API's, field for field, so a client
written against it reaches this service by changing the base URL. Added rather than changed: the
question type `tfu`, `logits` on every answer, and `calibration` on the body — with those, an
answer can be re-read at another temperature without asking anything twice. The whole contract is
in [docs/API.md](docs/API.md).

`--api-key` requires a bearer token; without it the service answers anyone who can reach the port.
The prompt travels with the weights, `--prompt` replaces the wording the model was measured with,
and `/health` reports which wording is in use so a changed one is visible rather than assumed.

## Using the model without any of this

The checkpoint is an ordinary classifier with 29 outputs, so `transformers` loads it directly and
`reader.py` beside the weights reads all four modes. See the
[model card](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b).

## Licence

Apache-2.0 — this package, its texts, and the model it reads (derived from Qwen3.5-4B, also
Apache-2.0). `LICENSE` and `NOTICE` carry the terms and the list of changes.

Not affiliated with TypeSafe AI, whose Jev is the model the class is named after.
