# typecastlm

**A client for a Jev-class decision model with open weights.** Ask a document a closed question,
get a probability back — no prose, no tokens generated, one forward pass. Jev is TypeSafe's closed
model for exactly this; a Jev-class model answers the same shapes of question, and this one does
it from weights you can download and run. Route a ticket, filter a feed, screen what reaches an
agent, rate a review: anything where a program needs a number rather than a paragraph.

Four modes:

| mode | the question | the answer |
|---|---|---|
| **`noul`** | yes or no, two criteria | `p(yes)`, a softmax over those two answers |
| **`tfu`** | the same question | one distribution over `true`, `false`, **`unsure`** |
| **`choice`** | 2–26 options, one correct | a probability per option |
| **`scale`** | an ordinal rubric, 10 levels or fewer | a probability per level |

Three of them — `noul`, `choice` and `scale` — carry Jev's names, Jev's arguments and Jev's
fields, so calling code written against that interface keeps its shape. The fourth, `tfu`, is
ours: Jev has nothing like it. The HTTP contract is in [docs/API.md](https://github.com/mihail-gribov/typecastlm/blob/main/docs/API.md), laid out the way the
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
* **Calibrated per mode** — a stated 0.8 comes out right about 80 % of the time, because each
  mode carries its own temperature, fitted and shipped with the weights (calibration error
  0.011–0.052, and refittable on your own rows).
* **Yours to run** — open weights, one command for a local service, and a client with a single
  dependency for whatever talks to it.

## Install

There is no hosted endpoint: the weights are open and the service is yours to run. Python 3.10 or
newer, and three ways to arrange it, differing in where the model sits.

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

The first two download the checkpoint once — 7.5 GB, 3.8B parameters, derived from Qwen3.5-4B —
and run it on a GPU; the numbers below were taken on a 16 GB consumer card. On CUDA, add the kernels the hybrid trunk wants — without them it falls
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

Criteria carry the meaning, so write them as descriptions of the material rather than as
instructions to a model. The other three modes are below.

## Modes

Every mode takes the material and a question and differs in what the answer ranges over. Each is
one forward pass and carries its own temperature, so a number from one mode is not comparable with
a number from another.

### `noul` — yes or no

```python
a = c.noul(doc, "Is the claim covered?",
           true="the policy covers it", false="the policy excludes it")

a.prob        # 0.95  — p(true) between the two answers that decide the question
a.margin      # +2.94 — the same number as log-odds, so a saturated probability still ranks
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
of two, so here the third competes with the other two. Use it where "nothing here decides it" is
an answer you act on, and `noul` where you need a number comparable with a two-answer detector. There is no third criterion to write: `unsure` is what is left when
neither of the two fits.

### `choice` — one of 2 to 26 options

```python
r = c.choice(policy, "How should this claim be settled?",
             {"deny_vacancy":      "excluded as repeated seepage",
              "pay_with_sublimit": "covered but capped by the concealed-water sublimit",
              "pay_in_full":       "covered in full"})

r.p           # a probability per option, sums to 1
r.verdict     # 'pay_with_sublimit'
r.confidence  # 0.74 — the probability of the leader, not a separate number
```

Exactly one option is correct and the options carry no order of their own. Inside they are marked
`A` to `Z` **in the order you list them**, one row per mark; your names travel from the request to
the answer and never reach the prompt, so renaming an option cannot move the answer — but
reordering can, because the mark is the position. Send the options in a fixed order and the
readings are comparable across documents; shuffle them and they are not — on a public set of 213
option questions, reordering the options alone flipped 29 verdicts. `marks` in the answer says
which option got which letter. Twenty-six is where single-token marks run out. Reading
holds to about six options and then slips — measured on a synthetic task with one right answer
among K: 0.995 up to six, 0.965 at eight, 0.935 at twelve, 0.86 from sixteen on. Two options are
allowed, but `noul` reads a two-way question better, because its rows were fitted for that
question and the marks were not.

### `scale` — a level on an ordinal rubric

```python
r = c.scale(review, "How positive is this review overall?",
            {"0": "very negative", "1": "negative", "2": "neutral",
             "3": "positive", "4": "very positive"})

r.p           # {'0': 0.06, '1': 0.08, '2': 0.11, '3': 0.54, '4': 0.20}
r.verdict     # '3'    — the level that leads
r.score       # 2.72   — the mean level, which is where an ordinal answer really sits
```

Levels are ordered, and each is marked with its own digit when that digit is a single character,
otherwise with the next mark from `0…9ABC…`. `score` averages the levels by their positions and
their probabilities, so a rubric of three levels answers between 0 and 2 and can land at 1.05; the
mark itself never enters the arithmetic, which is why a level marked `A` because it is the
eleventh counts as 11.

Ten levels is the recommendation, and what Jev enforces. Up to 36 are accepted here, since the
marks run `0…9` and then `A…Z` — but past nine the reading degrades badly (0.05 accuracy on levels
10–14 against 0.215 on 0–9), so more levels buy a finer number, not a better one. Nothing in the
reading enforces the order either, so a distribution with two separated peaks is possible and
means the rubric is being read as categories.

### The same modes in this process

`Reader` answers the same four questions without HTTP and returns plain dictionaries rather than
objects:

| over HTTP | in-process | what comes back |
|---|---|---|
| `c.noul(...)` | `r.noul(...)` | `{"p": {"yes", "no"}, "logits"}` |
| `c.tfu(...)` | `r.tfu(...)` | `{"p": {"true", "false", "unsure"}, "verdict", "logits"}` |
| `c.choice(...)` | `r.choice(...)` | `{"p": {option: prob}, "marks", "logits"}` |
| `c.scale(...)` | `r.scale(...)` | the same, keyed by level |

The reader takes its options as `[(name, description), …]` where the client takes a map, and it
applies the same shipped temperatures. `reader.py` beside the weights is this same code with no
package around it.

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
the [API reference](https://github.com/mihail-gribov/typecastlm/blob/main/docs/API.md). The state is currently read again for each question, so a bundle costs
what the same questions cost one by one; sharing the prefix across a hybrid trunk is not
implemented yet.

## Numbers

On public validation splits: AUC 0.939 on BoolQ, 0.955 on RTE and 0.943 on FEVER, where the third
answer separates `NOT ENOUGH INFO` from decidable rows with AUC 0.713.
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

rows = [(c.noul(text, q, true=T, false=F).logits, gold)      # gold is "true" or "false"
        for text, q, gold in my_labelled]
fitted = calibrate(rows, keys=("true", "false"))             # the two answers `noul` reads
mine = Client(temperature=fitted["temperature"])
```

`keys` matters: a `noul` answer carries the third logit as well, and fitting over all three fits
`tfu`'s temperature instead — on the same rows that is 1.1 where 0.6 is right. Two hundred rows
are enough, because one number is free.

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
in the [API reference](https://github.com/mihail-gribov/typecastlm/blob/main/docs/API.md).

`--api-key` requires a bearer token; without it the service answers anyone who can reach the port.
The prompt travels with the weights, `--prompt` replaces the wording the model was measured with,
and `/health` reports which wording is in use so a changed one is visible rather than assumed.

## Using the model without any of this

The checkpoint is an ordinary classifier with 39 outputs, so `transformers` loads it directly and
`reader.py` beside the weights reads all four modes. See the
[model card](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b).

## Licence

Apache-2.0 — this package, its texts, and the model it reads (derived from Qwen3.5-4B, also
Apache-2.0). `LICENSE` and `NOTICE` carry the terms and the list of changes.

Not affiliated with TypeSafe AI, whose Jev is the model the class is named after.
