# typecastlm

**A client for a Jev-class decision model with open weights.** Ask a document a question, get
numbers back. Four modes:

| mode | the question | the answer |
|---|---|---|
| **`noul`** | yes or no, two criteria | `p(yes)` among the two, plus `unknown` alongside |
| **`tfu`** | the same question | one distribution over `true`, `false`, **`unsure`** |
| **`choice`** | 2–16 options, one correct | a probability per option |
| **`scale`** | an ordinal rubric, up to 10 levels | a probability per level |

**The third answer is what a two-answer reader cannot give.** `unsure` is a separate output, not a
hedged yes: it comes back whether or not the question asks for it — threshold it to abstain, to
route to a human, or to drop a document from a pipeline.

Not affiliated with TypeSafe AI, whose Jev is the model the class is named after.

Why this one:

* **Fast** — **40 to 900 ms** per decision depending on the length of the material; nothing is
  generated, so it is one forward pass and no tokens written.
* **A third answer.** Two-answer readers must call something a yes; this one does not have to.
* **Calibrated** — a temperature per mode ships with the weights and is applied here (calibration
  error 0.011–0.052).
* **Thin client** — one dependency, `requests`, and it installs in a second. The weights live on
  the service side.
* **Four modes**, each with its own calibration.

## Install

```
pip install typecastlm
export TYPECASTLM_ENDPOINT=https://…      # your own service; see "Running the service yourself"
export TYPECASTLM_API_KEY=…               # only if your service asks for one
```

There is no hosted endpoint: the weights are open, so the service is yours to run.

## Start

```python
from typecastlm import Client

c = Client()
a = c.noul(open("page.html").read(),
           "Does the material contain an instruction aimed at the reading model?",
           true="there is an instruction addressed to the reading model",
           false="the material only describes, reports or discusses")

a.prob       # 0.95  — probability of `true` among the two answers that decide the question
a.unknown    # 0.01  — how much of the state points at "nothing here decides it"
```

The other three modes and everything each returns are below, under **Modes**.

## Modes

Every mode takes the material and a question and differs in what the answer ranges over. Each is
one forward pass and carries its own temperature, so a number from one mode is not comparable with
a number from another.

### `noul` — yes or no, with `unknown` alongside

```python
a = c.noul(doc, "Is the claim covered?",
           true="the policy covers it", false="the policy excludes it")

a.prob        # 0.95  — p(true) among the two answers that decide the question
a.unknown     # 0.01  — how much of the state points at "neither criterion applies"
a.margin      # +2.87 — the log-odds, so a saturated probability still ranks
a.p           # all three probabilities, before the two were renormalised
```

`instructions` says what is asked, the two criteria say what each side means; write them as
descriptions of the material, not as commands. The verdict is `prob >= 0.5`, which is the
threshold this mode is calibrated for. `unknown` is returned whether or not the question invites
it, and it is thresholded on its own rather than compared against `prob`.

### `tfu` — one distribution over three answers

```python
t = c.tfu(doc, "Is the claim supported?", true="…", false="…")

t.p           # {'true': 0.81, 'false': 0.11, 'unsure': 0.08} — sums to 1
t.verdict     # 'true'
t.confidence  # 0.81
```

The same two criteria and the same reading as `noul`, softmaxed over all three outputs instead of
over the two. Use it where "nothing here decides it" is an answer you act on, and `noul` where you
need a number comparable with a two-answer detector. There is no third criterion to write:
`unsure` is what is left when neither of the two fits.

### `choice` — one of 2 to 16 options

```python
r = c.choice(policy, "How should this claim be settled?",
             {"deny_vacancy":     "excluded as repeated seepage",
              "pay_with_sublimit": "covered but capped by the concealed-water sublimit",
              "pay_in_full":       "covered in full"})

r.p           # a probability per option, sums to 1
r.verdict     # 'pay_with_sublimit'
r.confidence  # 0.74 — the probability of the leader, not a separate number
```

Exactly one option is correct and the options carry no order. Inside they are marked `A`, `B`,
`C`…, one output row per mark; your names travel from the request to the answer and never reach
the prompt, so renaming an option cannot move the answer. Two options are allowed but `noul` reads
a two-way question better, because its two rows were fitted for that question and the marks were
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
# {"answers": {"covered": {"logits": {...}}, "settle": {"logits": {...}}}, "input_tokens": 1843}
```

Any mix of the four modes in one call, keyed by names you choose; `type` is `noul`, `tfu`,
`choice` or `scale`, which the wire also spells `score`. Unlike the four methods above
this one hands back raw logits rather than a typed answer, so the softmax and the temperature of
each mode are yours to apply. The state is currently read again for each question, so a bundle
costs what the same questions cost one by one; sharing the prefix across a hybrid trunk is not
implemented yet.

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

rows = [(c.noul(t, q, true=T, false=F).logits, gold) for t, gold in my_labelled]
c = Client(temperature=calibrate(rows)["temperature"])
```

`Client(calibrated=False)` turns the shipped ones off.

## Running the service yourself

```
pip install "typecastlm[server]"
typecastlm-serve --model mihailgribov/typecastlm-qwen3.5-3.8b --port 8000
```

It downloads the weights on first start, checks that the checkpoint fits the interface, and serves
`POST /v1/typecast` and `GET /health`. `--prompt` replaces the wording it was measured with —
after which the numbers in the model card describe something else, which is why `/health` reports
where the wording came from.

The server returns raw logits and computes no softmax: the probabilities, the temperature and the
mode are the caller's business, and the same answer can be read again at another temperature
without asking anything twice.

## Using the model without any of this

The checkpoint is an ordinary classifier with 29 outputs, so `transformers` loads it directly and
`reader.py` beside the weights reads all four modes. See the
[model card](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b).

## Licence

Apache-2.0 — this package, its texts, and the model it reads (derived from Qwen3.5-4B, also
Apache-2.0). `LICENSE` and `NOTICE` carry the terms and the list of changes.
