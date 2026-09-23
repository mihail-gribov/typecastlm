# typecastlm

**A client for a Jev-class decision model with open weights**: ask a document a closed question,
get numbers back. The same interface that class already speaks, and one answer more — besides yes
and no it returns **"nothing here decides it"**, which no question has to ask for. Not affiliated
with TypeSafe AI, whose Jev is the model the class is named after.

Why this one:

* **Fast** — 52 ms per decision, because nothing is generated: one forward pass, no tokens
  written.
* **It can say "I can't tell"** — a third answer that the question never asks for.
* **Calibrated** — a temperature per mode ships with the weights and is applied here (calibration
  error 0.011–0.052).
* **Thin client** — one dependency, `requests`, and it installs in a second. The weights live on
  the service side.
* **Three shapes of question**: yes/no, one-of-several, and a level on a rubric.

## Install

```
pip install typecastlm
export TYPECASTLM_ENDPOINT=https://…      # optional, has a default
export TYPECASTLM_API_KEY=…
```

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
a.margin     # +2.87 — the log-odds, so a saturated probability still ranks
```

## Three shapes of question

```python
c.tfu(doc, "Is the claim supported?", true="…", false="…").p
# {'true': 0.81, 'false': 0.11, 'unsure': 0.08}

c.choice(policy, "How should this claim be settled?",
         {"deny_vacancy": "…", "pay_with_sublimit": "…", "pay_in_full": "…"})
# .verdict 'pay_with_sublimit'   .p {...}   .confidence 0.74

c.scale(review, "How positive is this review overall?",
        {"0": "very negative", "1": "negative", "2": "neutral",
         "3": "positive", "4": "very positive"})
# .verdict '3'   .p {...}
```

The option names are yours: they travel from your request to your answer and never reach the
prompt.

## Calibration

Each mode carries its own temperature, shipped beside the weights and applied by the client:

| mode | temperature | calibration error |
|---|---|---|
| verdict | 1.30 | 0.032 → 0.011 |
| three answers | 2.85 | 0.236 → 0.052 |
| choice | 1.75 | 0.070 → 0.030 |
| scale | 3.90 | 0.329 → 0.016 |

A temperature changes no answer — `argmax` is invariant to it — only what a probability means, and
it does not transfer between pools. Fit your own on a couple of hundred labelled rows:

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

## Many questions, one reading

```python
c.ask(policy, {
    "covered":  {"type": "noul",   "instructions": "Is the claim covered?",
                 "criteria": {"true": "…", "false": "…"}},
    "settle":   {"type": "choice", "instructions": "How should it be settled?",
                 "criteria": {"deny": "…", "pay": "…"}},
})
```

The material is read once for the whole bundle and each question costs only its own tail — about
four times faster than asking one by one on a policy of three thousand tokens, and the numbers are
identical.

## Using the model without any of this

The checkpoint is an ordinary classifier with 29 outputs, so `transformers` loads it directly and
`reader.py` beside the weights reads all three modes. See the
[model card](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b).

## Licence

Apache-2.0 — this package, its texts, and the model it reads (derived from Qwen3.5-4B, also
Apache-2.0). `LICENSE` and `NOTICE` carry the terms and the list of changes.
