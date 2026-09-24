---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: text-classification
language:
  - en
tags:
  - qwen3
  - zero-shot-classification
  - decision-model
---

# typecastlm-qwen3.5-3.8b

**A Jev-class decision model with open weights.** You hand it material and a question; it answers
with numbers, not prose. Four modes, one forward pass each:

| mode | the question | the answer |
|---|---|---|
| **`noul`** | yes or no, two criteria | `p(yes)`, a softmax over those two answers |
| **`tfu`** | the same question | one distribution over `true`, `false`, **`unsure`** |
| **`choice`** | 2–16 options, one correct | a probability per option |
| **`scale`** | an ordinal rubric, up to 10 levels | a probability per level |

`noul`, `choice` and `scale` are Jev's modes under Jev's names; `tfu` is the one Jev does not
have.

**The third answer is what a two-answer reader cannot give.** `unsure` is an output of its own,
not a hedged yes: no criterion asks for it, and it carries what neither of the two criteria fits.
On FEVER it separates undecidable material from decidable with AUC 0.713. Threshold it to abstain,
to route to a human, or to drop a document from a pipeline. It is read in its own mode, with its
own temperature, and stays out of `noul`, whose probability is a softmax over the two answers
that decide the question and nothing else.

Why this one:

* **Fast** — p50 48 ms on material under 200 tokens and 566 ms at 1000–4000, on a 16 GB consumer
  card; nothing is generated, so a decision is one forward pass and no tokens written.
* **Calibrated** — a temperature per mode ships with the weights, so a probability means what it
  says.
* **Small and open** — 3.76B parameters, 7.1 GB, Apache-2.0, runs on one card.
* **Four modes** over one head, each with its own calibration.

Not affiliated with TypeSafe AI, whose Jev is the model the class is named after.

On the public tasks of [JevBench](https://github.com/fstandhartinger/jevbench) it answers 0.792
correctly, measured by us.

| | |
|---|---|
| client and server | [`typecastlm` on PyPI](https://pypi.org/project/typecastlm/) — `pip install typecastlm` |
| source | [github.com/mihail-gribov/typecastlm](https://github.com/mihail-gribov/typecastlm) |
| HTTP contract | [docs/API.md](https://github.com/mihail-gribov/typecastlm/blob/master/docs/API.md) — the Jev API, served from these weights |

## Start

```bash
pip install "typecastlm[local]" flash-linear-attention fla-core
```

```python
from typecastlm import Reader

r = Reader("mihailgribov/typecastlm-qwen3.5-3.8b")
a = r.noul(document, "Is the claim supported by the material?",
           true="the material supports it", false="the material contradicts it")
a["p"]["yes"]                      # 0.83 — over yes and no

t = r.tfu(document, "Is the claim supported by the material?",
          true="the material supports it", false="the material contradicts it")
t["p"]["unsure"]                   # 0.12 — the same reading, over three answers
```

`reader.py` in this repository is the same code, if you would rather not install the package. As
an HTTP service: `pip install "typecastlm[server]"` and `typecastlm-serve --model
mihailgribov/typecastlm-qwen3.5-3.8b`, which answers the Jev API — same route, same request body,
same answer objects — so a client written against that interface reaches it by changing the base
URL.

Or without any of that — it is an ordinary classifier:

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="mihailgribov/typecastlm-qwen3.5-3.8b",
                top_k=None, function_to_apply="none")
```

The head has 29 outputs: `true`, `false`, `unsure`, and one per answer mark (`mark_A` … `mark_P`,
`mark_0` … `mark_9`). They are answers to different questions, so take the ones your question uses
and softmax over those — never over all 29. The wording the numbers below were measured with is in
`prompt.json`.

## Modes

Every mode takes the material and a question and differs in what the answer ranges over. One
forward pass each; each has its own temperature in `prompt.json`.

### `noul`: yes or no

```python
r.noul(policy, "Is the claim covered?",
       true="the policy covers it", false="the policy excludes it")
# {"p": {"yes": 0.83, "no": 0.17}, "logits": {...}}
```

Two criteria, one per side, and a softmax over the two answers they name, at the `verdict`
temperature and calibrated for a threshold at 0.5. The third output takes no part in it. Outputs
read: `true`, `false`.

### `tfu`: yes, no, or neither

```python
r.tfu(policy, "Is the claim covered?",
      true="the policy covers it", false="the policy excludes it")
# {"p": {"true": 0.62, "false": 0.22, "unsure": 0.16}, "verdict": "true", "logits": {...}}
```

The same two criteria and the same forward pass, softmaxed over three answers at the
`three_answers` temperature. No criterion is written for `unsure`: it carries what neither of the
two fits. The two modes are fitted separately, so a number from one is not a number from the
other — read whichever mode you act on, and do not divide the logits of one by the temperature of
the other. Outputs read: `true`, `false`, `unsure`.

### `choice`: one of several options

```python
r.choice(policy, "How should the claim be settled?",
         [("deny", "excluded as repeated seepage"),
          ("sublimit", "covered but capped by the concealed-water sublimit"),
          ("pay", "covered in full")])
# {"p": {"deny": 0.01, "sublimit": 0.97, "pay": 0.02}, "marks": {...}, "logits": {...}}
```

Two to sixteen options, exactly one correct, no order among them. Option names are yours and do
not reach the prompt: inside, options are marked `A`, `B`, `C`… and `marks` says which got which.
Outputs read: `mark_A` … one per option.

### `scale`: a level on an ordinal rubric

```python
r.scale(review, "How positive is this review?",
        [("0", "very negative"), ("1", "negative"), ("2", "neutral"),
         ("3", "positive"), ("4", "very positive")])
# {"p": {"0": 0.06, "1": 0.08, "2": 0.11, "3": 0.54, "4": 0.20}, ...}
```

Levels are ordered and marked with their own digits when those are single characters, otherwise
with `0…9ABC…`. Up to ten levels; beyond that the marks read poorly. Outputs read: `mark_0` …
one per level.

### Several questions about one state

```python
r.noul_many(policy, [(q1, true1, false1), (q2, true2, false2)])
```

Returns one `noul` result per question. The state is read again for each, so a bundle costs what the
questions cost separately.

## Numbers

[JevBench v1.2](https://github.com/fstandhartinger/jevbench), public set, 231 tasks, measured
here:

| | accuracy |
|---|---|
| all 231 tasks | **0.792** |
| choice / verdict / scale | 0.820 / 0.743 / 0.778 |
| easy / standard / hard | 1.000 / 0.917 / 0.622 |

Public pools, validation splits:

| benchmark | rows | AUC | accuracy @0.5 | calibration error |
|---|---|---|---|---|
| SuperGLUE BoolQ | 3000 | 0.939 | 0.881 | 0.060 |
| SuperGLUE RTE | 277 | 0.955 | 0.892 | 0.042 |
| SuperGLUE WiC | 638 | 0.702 | 0.627 | 0.101 |
| FEVER (NLI form) | 2000 | 0.943 | 0.884 | 0.036 |

On FEVER the third answer separates `NOT ENOUGH INFO` from decidable rows with AUC 0.713, and
three-class accuracy is 0.668.

**Speed**, one decision at a time on a 16 GB consumer card. What sets it is the length of the
material, not the question:

| material | p50 | p90 |
|---|---|---|
| under 200 tokens | 48 ms | 75 ms |
| 200–1000 | 93 ms | 140 ms |
| 1000–4000 | 566 ms | 666 ms |

Over the 74 binary JevBench tasks as they come: p50 52 ms, p95 220 ms, p99 659 ms. Install
`flash-linear-attention` and `fla-core`, or the trunk falls back to a slow path and p50 triples.

## Calibration

The values are in `prompt.json` under `calibration`, one temperature per mode, applied by
`reader.py` and by the client. What each was fitted on, since that is what decides whether it
carries over to your material:

| mode | key in `prompt.json` | fitted on |
|---|---|---|
| `noul` | `verdict` | FEVER dev, decidable rows — claims against evidence |
| `tfu` | `three_answers` | FEVER dev, a third of it undecidable by construction |
| `choice` | `choice` | ARC-Challenge — four options, knowledge questions |
| `scale` | `scale` | SST-5 — five ordered levels of sentiment |

A temperature changes no answer, only the probability. If your pool is harder or easier than
those, refit on a couple of hundred labelled rows with `typecastlm.calibrate`, or read the logits
and skip it.

## Licence

Apache-2.0 — the weights, inherited from
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), and the texts here. `LICENSE` carries the
licence, `NOTICE` states what was changed.
