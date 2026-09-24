# API reference

The service answers questions about one state. It follows the shape of the Jev API — same
endpoint layout, same request body, same question types under the same names — with one question
type added, `tfu`, and one deliberate difference in the answer: this service returns logits and
the calibration rather than finished probabilities.

## Evaluation endpoint

```
POST http://<host>:<port>/v1/typecast
Content-Type: application/json
Authorization: Bearer <key>        # only if your deployment adds auth; typecastlm-serve has none
```

`GET /health` reports the model, its labels, the device, the prompt in use, the state limit and
the calibration, and is the place to check that a deployment is the one your numbers came from.

## Request body

| field | type | |
|---|---|---|
| `state` | string | required — the material every question is asked about |
| `questions` | object | required — a map of your keys to question objects; answers come back under the same keys |
| `model` | string | optional and ignored — a process serves one checkpoint, and the answer names it |

A state longer than `max_state_tokens` (32768 for this checkpoint) is not refused: the middle is
dropped and the two halves are kept with ` […] ` between them, because a question about a
truncated head alone is answered confidently and wrongly.

## Question types

Every question carries `instructions`, the question itself, and `criteria`, which say what the
answers mean. Option and level names are yours: they key the answer and never reach the prompt.

### `noul` — yes or no

Exactly two criteria. The two answers decide the question between them.

```json
{"state": "…", "questions": {"covered": {
  "type": "noul",
  "instructions": "Is the claim covered by this policy?",
  "criteria": {"true": "the policy covers it", "false": "the policy excludes it"}}}}
```

### `tfu` — yes, no, or neither

The same two criteria as `noul`, read over three answers instead of two. The third is not a
criterion anyone writes: it is what is left when neither of the two fits. Jev has no such type.

```json
{"type": "tfu",
 "instructions": "Does the material support the claim?",
 "criteria": {"true": "the material supports it", "false": "the material contradicts it"}}
```

### `choice` — one of several options

`criteria` is a map of options to descriptions, two to sixteen of them, exactly one correct and no
order among them. Sixteen is where this checkpoint's marks end; Jev takes up to 255.

```json
{"type": "choice",
 "instructions": "How should this claim be settled?",
 "criteria": {"deny_vacancy": "excluded as repeated seepage",
              "pay_with_sublimit": "covered but capped by the concealed-water sublimit",
              "pay_in_full": "covered in full"}}
```

### `score` — a level on an ordinal rubric

`criteria` is a map of levels to descriptions, in order, at least two. Ten is the documented
limit, Jev's as well: the checkpoint carries marks for more and the service will accept them, but
the reading degrades past ten. `scale` is accepted as a synonym of `score`.

```json
{"type": "score",
 "instructions": "How positive is this review overall?",
 "criteria": {"0": "very negative", "1": "negative", "2": "neutral",
              "3": "positive", "4": "very positive"}}
```

## Response body

```json
{"answers": {"covered": {"kind": "noul",
                         "logits": {"true": 3.1, "false": -0.4, "unsure": -2.2}}},
 "usage": {"input_tokens": 131},
 "model": "typecastlm-qwen3.5-3.8b",
 "calibration": {"verdict": {"temperature": 1.3}, "three_answers": {"temperature": 2.85},
                 "choice": {"temperature": 1.75}, "scale": {"temperature": 3.9}}}
```

| field | |
|---|---|
| `answers` | one entry per question, under the key you sent |
| `usage.input_tokens` | tokens read, summed over the questions in the call |
| `model` | the checkpoint that answered |
| `calibration` | the temperature per mode, shipped with the weights; it belongs to the checkpoint, so a client does not have to guess it |

Questions in one call are answered against the same state but read it separately, so a bundle
costs what the questions cost one by one.

## Answer types

Every answer carries `kind`, echoing the question type, and `logits`, one raw number per answer.
Probabilities are the caller's: divide by the mode's temperature and softmax over the answers that
mode uses. Which answers those are is the whole difference between `noul` and `tfu`.

| kind | logits | how to read them |
|---|---|---|
| `noul` | `true`, `false`, `unsure` | softmax over `true` and `false` at the `verdict` temperature; `unsure` takes no part |
| `tfu` | `true`, `false`, `unsure` | softmax over all three at the `three_answers` temperature |
| `choice` | one per option | softmax over the options at the `choice` temperature |
| `score` | one per level | softmax over the levels at the `scale` temperature |

`noul` and `tfu` read the same three logits and differ only in what the softmax runs over, so one
call of either can be re-read as the other — but not with the other's temperature, which is fitted
for its own mode.

Returning logits rather than probabilities is what lets an answer be read twice without asking
twice, and what keeps the temperature visible instead of baked in. A client that wants Jev's
shape — `{"noul": 0.94}` — computes it in one line from `logits`.

## Errors

| status | when |
|---|---|
| 400 | no questions; an unknown `type`; a yes/no question without exactly two criteria; a `choice` or `score` with fewer than two options; more options than the checkpoint has marks |
| 422 | a body that is not the shape above |
| 500 | the model failed to answer |

408, 429, 500, 502, 503, 504 and 529 are the codes the bundled client retries, with exponential
back-off and the `Retry-After` header honoured when one is sent.
