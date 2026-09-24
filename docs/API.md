# API reference

The service answers questions about one state. The route, the request body, the answer objects
and the error codes are the Jev API's, field for field, so a client written against it reaches
this service by changing the base URL. Three things are added and nothing is changed: the question
type `tfu`, a `logits` field on every answer, and the checkpoint's `calibration` on the body.

## Evaluation endpoint

```
POST http://<host>:<port>/v1/systemone
Content-Type: application/json
Authorization: Bearer <key>        # required when the service was started with --api-key
```

`/v1/typecast` is the same route under the name this service answered to in 1.0.0.

`GET /health` reports the model, its labels, the device, the prompt in use, the state limit and
the calibration, and is the place to check that a deployment is the one your numbers came from.

## Request body

| field | type | |
|---|---|---|
| `state` | string | required — the material every question is asked about |
| `questions` | object | required — a map of your keys to question objects; answers come back under the same keys |
| `model` | string | optional and ignored — a process serves one checkpoint, and the answer names it |

Answers come back under your keys, in one body, however many questions the call carries.

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

`criteria` is a map of options to descriptions, two to twenty-six of them, exactly one correct
and no order among them. Twenty-six is where single-token marks end; Jev takes up to 255.

```json
{"type": "choice",
 "instructions": "How should this claim be settled?",
 "criteria": {"deny_vacancy": "excluded as repeated seepage",
              "pay_with_sublimit": "covered but capped by the concealed-water sublimit",
              "pay_in_full": "covered in full"}}
```

### `score` — a level on an ordinal rubric

`criteria` is an ordered list of levels, at least two of them. Ten is the recommended limit and
the one Jev enforces; this service accepts up to 36, because the marks run `0…9` and then `A…Z`.
Past nine the reading degrades — on levels 10–14 accuracy is 0.05 against 0.215 on 0–9 — so the
extra levels are there for a rubric you already have, not as a reason to make one. `scale` is
accepted as a synonym of `score`.

```json
{"type": "score",
 "instructions": "How frustrated is the customer?",
 "criteria": ["Calm", "Frustrated", "Very angry"]}
```

A list gives the levels the positions `0`, `1`, `2`… and those positions key the answer. A map is
accepted too, and then its keys are the ones that come back — useful when the levels already have
names in your code.

## Response body

```json
{"model": "typecastlm-qwen3.5-3.8b",
 "answers": {"is_urgent": {"type": "noul", "noul": 0.95,
                           "logits": {"true": 3.1, "false": -0.4, "unsure": -2.2}}},
 "usage": {"input_tokens": 307, "output_tokens": 0},
 "calibration": {"verdict": {"temperature": 1.3}, "three_answers": {"temperature": 2.85},
                 "choice": {"temperature": 1.75}, "scale": {"temperature": 3.9}}}
```

| field | |
|---|---|
| `model` | the checkpoint that answered |
| `answers` | one object per question, under the key you sent |
| `usage.input_tokens` | tokens read, summed over the questions in the call |
| `usage.output_tokens` | always 0 — nothing is generated |
| `calibration` | added here: the temperature per mode, so a caller reading `logits` does not have to guess the one the probabilities were read at |

Questions in one call are answered against the same state but read it separately, so a bundle
costs what the questions cost one by one.

## Answer types

Every answer carries `type`, the probabilities its mode is read at, and `logits` — the raw head
outputs the question used, which no other field lets you recover.

### `noul`

```json
{"type": "noul", "noul": 0.95, "logits": {"true": 3.1, "false": -0.4, "unsure": -2.2}}
```

`noul` is the probability of the `true` criterion, a softmax over `true` and `false` at the
`verdict` temperature. The third output is in `logits` and takes no part in the number.

### `tfu`

```json
{"type": "tfu", "tfu": "unsure",
 "probabilities": {"true": 0.21, "false": 0.16, "unsure": 0.63}, "confidence": 0.63,
 "logits": {"true": 3.1, "false": -0.4, "unsure": -2.2}}
```

The same three logits read as one distribution at the `three_answers` temperature. `tfu` names the
leading answer and `confidence` is its probability. The two modes are fitted separately, so one
call can be re-read as the other only at that other mode's temperature.

### `choice`

```json
{"type": "choice", "choice": "billing",
 "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0}, "confidence": 0.81,
 "marks": {"billing": "A", "technical": "B", "sales": "C"},
 "logits": {"billing": 4.1, "technical": 1.9, "sales": -3.0}}
```

Keys are your option names throughout; inside they are marked `A` to `Z` **in the order the
`criteria` map lists them**, and `marks` in the answer says which option got which. The mark is
the position, so renaming an option cannot move the answer and reordering can: on the benchmark's
213 option tasks, reordering flipped 29 verdicts. Fix the order if you want readings that compare
across documents.

### `score`

```json
{"type": "score", "score": 1.05,
 "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
 "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}, "confidence": 0.92,
 "logits": {"0": -3.0, "1": 4.0, "2": 1.0}}
```

`marks` names the mark each level was given. `score` is the mean level, `sum(i * p(i))` over the levels in the order you gave them, so it lies
between 0 and one less than the number of levels and lands between levels: 1.05 is just past
`Frustrated`. Positions carry the arithmetic, never the mark, so a level marked `A` because it is
the eleventh counts as 11. `legend` says what each position means, which is what makes that number readable.

## Coming from Jev

A client written against that API reaches this service by changing the base URL. Everything it
sends is understood and everything it reads is there. Four differences, all of them in what is
accepted rather than in what comes back:

| | Jev | here |
|---|---|---|
| `choice` options | up to 255 | up to 26 — a request with more is refused, because a mark has to be one token |
| `score` levels | up to 10 | up to 36 accepted, 10 recommended |
| `usage.output_tokens` | counts the answer | always 0: nothing is generated |
| rate limits | `429`, `529` | a service of your own has none |

A question with no `criteria` is answered against the wording in `prompt.json` rather than
refused, the state limit is the same 32768 tokens, and `model` is accepted and ignored, so
`"model": "jev-latest"` does no harm. What is added: the question type `tfu`, `logits` on every
answer, and `calibration` on the body.

## Errors

| status | when |
|---|---|
| `401 Unauthorized` | the service was started with `--api-key` and the `Authorization` header does not carry it |
| `422 Unprocessable Entity` | no questions; an unknown `type`; a yes/no question without exactly two criteria; a `choice` or `score` with fewer than two options; more options than there are single-token marks; a body that is not the shape above |
| `500` | the model failed to answer |

A service of your own has no rate limit, so `429` and `529` do not arise here; the bundled client
retries them anyway, along with 408, 500, 502, 503 and 504, with exponential back-off and the
`Retry-After` header honoured when one is sent.
