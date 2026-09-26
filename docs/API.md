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

`GET /v1/models`, behind the same key, lists what a request may name in `model`: the checkpoint
under its own name and the alias `typecastlm-latest`, each with a description and a release date,
the way the Jev API lists its models. A process serves one checkpoint, so the list is that one
and its alias, and a request naming anything else is answered by it anyway.

```json
{"models": [{"name": "typecastlm-qwen3.5-3.8b", "release_date": "2026-09-24",
             "description": "Jev-class decision model with open weights: noul, tfu, choice and score, one forward pass each; derived from Qwen/Qwen3.5-4B."},
            {"name": "typecastlm-latest", "release_date": "2026-09-24",
             "description": "Alias of typecastlm-qwen3.5-3.8b, the one checkpoint this service holds."}]}
```

`GET /openapi.json` is this contract as a schema — every question and answer type, every error
— and `GET /docs` renders it; the same file is committed as [openapi.json](openapi.json) for a
client generator or a reviewer who does not start the service.

`GET /health`, with no key, reports the model, its labels, the device and dtype, the prompt in
use, the state limit, the calibration and the queue, and is the place to check that a deployment
is the one your numbers came from.

Every response carries `X-Request-Id` — yours if the request sent one — and `X-Process-Time-Ms`,
the whole request as the server saw it, waiting included.

## Request body

| field | type | |
|---|---|---|
| `state` | string, object or array | required — the material every question is asked about; an object or an array is read as JSON |
| `questions` | object | required — a map of your keys to question objects; answers come back under the same keys |
| `model` | string | optional and ignored — a process serves one checkpoint, and the answer names it |

Inside a question, `instructions` and every criterion description take the same three forms as
`state` — a string, an object or an array — and may be omitted, as that API allows: a choice or a
level without a description is read by its name alone, and a yes/no question without criteria is
asked against the wording in `prompt.json`.

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

Questions in one call share one pass over the state: the prompts' common prefix — the wording,
the state — is run once, and each question's tail continues from it, so a bundle of twenty
costs one state and twenty tails, which is what `usage.input_tokens` counts. Against the
questions asked one by one the probabilities move by 0.005 on average and 0.02 at the 95th
percentile; a near-tie between two levels of a rubric can flip with them.

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
| rate limits | `429`, `529` | no `429`: nothing is metered. `529` when the line for the model is full, with `Retry-After` |

A question with no `criteria` is answered against the wording in `prompt.json` rather than
refused, the state limit is the same 32768 tokens, and `model` is accepted and ignored, so
`"model": "jev-latest"` does no harm. What is added: the question type `tfu`, `logits` on every
answer, and `calibration` on the body.

## Errors

| status | when |
|---|---|
| `401 Unauthorized` | the service was started with `--api-key` and the `Authorization` header does not carry it |
| `422 Unprocessable Entity` | no questions; an unknown `type`; a yes/no question without exactly two criteria; a `choice` or `score` with fewer than two options; more options than there are single-token marks; a body that is not the shape above |
| `529` | the model is busy and `--queue` requests are already waiting; `Retry-After` says when to come back, in seconds |
| `500` | the model failed to answer |

A `422` carries `detail` as a list, the shape the Jev API and FastAPI give a body that does not
parse, and a question that cannot be asked is reported the same way with its key in `loc`:

```json
{"detail": [{"loc": ["body", "questions", "settle"],
             "msg": "a choice question takes at least two options", "type": "value_error"}]}
```

The model answers one request at a time — two on the same GPU do not finish sooner than one
after the other — so a service of your own has no `429` but does have a line, and `529` is what
it says when the line is full. The bundled client retries it, along with 408, 429, 500, 502, 503
and 504, with exponential back-off and the `Retry-After` header honoured when one is sent.
