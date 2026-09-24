# Changelog

## 1.2.0 — unreleased

The service grows into a deployment, and the rest of the Jev API arrives.

**Added**

- A Docker image and a compose file: `docker compose up -d --build` serves the API on a GPU host
  with the libraries pinned to the versions the numbers were measured with, and the weights in a
  volume that survives rebuilds. A second compose layer runs the same image on a CPU. The whole
  of it is in `docs/DOCKER.md`.
- `GET /v1/models`, behind the same key, listing the checkpoint and the alias
  `typecastlm-latest` in the shape that API gives a model.
- Every `typecastlm-serve` flag is also `TYPECASTLM_<FLAG>` in the environment; the container is
  configured that way and runs the command with no arguments.
- The model answers one request at a time and the rest wait, up to `--queue` of them (32); past
  that the service says `529` with `Retry-After`, which the client already retried on.
- `instructions` and criterion descriptions take a string, an object, an array or nothing, as
  the Jev API allows: an option without a description is read by its name alone. Until now an
  object was a `500` and a missing description a `TypeError`.
- A `422` for a question that cannot be asked now names it: `detail` is the list FastAPI gives a
  malformed body, with the question's key in `loc`, so a bundle of twenty is refused with the one
  that is wrong.
- `X-Request-Id` and `X-Process-Time-Ms` on every response; `/health` reports dtype, version and
  the queue's load.

**Changed**

- The answer's `model` is the checkpoint's short name (`typecastlm-qwen3.5-3.8b`), never a
  filesystem path, whichever way the weights were loaded.
- `--api-key` takes several tokens separated by commas.
- `transformers>=5` in the `local` and `server` extras: the checkpoint's tokenizer is saved in
  its format and 4.x cannot load it. `jinja2` and `accelerate` are listed too, since transformers
  stopped pulling the first in and the second is what `device_map` wants.

**Fixed**

- On a CPU with `flash-linear-attention` installed the trunk called its Triton kernels and failed
  with `0 active drivers`. The service now hides the package when it is not on CUDA, and the same
  image serves both.

## 1.1.2

**Fixed**

- The service refused a `state` that was not a string. The API this follows takes a string, an
  object or an array, and a caller sending structured material got a 422 — on a public suite of
  231 decisions that was 35 of them. An object or an array is now rendered as JSON, which is what
  the readers were measured on.

**Added**

- `choice` and `score` answers carry `marks`: which letter or digit each option was given. The
  mark is the option's position, so renaming an option cannot move the answer but reordering can,
  and until now nothing over HTTP made that visible. Measured on a public suite: reordering the
  options of 213 questions flipped 29 verdicts. README and `docs/API.md` now say so, and
  normalising the per-letter bias does not fix it — the letter and the position are the same
  number, and subtracting it removes one flip in eighteen.

## 1.1.1

`NOTICE` only — no code changed. It still described the head as 29 rows, which it stopped being
when the checkpoint was repacked to carry all 26 letters; a statement of what a derivative changed
is the one file that has to be right. It now says 39 rows and 36 marks, and names where the
wording and the temperatures live.

## 1.1.0

`noul` was reading three logits where two decide the question, and the third at the wrong
temperature. It is now a softmax over the two criteria, which is what the mode means and what the
API it copies returns. The service, meanwhile, answered in a shape of its own; it now answers in
Jev's, field for field.

**Breaking**

- `Answer.unknown` and `Answer.p` are gone. `noul` answers over two criteria; the third answer is
  `tfu`, a mode with its own temperature. A verdict and an abstention are two readings of one pass
  and were never one number.
- `Reader.ask` is `Reader.noul` and `Reader.ask_many` is `Reader.noul_many`, so every mode carries
  the name the API uses. The old names still work.
- The service answers `POST /v1/systemone` with `{"type": ..., "noul"|"choice"|"score"|"tfu": ...,
  "probabilities", "confidence", "legend"}` instead of `{"kind", "logits"}`. `logits` is still
  there on every answer, beside those fields rather than instead of them. `/v1/typecast` still
  routes to the same handler.

**Added**

- `choice` takes up to 26 options, where it took 16. A mark is read by the model's own output row
  for that token, which is what the head caches, so a mark the head lacks is still readable: the
  reader and the service take the row from the embedding and get the same number to the last bit.
  The checkpoint was repacked to carry all 26 letters as well, which is what the plain
  `text-classification` pipeline reads, and split into shards so that the next change to the head
  costs the 195 KB it weighs rather than 7.5 GB.
- `score` accepts an ordered list of levels, the form the Jev API documents; a map still works and
  then its keys are what comes back.
- `usage.output_tokens`, always 0 — nothing is generated.
- `Choice.score` on a `scale` answer: the mean level, which the service was already returning and
  the client dropped.
- `tests/live_service.py` — the client against a running service: four modes, a bundle, the
  refusals and the key. Not part of `pytest`, because it needs weights.

**Fixed**

- `TYPECASTLM_ENDPOINT=http://localhost:8000` — the address anyone writes after starting the
  service — posted to the bare host and got a 404: the client never added the route. A host
  without a path now gets `/v1/systemone`; an endpoint that carries a path of its own is left
  alone, since a gateway may mount the service anywhere.
- The marks mode read the wrong row and asked the wrong question. A mark appears as `" A"` and as
  `"A"`, and the reading takes the stronger of the two — which is what the temperatures were
  fitted with, and what the head could not express, since it carries one row per mark. The
  wording drifted too: the service and the reader asked a rubric with named levels two different
  questions, and neither was the one the numbers were measured with.
- The wording of the marks mode now travels with the weights, under `marks` in `prompt.json`,
  where the two-criteria wording already was. `tests/same_reading.py` compares the prompt the
  reader builds with the prompt the service builds, character by character.
- The client never honoured `Retry-After`: the value was read into a variable that had just been
  set to `None`, so a service asking for a longer pause got the doubling instead.
- The connection pool was mounted for `https://` only, so a local service — which is the usual
  one, since there is no hosted endpoint — fell back to the default pool.
- `--api-key` on the service: a bearer token, and 401 without it. Validation failures answer 422.
- `docs/API.md` — the whole HTTP contract, laid out the way the reference it follows is, and a
  section naming every difference a caller moving from Jev would meet.
- `scripts/sync_model.py --check` proves that the reader shipped beside the weights is this
  package's own, and `scripts/publish_hf.py` pushes the model repository from the clone next to
  this one. `docs/RELEASING.md` says where each repository lives and what goes into it.

## 1.0.0

The first release, and a major one because the interface is now whole: all four question types are
implemented, so there is nothing left that a caller would have to work around. What this version
promises to keep:

- the question types `noul`, `tfu`, `choice` and `score`, and the shape of what they return;
- the contract with a checkpoint — the first three head outputs are `true`, `false`, `unsure` in
  that order, and answer marks follow as `mark_*`;
- a temperature per mode in `prompt.json`, applied client-side.

Breaking any of those would be 2.0.0.

- **Three modes.** `choice` (two to sixteen options) and `scale` (an ordinal rubric) join `noul`
  and `tfu`. They are not approximations: the checkpoint carries a head row per answer mark, so a
  choice costs the same one pass as a verdict.
- **New checkpoint**: `mihailgribov/typecastlm-qwen3.5-3.8b` — Qwen3.5-4B cut at block 27, a head
  of 29 rows (three answers and 26 marks).
- **A temperature per mode**, shipped with the weights and applied client-side.
- **Per-answer shifts removed.** The third row of the head is now a direction computed from data
  and scaled to the other two; the offset the shifts corrected went with the old construction.
  Fitting them on top now buys 0.005 of calibration error and no accuracy.
- `calibrate()` fits one temperature by minimising calibration error, not cross-entropy — the
  latter chose 5.63 where 2.85 is right, and made the error larger.

## 0.1.0

First release. A client for the decision service, and only that.

* `Client.noul` — one closed question about one document, answered with a probability, its
  log-odds, and the weight of "nothing here decides it".
* `Client.ask` — the same in the service's request body, one question at a time.
* `typecastlm` command — one question over a file of documents, JSON lines in and out.
* One dependency: `requests`.

Refused on purpose rather than approximated: question types `choice` and `score`, and bundles of
more than one question. Both will arrive measured or not at all.
