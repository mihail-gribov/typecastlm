# Changelog

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
