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

- `score` accepts an ordered list of levels, the form the Jev API documents; a map still works and
  then its keys are what comes back.
- `usage.output_tokens`, always 0 — nothing is generated.
- `--api-key` on the service: a bearer token, and 401 without it. Validation failures answer 422.
- `docs/API.md` — the whole HTTP contract, laid out the way the reference it follows is.
- `model/` holds what ships beside the weights, and `scripts/sync_model.py --check` proves the
  reader there is the package's own.

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
