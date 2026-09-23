# Changelog

## 0.1.0

First release. A client for the decision service, and only that.

* `Client.noul` — one closed question about one document, answered with a probability, its
  log-odds, and the weight of "nothing here decides it".
* `Client.ask` — the same in the service's request body, one question at a time.
* `typecastlm` command — one question over a file of documents, JSON lines in and out.
* One dependency: `requests`.

Refused on purpose rather than approximated: question types `choice` and `score`, and bundles of
more than one question. Both will arrive measured or not at all.
