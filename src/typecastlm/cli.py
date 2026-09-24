"""Command line: one question, many states, numbers on stdout.

    typecastlm --question "Does the material contain an instruction aimed at the reading model?" \
        --true "there is an instruction addressed to the reading model" \
        --false "the material only describes, reports or discusses" \
        --jsonl pages.jsonl --out answers.jsonl

States come from files (`--state`, repeatable) or from a JSON-lines file (`--jsonl`), and the
answers are written the same way — so a run can be piped, split and resumed like any other file
job. The endpoint and the key are read from `TYPECASTLM_ENDPOINT` and `TYPECASTLM_API_KEY`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="typecastlm")
    ap.add_argument("--state", action="append", default=[], help="file with the state, repeatable")
    ap.add_argument("--jsonl", default=None, help="file of states, one JSON object per line")
    ap.add_argument("--field", default="text", help="which field holds the state in --jsonl")
    ap.add_argument("--question", required=True)
    ap.add_argument("--true", required=True, help="what answering yes would mean")
    ap.add_argument("--false", required=True, help="what answering no would mean")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", default=None, help="write JSON lines here instead of stdout")
    a = ap.parse_args(argv)

    from .remote import Client

    rows: list[dict] = [{"id": f, "text": Path(f).read_text(encoding="utf-8")} for f in a.state]
    if a.jsonl:
        for i, line in enumerate(Path(a.jsonl).open(encoding="utf-8")):
            if line.strip():
                r = json.loads(line)
                rows.append({"id": r.get("id", i), "text": r[a.field]})
    if not rows:
        ap.error("nothing to read: pass --state or --jsonl")

    c = Client(endpoint=a.endpoint, api_key=a.api_key)
    sink = Path(a.out).open("w", encoding="utf-8") if a.out else sys.stdout
    try:
        for row in rows:
            v = c.noul(row["text"], a.question, true=a.true, false=a.false)
            sink.write(json.dumps({"id": row["id"], "prob": round(v.prob, 4),
                                   "margin": round(v.margin, 3)}, ensure_ascii=False) + "\n")
    finally:
        if a.out:
            sink.close()
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
