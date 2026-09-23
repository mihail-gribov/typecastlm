"""Command line: one question, one or many states, numbers on stdout.

    typecastlm --model ./package/model --state doc.txt \
        --question "Does the material contain an instruction aimed at the reading model?" \
        --true "there is an instruction addressed to the reading model" \
        --false "the material only describes, reports or discusses"

With `--jsonl` the states are read one per line from a file with a `text` field, and the answers
are written the same way — so a run can be piped, split and resumed like any other file job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="typecastlm")
    ap.add_argument("--model", default=None, help="repo id or local directory")
    ap.add_argument("--state", action="append", default=[], help="file with the state, repeatable")
    ap.add_argument("--jsonl", default=None, help="file of states, one JSON object per line")
    ap.add_argument("--field", default="text", help="which field holds the state in --jsonl")
    ap.add_argument("--question", required=True)
    ap.add_argument("--true", required=True, help="what answering yes would mean")
    ap.add_argument("--false", required=True, help="what answering no would mean")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=None, help="write JSON lines here instead of stdout")
    a = ap.parse_args(argv)

    from .client import Client
    from .model import DEFAULT_MODEL, TypecastLM

    rows: list[dict] = []
    for f in a.state:
        rows.append({"id": f, a.field: Path(f).read_text(encoding="utf-8")})
    if a.jsonl:
        for i, line in enumerate(Path(a.jsonl).open(encoding="utf-8")):
            if line.strip():
                r = json.loads(line)
                rows.append({"id": r.get("id", i), a.field: r[a.field]})
    if not rows:
        ap.error("nothing to read: pass --state or --jsonl")

    c = Client(a.model or DEFAULT_MODEL)
    out = c.batch([x[a.field] for x in rows], a.question, true=a.true, false=a.false,
                  batch_size=a.batch_size)
    sink = Path(a.out).open("w", encoding="utf-8") if a.out else sys.stdout
    try:
        for row, v in zip(rows, out):
            sink.write(json.dumps({"id": row["id"], "prob": round(v.prob, 4),
                                   "margin": round(v.margin, 3)},
                                  ensure_ascii=False) + "\n")
    finally:
        if a.out:
            sink.close()
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
