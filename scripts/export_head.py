#!/usr/bin/env python3
"""Write the head as a file a client can read without torch: `head.json`.

The GGUF carries the trunk only. What turns its vector into answers is 39 directions — the
three verdict rows of the classifier and the embedding rows of the 36 marks, each mark in its
two surface forms (`" A"` and `"A"`, read by the stronger) — and those are written here as
plain float lists, ~2 MB, beside `prompt.json`. A reader with `requests` alone can then do the
whole reading: prompt, vector from llama-server, 39 dot products, softmax at the mode's
temperature.

    scripts/export_head.py <checkpoint dir> ../model/head.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    from typecastlm.server import Reader

    r = Reader(sys.argv[1], device="cpu")
    W = r.model.score.weight.float().cpu()
    rows = {name: [W[i].tolist()] for i, name in enumerate(r.EXPECTED)}
    marks = r.ORDINAL                          # 0..9 then A..Z: every mark either mode may use
    missing = []
    for m in marks:
        rr = r.mark_rows(m)
        if rr is None:
            missing.append(m)
            continue
        rows[f"mark_{m}"] = rr.cpu().tolist()
    out = {"model": r.served, "hidden_size": W.shape[1], "labels": list(r.EXPECTED),
           "marks": [m for m in marks if m not in missing], "rows": rows,
           "note": "Each entry is a list of rows, one per surface form of the mark; the reading "
                   "is the maximum of row·h over them. The verdict rows are the classifier's "
                   "own, computed, not tokens. h is the trunk's last hidden state after the "
                   "final norm, unnormalised."}
    Path(sys.argv[2]).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {sys.argv[2]}: {len(rows)} names, {sum(len(v) for v in rows.values())} rows, "
          f"missing marks {missing}, {Path(sys.argv[2]).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
