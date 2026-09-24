#!/usr/bin/env python3
"""Regenerate the files that ship beside the weights from the package they belong to.

`model/reader.py` is the package's local reader with no package around it, so that the model
repository stays usable without pip. One source, one copy, and `--check` to prove they have not
drifted apart — which is what the tests call.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [(ROOT / "src" / "typecastlm" / "local.py", ROOT / "model" / "reader.py")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sync_model")
    ap.add_argument("--check", action="store_true", help="report drift instead of fixing it")
    a = ap.parse_args(argv)

    bad = 0
    for src, dst in PAIRS:
        same = dst.exists() and dst.read_bytes() == src.read_bytes()
        if same:
            print(f"ok      {dst.relative_to(ROOT)}")
            continue
        if a.check:
            print(f"DRIFTED {dst.relative_to(ROOT)} — run scripts/sync_model.py", file=sys.stderr)
            bad += 1
        else:
            shutil.copyfile(src, dst)
            print(f"copied  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
