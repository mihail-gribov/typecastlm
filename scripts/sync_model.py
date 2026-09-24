#!/usr/bin/env python3
"""Keep the model repository's copies in step with their sources here.

Two repositories are published from this workspace: this one, which is the package, and the model
repository next to it, which is the weights and the few files that must sit beside them. The
second carries a copy of the reader, so that the weights are usable without pip — and a copy that
nobody checks is a copy that goes stale, which is how the model repository came to ship a reader
two releases old.

    model/reader.py   <- src/typecastlm/local.py

`--check` reports drift instead of fixing it, which is what the tests call. Both are quiet about
a model repository that is not there: this package is complete without it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT.parent / "model"          # the clone of the model repository, beside this one


def pairs(model: Path) -> list[tuple[Path, Path]]:
    return [(ROOT / "src" / "typecastlm" / "local.py", model / "reader.py")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sync_model")
    ap.add_argument("--check", action="store_true", help="report drift instead of fixing it")
    ap.add_argument("--model-repo", default=str(MODEL), help=f"default: {MODEL}")
    a = ap.parse_args(argv)

    model = Path(a.model_repo)
    if not model.is_dir():
        print(f"no model repository at {model}; nothing to sync")
        return 0

    bad = 0
    for src, dst in pairs(model):
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            print(f"ok      {dst}")
        elif a.check:
            print(f"DRIFTED {dst} — run scripts/sync_model.py", file=sys.stderr)
            bad += 1
        else:
            shutil.copyfile(src, dst)
            print(f"copied  {src.relative_to(ROOT)} -> {dst}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
