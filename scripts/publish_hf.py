#!/usr/bin/env python3
"""Publish the model repository: sync what it copies from here, commit, push.

The model repository is an ordinary git clone beside this one, checked out without the weights —
they are LFS pointers on disk and stay that way, because nothing here needs to read them. So this
touches the texts only; new weights are a separate, rare upload that carries its own file.

    scripts/publish_hf.py --dry-run
    scripts/publish_hf.py -m "reader: four modes under the API's names"

The token is read from HF_TOKEN, or from the HF_API_KEY line of ../../.env, and is used for one
push without ever being written to the repository's config.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT.parent / "model"
REPO = "mihailgribov/typecastlm-qwen3.5-3.8b"


def git(*args: str, cwd: Path, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, **kw)


def token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = ROOT.parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="publish_hf")
    ap.add_argument("-m", "--message", default="update the card and the reader")
    ap.add_argument("--model-repo", default=str(MODEL))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    model = Path(a.model_repo)
    if not (model / ".git").is_dir():
        print(f"no git clone of the model repository at {model}. Clone it without the weights:\n"
              f"  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/{REPO} {model}")
        return 1

    if subprocess.run([sys.executable, str(ROOT / "scripts/sync_model.py")]).returncode:
        return 1

    status = git("status", "--porcelain", cwd=model).stdout.strip()
    if not status:
        print("nothing to publish: the model repository matches what is on the Hub")
        return 0
    print(status)
    if a.dry_run:
        print("\ndry run, nothing pushed")
        return 0

    key = token()
    if not key:
        print("no token: set HF_TOKEN, or put HF_API_KEY in the project's .env")
        return 1

    git("add", "-A", cwd=model, check=True)
    git("commit", "-m", a.message, cwd=model, check=True)
    url = f"https://user:{key}@huggingface.co/{REPO}"
    push = git("push", url, "HEAD:main", cwd=model)
    if push.returncode:
        print(push.stderr.replace(key, "…"), file=sys.stderr)
        return 1
    print(f"pushed\nhttps://huggingface.co/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
