#!/usr/bin/env python3
"""Upload the model repository: the card, the reader and the prompt from here, the weights from
wherever they were built.

Publication is irreversible in the sense that matters — the files become public the moment they
land — so this never runs by itself and never guesses the weights directory.

    scripts/publish_hf.py --weights ../experiments/70_base_qwen35/package_q35_marks --dry-run
    scripts/publish_hf.py --weights ../experiments/70_base_qwen35/package_q35_marks
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "mihailgribov/typecastlm-qwen3.5-3.8b"
FROM_REPO = ["model/README.md", "model/reader.py", "model/prompt.json", "LICENSE", "NOTICE"]
FROM_WEIGHTS = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="publish_hf")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--weights", default=None, help="directory holding the built checkpoint")
    ap.add_argument("--texts-only", action="store_true", help="card, reader and prompt only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    import subprocess
    if subprocess.run(["python3", str(ROOT / "scripts" / "sync_model.py"), "--check"]).returncode:
        return 1

    plan: list[tuple[Path, str]] = [(ROOT / p, Path(p).name) for p in FROM_REPO]
    if not a.texts_only:
        if not a.weights:
            ap.error("pass --weights, or --texts-only to send just the texts")
        w = Path(a.weights).resolve()
        plan += [(w / n, n) for n in FROM_WEIGHTS]

    missing = [str(src) for src, _ in plan if not src.exists()]
    if missing:
        print("missing:\n  " + "\n  ".join(missing))
        return 1

    for src, dst in plan:
        print(f"{src.stat().st_size / 1e6:10.1f} MB  {dst}")
    if a.dry_run:
        print("\ndry run, nothing sent")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    for src, dst in plan:
        print(f"uploading {dst} …", flush=True)
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=a.repo,
                        repo_type="model")
    print(f"\nhttps://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
