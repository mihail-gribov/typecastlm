#!/usr/bin/env python3
"""Upload the GGUF files into the model repository, beside the weights, skipping what is there.

    scripts/publish_gguf.py <dir with the .gguf files>            # upload what is missing
    scripts/publish_gguf.py <dir> --status                        # what is there, then exit

One repository holds the model in every form, and a tag picks one: `llama-server -hf
<repo>:Q8_0` finds the file by the tag in its name, so there is one file per tag and the names
are fixed here. A file already in the repository with the same size is skipped, which makes a
run cut short resumable by running it again. `head.json` and the card travel with the clone
(`scripts/publish_hf.py`); this script is for the files too large for it.

The token is `HF_TOKEN`, or the `HF_API_KEY` line of the project's `.env`.
"""
import os
import sys
import time
from pathlib import Path

REPO = "mihailgribov/typecastlm-qwen3.5-3.8b"
FILES = ["typecastlm-qwen3.5-3.8b-q8_0.gguf", "typecastlm-qwen3.5-3.8b-bf16.gguf"]


def token() -> str:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit("no token: set HF_TOKEN, or put HF_API_KEY in the project's .env")


def main() -> int:
    from huggingface_hub import HfApi

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    src = Path(args[0])
    api = HfApi(token=token())
    there = {s.path: s.size for s in api.list_repo_tree(REPO, recursive=False)}
    todo = []
    for name in FILES:
        if not (src / name).exists():
            print(f"{'absent here':14s} {name}", flush=True)
            continue
        size = (src / name).stat().st_size
        state = "done" if there.get(name) == size else ("other size" if name in there else "missing")
        print(f"{state:14s} {name} {size / 1e9:.2f} GB", flush=True)
        if state != "done":
            todo.append(name)
    print(f"{len(FILES) - len(todo)}/{len(FILES)} in {REPO}", flush=True)
    if "--status" in sys.argv or not todo:
        return 0
    for name in todo:
        size = (src / name).stat().st_size
        print(f"uploading {name} ({size / 1e9:.2f} GB) …", flush=True)
        t0 = time.time()
        api.upload_file(path_or_fileobj=str(src / name), path_in_repo=name, repo_id=REPO,
                        commit_message=name)
        print(f"done {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
