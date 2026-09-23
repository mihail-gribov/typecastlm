"""The fallback prompt.

A checkpoint carries its own `prompt.json`, and that copy is the one that counts: the wording and
the weights were measured together, and a second copy here would drift away from it. What stays is
a fallback for a checkpoint that predates the file, and the defaults for the two criteria when a
caller leaves them out.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent / "config.json"


def defaults() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))["prompt"]
