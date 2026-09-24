#!/usr/bin/env python3
"""Write the service's OpenAPI schema to docs/openapi.json, without loading any weights.

    scripts/export_openapi.py            # writes the file
    scripts/export_openapi.py --check    # exits 1 if the file is stale

The schema is rendered from `typecastlm.schema` and the routes in `typecastlm.server`; the
reader is a stub with the few attributes the routes read at import time. The committed file
is what a client generator or a reviewer reads without starting the service, and the test in
tests/test_smoke.py keeps it in step with the code.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typecastlm.server import Reader, build_app  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"


def schema() -> dict:
    r = object.__new__(Reader)
    r.name, r.served = "mihailgribov/typecastlm-qwen3.5-3.8b", "typecastlm-qwen3.5-3.8b"
    r.labels, r.device, r.prompt_source, r.max_state_tokens = [], "cuda:0", "model repository", 0
    r.prompt_cfg = {}
    return build_app(r).openapi()


def main() -> int:
    text = json.dumps(schema(), indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        if OUT.exists() and OUT.read_text(encoding="utf-8") == text:
            print(f"{OUT.name}: up to date")
            return 0
        print(f"{OUT.name}: stale — run scripts/export_openapi.py", file=sys.stderr)
        return 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
