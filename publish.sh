#!/usr/bin/env bash
# Publishing to PyPI. This never runs by itself: publication is irreversible — a version
# number on PyPI is taken forever and cannot be re-uploaded.
#
#   ./publish.sh test     -> TestPyPI, for a rehearsal
#   ./publish.sh          -> PyPI
set -eu
cd "$(dirname "$0")"

rm -rf dist
uv build
ls -la dist/

if [ "${1:-}" = "test" ]; then
    uv publish --publish-url https://test.pypi.org/legacy/ dist/*
else
    echo "This publishes typecastlm 0.1.0 to PyPI. Ctrl-C if not now."
    read -r -p "Continue? [y/N] " a
    [ "$a" = "y" ] || exit 1
    uv publish dist/*
fi
