#!/usr/bin/env bash
# Выкладка на PyPI. НЕ запускается сама — это публикация, и она необратима:
# номер версии на PyPI занимается навсегда, перезалить его нельзя.
#
#   ./publish.sh test     -> TestPyPI, для проверки
#   ./publish.sh          -> PyPI
set -eu
cd "$(dirname "$0")"

rm -rf dist
uv build
ls -la dist/

if [ "${1:-}" = "test" ]; then
    uv publish --publish-url https://test.pypi.org/legacy/ dist/*
else
    echo "Это выкладка на PyPI под именем typecastlm 0.1.0. Ctrl-C, если не сейчас."
    read -r -p "Продолжить? [y/N] " a
    [ "$a" = "y" ] || exit 1
    uv publish dist/*
fi
