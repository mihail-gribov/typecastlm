#!/usr/bin/env bash
# Build the image from this checkout and push it to GHCR under the package's version and `latest`.
#
#   scripts/publish_image.sh            # build, log in, push
#   scripts/publish_image.sh --status   # what is built here and what the registry has, then exit
#
# The login uses the token `gh` holds, which needs the `write:packages` scope:
#   gh auth refresh -h github.com -s write:packages
# A package pushed for the first time is private; its visibility is set to public once, in the
# package settings on GitHub. The image must have passed tests/live_service.py before this runs.
set -eu
cd "$(dirname "$0")/.."
OWNER=${OWNER:-mihail-gribov}
NAME=ghcr.io/$OWNER/typecastlm
VERSION=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)

status() {
    echo "version here: $VERSION"
    docker images "$NAME" --format '  built: {{.Repository}}:{{.Tag}} {{.Size}} ({{.CreatedSince}})' | sort
    for tag in "$VERSION" latest; do
        if docker manifest inspect "$NAME:$tag" >/dev/null 2>&1; then echo "  registry has $NAME:$tag"; else echo "  registry lacks $NAME:$tag (or it is private and not logged in)"; fi
    done
}
if [ "${1:-}" = "--status" ]; then status; exit 0; fi

if ! gh auth status 2>&1 | grep -q "write:packages"; then
    echo "the gh token lacks write:packages; run: gh auth refresh -h github.com -s write:packages" >&2
    exit 1
fi
trap 'echo; echo "stopped; continue: scripts/publish_image.sh (pushed layers are kept by the registry)"; exit 130' INT TERM
docker build -t "$NAME:$VERSION" -t "$NAME:latest" .
gh auth token | docker login ghcr.io -u "$OWNER" --password-stdin
docker push "$NAME:$VERSION"
docker push "$NAME:latest"
status
