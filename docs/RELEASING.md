# Releasing

Two repositories are published, and they hold different things.

| | what it is | where it goes |
|---|---|---|
| this one | the package: client, service, reader, tests, docs | [PyPI](https://pypi.org/project/typecastlm/) and [GitHub](https://github.com/mihail-gribov/typecastlm) |
| the model repository | the weights, and the few files that must sit beside them: the card, `reader.py`, `prompt.json` | [Hugging Face](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b) |
| the image | the service with its libraries pinned, built from this repository by Actions on a release tag | [GHCR](https://github.com/mihail-gribov/typecastlm/pkgs/container/typecastlm) |
| the GGUF files and `head.json` | the trunk for launchers and the head the client applies (`docs/LAUNCHERS.md`) | the model repository, beside the weights |

Checked out side by side, which is what the scripts assume:

```
typecastlm/
    package/     this repository
    model/       a clone of the model repository, without the weights:
                 GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b model
```

The weights stay on the Hub. In the clone they are LFS pointers of a few bytes, and nothing here
reads them, so a working copy costs a third of a megabyte instead of seven gigabytes.

## The package

```
scripts/publish_pypi.sh test      # TestPyPI, for a rehearsal
scripts/publish_pypi.sh           # PyPI — a version number there is taken forever
```

Before either: the version in `pyproject.toml`, an entry in `CHANGELOG.md`, `pytest`, and
`docs/openapi.json` regenerated with `scripts/export_openapi.py` if a route or a schema moved.

## The image

Nothing is pushed from a workstation: the image is 8 GB and a release tag builds it where the
network is fast.

```
git tag v1.2.0 && git push origin main v1.2.0
```

`.github/workflows/docker.yml` then builds it and pushes `ghcr.io/mihail-gribov/typecastlm:1.2.0`
and `:latest`. The tag is the package version with a `v`, so the image and the wheel with the
same number are the same code. The first publication needs one manual step on GitHub: the
package is private until its visibility is set to public in the package settings. Before
tagging, the same image should have passed `tests/live_service.py` when built locally.

## The model repository

`reader.py` there is a copy of `src/typecastlm/local.py` — the same reader with no package around
it, so the weights are usable without pip. One source, one copy, and a test that fails when they
drift apart.

```
scripts/sync_model.py             # copy it over; --check only reports
scripts/publish_hf.py --dry-run   # what would be pushed
scripts/publish_hf.py -m "…"      # sync, commit, push
```

The token comes from `HF_TOKEN` or from `HF_API_KEY` in the project's `.env`, and is used for the
one push without being written into the clone's config.

`head.json` is written by `scripts/export_head.py` from the checkpoint and committed to the
clone like the other texts; the GGUF files are made by `scripts/convert_gguf.py` and
`llama-quantize` and uploaded like weights, into a `gguf/` folder of the model repository.
Before either ships, `tests/live_launcher.py` against a llama-server on the Q8_0 file.

New weights are a separate matter: they are uploaded with `huggingface_hub.HfApi().upload_file`
from wherever they were built, not from this working copy, which does not carry them.
