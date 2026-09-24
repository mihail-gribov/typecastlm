# Releasing

Two repositories are published, and they hold different things.

| | what it is | where it goes |
|---|---|---|
| this one | the package: client, service, reader, tests, docs | [PyPI](https://pypi.org/project/typecastlm/) and [GitHub](https://github.com/mihail-gribov/typecastlm) |
| the model repository | the weights, and the few files that must sit beside them: the card, `reader.py`, `prompt.json` | [Hugging Face](https://huggingface.co/mihailgribov/typecastlm-qwen3.5-3.8b) |

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

Before either: the version in `pyproject.toml`, an entry in `CHANGELOG.md`, and `pytest`.

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

New weights are a separate matter: they are uploaded with `huggingface_hub.HfApi().upload_file`
from wherever they were built, not from this working copy, which does not carry them.
