"""Which profile fits which weights.

A profile holds everything that is not in the weights: where the trunk is cut, which words the
answers are, and how the prompt is assembled. Three sources, in order of authority:

1. a profile passed by the caller — an explicit choice always wins;
2. `reader.json` lying next to the weights — a packed model carries the way it is read;
3. a bundled profile matched by the base model name from `config.json`.

The order matters because the same base model can be packed in several ways, and the file shipped
with the weights knows which one it is.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "profiles"
FIELDS = ("base_model", "cut_layer", "answers", "system", "head_two", "head_many", "body", "tail",
          "max_state_tokens")


# Роли ответов зовутся по-разному в разных источниках: упаковщик пишет `true/false`, профиль —
# `yes/no`. Приводим к одному виду, иначе `noul` не найдёт своих слов.
ROLES = {"true": "yes", "false": "no", "yes": "yes", "no": "no", "unknown": "unknown"}


def norm_answers(d: dict) -> dict:
    return {ROLES.get(k.lower(), k.lower()): str(v).strip() for k, v in d.items()}


def available() -> dict[str, dict]:
    """Bundled profiles by name."""
    out = {}
    for p in sorted(HERE.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["answers"] = norm_answers(d.get("answers", {}))
        d["_dir"] = str(HERE)
        out[p.stem] = d
    return out


def _from_dir(model: str | Path) -> dict | None:
    """The profile a packed checkpoint carries beside its weights, if there is one."""
    p = Path(model) / "reader.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if "answers" in d:
        d["answers"] = norm_answers(d["answers"])
    # The head file travels with the weights, so paths resolve against the checkpoint directory.
    d["_dir"] = str(Path(model))
    return d


def _base_name(model: str | Path) -> str:
    cfg = Path(model) / "config.json"
    if cfg.exists():
        d = json.loads(cfg.read_text(encoding="utf-8"))
        return str(d.get("base_model") or d.get("_name_or_path") or "")
    return str(model)


def resolve(model: str | Path, profile: str | dict | None = None) -> dict:
    """The profile to read these weights with, or a refusal naming what is known."""
    if isinstance(profile, dict):
        return profile
    known = available()
    if isinstance(profile, str):
        if profile in known:
            return known[profile]
        p = Path(profile)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        raise KeyError(f"unknown profile {profile!r}; bundled: {sorted(known)}")
    own = _from_dir(model)
    if own:
        # У весов может не быть головы в `reader.json` — тогда берём её из встроенного профиля с
        # той же базовой моделью. Сверка на несовпадение всё равно сработает при загрузке.
        # Старый или неполный файл дополняется встроенным профилем той же базовой модели: без
        # этого не хватало бы кусков промпта, и ошибка вылезла бы только на первом вопросе.
        base = next((p for p in known.values()
                     if p.get("base_model") == own.get("base_model")), None)
        if base:
            for k, v in base.items():
                if k not in own and k != "_dir":
                    own[k] = v
            if "head" not in own or not isinstance(own.get("head"), dict):
                own["head"], own["_dir"] = base["head"], base["_dir"]
        return own
    name = _base_name(model).lower()
    for prof in known.values():
        if prof["base_model"].lower() in name or name.endswith(prof["name"]):
            return prof
    raise LookupError(
        f"no profile for {model!r}. A profile says where to cut the trunk and which words the "
        f"answers are, and it has to be measured for each base model — bundled: {sorted(known)}. "
        f"Pass profile=... to choose one, or a path to your own.")
