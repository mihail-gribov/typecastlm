"""The service: loads the model once, answers questions about documents.

This is the half that carries the weights. It is installed on purpose — `pip install
typecastlm[server]` — because the client must stay light, and the two rarely live on the same
machine.

    typecastlm-serve --model mihailgribov/typecastlm-qwen3.5-3.8b --port 8000

The model name is the only thing it needs: the weights come from the Hub on first start and are
cached, and the prompt comes with them (`prompt.json` beside the weights). That is deliberate —
the wording and the weights were measured together, and keeping the wording here instead would let
the two drift apart.

Every flag has an environment variable of the same name under `TYPECASTLM_` (`TYPECASTLM_MODEL`,
`TYPECASTLM_PORT`, `TYPECASTLM_API_KEY`, …), which is how the Docker image is configured: the
container runs `typecastlm-serve` with no arguments and reads its environment.

The routes, the request body, the answer objects and the error codes are the Jev API's, field for
field, so a client written against that interface reaches this service by changing the base URL:
`POST /v1/systemone` answers questions, `GET /v1/models` names the checkpoint, a bearer token
guards both. Added rather than changed: the question type `tfu`, `logits` on every answer, the
checkpoint's `calibration` on the body, and `/health` for the deployment itself.

    POST /v1/systemone
    {"state": "...", "questions": {"is_urgent": {"type": "noul", "instructions": "...",
                                                 "criteria": {"true": "...", "false": "..."}}}}
    -> {"model": "...",
        "answers": {"is_urgent": {"type": "noul", "noul": 0.95,
                                  "logits": {"true": 3.1, "false": -0.4, "unsure": -2.2}}},
        "usage": {"input_tokens": 307, "output_tokens": 0}, "calibration": {...}}

The whole contract is in `docs/API.md`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

DEFAULT_MODEL = "mihailgribov/typecastlm-qwen3.5-3.8b"
DEFAULT_QUEUE = 32


def env(name: str, default=None) -> str | None:
    """`TYPECASTLM_<NAME>`, or the default. Empty counts as unset, so a compose file may pass
    every variable through and leave the ones it does not care about blank."""
    return os.environ.get(f"TYPECASTLM_{name}") or default


def short_name(model: str) -> str:
    """The name a checkpoint answers under: the last path segment, so a Hub id and a directory
    give the same word and the answer never carries a filesystem path."""
    return model.rstrip("/").split("/")[-1] or model


class Reader:
    """The model, its prompt, and one method that turns a question into three numbers."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 dtype: str = "bfloat16", max_state_tokens: int | None = None,
                 strict: bool = True, prompt: str | Path | None = None):
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not str(device).startswith("cuda"):
            self._without_fla()

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.name = torch, model
        self.served = short_name(model)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "right"          # the head reads the rightmost non-pad token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map=device).eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.col = {name: i for i, name in enumerate(self.labels)}
        self.marks = {n[len("mark_"):]: i for n, i in self.col.items()
                      if n.startswith("mark_")}
        self._rows: dict = {}                    # mark -> a row per surface form
        self.prompt_cfg, self.prompt_source = self._prompt(model, prompt)
        self.max_state_tokens = max_state_tokens or self.prompt_cfg["max_state_tokens"]
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and self.max_state_tokens > limit:
            raise ValueError(f"--max-state-tokens {self.max_state_tokens} is past what the model "
                             f"can attend to ({limit} positions)")
        self.check(strict=strict)

    @staticmethod
    def _without_fla() -> None:
        """Off the GPU, the linear-attention kernels must stay out of sight.

        `flash-linear-attention` is what makes the hybrid trunk fast on CUDA, and transformers
        picks it up whenever the package can be imported. Its kernels are Triton, and on a CPU
        tensor they fail with `0 active drivers` instead of falling back. Hiding the package
        before transformers looks for it sends the trunk down its torch path, which is slow and
        correct — the only kind of CPU service there is.
        """
        if sys.modules.get("fla") is not None and "fla" in sys.modules:
            return                               # already imported: nothing to hide any more
        sys.modules["fla"] = None                # find_spec() and import both answer "absent"

    EXPECTED = ("true", "false", "unsure")
    PROMPT_FIELDS = ("system", "format_two", "means_two", "body", "tail", "max_state_tokens")

    def check(self, strict: bool = True) -> list[str]:
        """Is this checkpoint the thing the clients are written against?

        A model with two outputs loses `unsure` without a word, and one whose labels sit in
        another order swaps yes and no — both keep answering, plausibly, and wrongly. So the
        shape is checked once at startup and the process refuses to serve a mismatch.
        """
        bad = []
        if not hasattr(self.model, "score"):
            bad.append(f"not a sequence classifier: {type(self.model).__name__} has no `score`")
        n = len(self.labels)
        if n < 3:
            bad.append(f"the clients read at least three outputs, this model has {n}: "
                       f"{self.labels}")
        elif tuple(n.lower() for n in self.labels[:3]) != self.EXPECTED:
            bad.append(f"the first three labels are {self.labels[:3]}, expected "
                       f"{list(self.EXPECTED)} in that order — the order is what carries the "
                       "meaning")
        if not self.marks:
            # The modes read marks from the embedding, so a head without `mark_*` rows still
            # answers a choice here. It is not the documented checkpoint, though: through the
            # plain `text-classification` pipeline such a model can only answer yes or no.
            bad.append("no `mark_*` outputs: this is not the checkpoint the clients document")
        missing = [f for f in self.PROMPT_FIELDS if f not in self.prompt_cfg]
        if missing:
            bad.append(f"prompt.json is missing {missing}")
        # A partial calibration is worse than none: the client reads a missing mode at
        # temperature 1.0 without noticing.
        cal = self.prompt_cfg.get("calibration") or {}
        modes = {"verdict", "three_answers", "choice", "scale"}
        have = modes & set(cal)
        if cal and have != modes:
            bad.append(f"calibration covers {sorted(have)}; the client expects {sorted(modes)}, "
                       "and reads a missing mode at temperature 1.0")
        w = getattr(getattr(self.model, "score", None), "weight", None)
        if w is not None and w.shape[0] != n:
            bad.append(f"head has {w.shape[0]} rows against {n} labels")
        if bad and strict:
            raise ValueError("this checkpoint does not fit the interface:\n  - "
                             + "\n  - ".join(bad))
        return bad

    @staticmethod
    def _prompt(model: str, override: str | Path | None) -> tuple[dict, str]:
        """Which wording to use, and where it came from.

        Two sources, and which one is in play has to stay visible. The measured wording ships
        beside the weights: the numbers in the model card are true of THAT wording and of no
        other. An override replaces it on purpose — a different question style, another language
        — and from then on the card's numbers describe something else, which is why the source is
        reported in `/health` rather than quietly assumed.

        There is no third. A checkpoint without the file is an error, not an occasion to reach for
        a copy lying around: substituting a wording is exactly the kind of change that breaks
        nothing and invalidates everything measured. `model/prompt.json` is there to be copied
        and passed on purpose, never picked up on its own.
        """
        if override:
            return json.loads(Path(override).read_text(encoding="utf-8")), f"override: {override}"
        p = Path(model) / "prompt.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), "model directory"
        try:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(model, "prompt.json"))
            return json.loads(path.read_text(encoding="utf-8")), "model repository"
        except Exception as e:
            raise FileNotFoundError(
                f"{model} ships no prompt.json ({type(e).__name__}), and there is no default to "
                f"fall back on: the wording is part of what was measured. Pass one with "
                f"--prompt (see model/prompt.json in the typecastlm repository)") from e

    def metadata(self) -> list[dict]:
        """What `GET /v1/models` lists: the checkpoint under its own name, and the alias
        `typecastlm-latest`, both in the shape the Jev API gives a model — name, description,
        release date. A process serves one checkpoint, so the list has one model and one alias,
        and a request naming either (or anything else) is answered by it.

        The date is the checkpoint's `release_date` from `prompt.json` when it carries one, else
        the day its `config.json` was written — a download date for a Hub checkpoint, which is
        why the key is the honest source and the file date only the fallback.
        """
        date = self.prompt_cfg.get("release_date")
        if not date:
            try:
                from transformers.utils import cached_file

                stamp = Path(cached_file(self.name, "config.json")).stat().st_mtime
                date = _dt.date.fromtimestamp(stamp).isoformat()
            except Exception:                    # no config on disk: the date is unknown
                date = "unknown"
        base = self.prompt_cfg.get("base", "")
        desc = ("Jev-class decision model with open weights: noul, tfu, choice and score, one "
                "forward pass each" + (f"; derived from {base}" if base else "") + ".")
        return [{"name": self.served, "description": desc, "release_date": date},
                {"name": "typecastlm-latest", "release_date": date,
                 "description": f"Alias of {self.served}, the one checkpoint this service holds."}]

    def build(self, state: str, instructions: str, true: str, false: str) -> str:
        C = self.prompt_cfg
        ids = self.tok.encode(state, add_special_tokens=False)
        if len(ids) > self.max_state_tokens:
            half = self.max_state_tokens // 2
            state = (self.tok.decode(ids[:half], skip_special_tokens=True) + " […] "
                     + self.tok.decode(ids[-half:], skip_special_tokens=True))
        body = (C["format_two"] + C["body"].format(state=state, question=instructions)
                + C["means_two"].format(true=true, false=false))
        msgs = [{"role": "system", "content": C["system"]}, {"role": "user", "content": body}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + C["tail"]

    KINDS = ("noul", "tfu", "choice", "score")
    ALIASES = {"scale": "score"}   # the client calls it `scale`, the wire has always said `score`
    CHOICE_SYSTEM = "You answer with exactly one letter from the given list."
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ORDINAL = "0123456789" + LETTERS

    def mark_rows(self, mark: str):
        """The directions that read this mark, one per surface form, or None if it is not a token.

        A mark row is the model's own output row for that token — that is how the head was packed
        — so a mark the head does not carry is still readable: the row is in the embedding, and
        taking it from there gives the same number to the last bit.

        A mark appears as `" A"` and as `"A"`, usually both single tokens and of different
        strength. The strongest is read, never their mean: averaging mixes an exact measurement
        with its weak copies (0.858 against 0.873 on ARC).
        """
        if mark not in self._rows:
            ids = [t[0] for t in (self.tok(x, add_special_tokens=False)["input_ids"]
                                  for x in (" " + mark, mark)) if len(t) == 1]
            emb = self.model.get_input_embeddings().weight      # tied, and where marks come from
            self._rows[mark] = self.torch.stack([emb[i].float() for i in ids]) if ids else None
        return self._rows[mark]

    def marks_for(self, keys: list[str], ordinal: bool) -> list[str]:
        """Marks for the options: a rubric's own digits when usable, else `0..9A..Z`; letters
        for a choice. A mark the head lacks is taken from the embedding it was copied from."""
        if ordinal:
            own = [str(k) for k in keys]
            if len(own) <= 10 and all(x.isdigit() and len(x) == 1 and self.mark_rows(x) is not None
                                      for x in own):
                return own
            row = [c for c in self.ORDINAL if self.mark_rows(c) is not None]
        else:
            row = [c for c in self.LETTERS if self.mark_rows(c) is not None]
        if len(row) < len(keys):
            raise ValueError(f"{len(keys)} options, but only {len(row)} marks are single tokens "
                             "for this model")
        return row[: len(keys)]

    def build_marks(self, state: str, instructions: str, options: list[tuple[str, str]],
                    marks: list[str], ordinal: bool) -> str:
        ids = self.tok.encode(state, add_special_tokens=False)
        if len(ids) > self.max_state_tokens:
            half = self.max_state_tokens // 2
            state = (self.tok.decode(ids[:half], skip_special_tokens=True) + " […] "
                     + self.tok.decode(ids[-half:], skip_special_tokens=True))
        C = self.prompt_cfg.get("marks") or {}
        if not ordinal:
            ask = C.get("choice_ask", "Answer with one letter.")
        elif marks[0].isdigit():
            ask = C.get("score_ask_digits", "Answer with the number of the level that rates it.")
        else:
            ask = C.get("score_ask_marks", "Answer with the letter of the level that rates it.")
        body = (f"<state>\n{state}\n</state>\n\n{instructions}\n\n"
                + "\n".join(f"{m}. {d}" for m, (_, d) in zip(marks, options))
                + f"\n\n{ask}")
        msgs = [{"role": "system", "content": C.get("system", self.CHOICE_SYSTEM)},
                {"role": "user", "content": body}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + self.prompt_cfg["tail"]

    def answer_many(self, state, questions: dict) -> tuple[dict, int]:
        """Answer several questions about one state.

        Sharing the prefix across a hybrid trunk is not implemented yet, so the questions are
        answered one by one and a bundle costs what asking them separately costs. A question
        that cannot be asked names itself: the error carries the caller's key, so a bundle of
        twenty is refused with the one that is wrong.
        """
        out, tokens = {}, 0
        for key, q in questions.items():
            if not isinstance(q, dict):
                raise QuestionError(key, "a question is an object with `type`, `instructions` "
                                         "and `criteria`")
            try:
                ans, t = self.answer(state, q)
            except ValueError as e:
                raise QuestionError(key, str(e)) from e
            out[key], tokens = ans, tokens + t
        return out, tokens

    def kind_of(self, q: dict) -> str:
        kind = self.ALIASES.get(q.get("type", "noul"), q.get("type", "noul"))
        if kind not in self.KINDS:
            raise ValueError(f"question type {q.get('type')!r} is not implemented; this service "
                             f"answers {self.KINDS}")
        return kind

    @staticmethod
    def as_text(value) -> str:
        """A field as text. The API this follows takes a string, an object, an array or nothing
        wherever prose is expected — the state, the instructions, a criterion — and a third of a
        public benchmark's own items send objects. An object or an array becomes JSON, which is
        what the readers were measured on; nothing becomes an empty string, for the caller to
        decide what that means."""
        if value is None:
            return ""
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def prepare(self, state, q: dict) -> tuple[str, list[str], list[str] | None, dict]:
        """Prompt text, answer names, the marks that carry them if any, and the legend.

        `criteria` is a map of names to descriptions, and for `score` it may also be an ordered
        list, which is the form the Jev API documents; a list is read as levels 0, 1, 2 and so on.
        A description may be missing — the API allows it — and then the name stands in for it:
        a choice without a description is interpreted by its name alone. A yes/no question
        without criteria is asked against the wording in `prompt.json`.
        """
        kind = self.kind_of(q)
        state = self.as_text(state)
        ask = self.as_text(q.get("instructions"))
        crit = q.get("criteria")
        if kind in ("noul", "tfu"):
            default = self.prompt_cfg.get("criteria_default", {})
            crit = crit or default              # absent or empty: the shipped wording
            if not isinstance(crit, dict) or len(crit) != 2:
                raise ValueError("a yes/no question takes exactly two criteria; the third answer "
                                 "is read without being asked for")
            t, f = [self.as_text(v) or default.get(str(k), str(k)) for k, v in crit.items()]
            return self.build(state, ask, t, f), list(self.EXPECTED), None, {}
        if crit is None:
            raise ValueError(f"a {kind} question takes `criteria`: the options it chooses among")
        if isinstance(crit, (list, tuple)):
            crit = {str(i): d for i, d in enumerate(crit)}
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError(f"a {kind} question takes at least two options")
        options = [(str(k), self.as_text(d) or str(k)) for k, d in crit.items()]
        marks = self.marks_for([k for k, _ in options], ordinal=(kind == "score"))
        text = self.build_marks(state, ask, options, marks, ordinal=(kind == "score"))
        return text, [k for k, _ in options], marks, dict(options)

    def _soft(self, logits: dict, mode: str) -> dict:
        v = {k: z / self.temp(mode) for k, z in logits.items()}
        m = max(v.values())
        e = {k: math.exp(x - m) for k, x in v.items()}
        s = sum(e.values())
        return {k: x / s for k, x in e.items()}

    def temp(self, mode: str) -> float:
        return float((self.prompt_cfg.get("calibration", {}).get(mode) or {})
                     .get("temperature", 1.0))

    def read(self, kind: str, logits: dict, legend: dict, marks: list[str] | None = None) -> dict:
        """Logits to the answer body of this question type.

        The shape is the Jev API's, field for field, so a caller written against it needs only
        another base URL. `logits` ride along beside it, because a reading is reproducible from
        them at any temperature while a finished probability is not.
        """
        if kind == "noul":
            two = {k: logits[k] for k in list(logits)[:2]}
            return {"type": "noul", "noul": self._soft(two, "verdict")[list(two)[0]]}
        mode = {"tfu": "three_answers", "choice": "choice", "score": "scale"}[kind]
        p = self._soft(logits, mode)
        lead = max(p, key=p.get)
        if kind == "score":
            score = sum(i * p[k] for i, k in enumerate(p))
            return {"type": "score", "score": score, "legend": legend,
                    "probabilities": p, "confidence": p[lead],
                    "marks": dict(zip(p, marks)) if marks else {}}
        if kind == "choice":
            return {"type": "choice", "choice": lead, "probabilities": p, "confidence": p[lead],
                    "marks": dict(zip(p, marks)) if marks else {}}
        return {"type": "tfu", "tfu": lead, "probabilities": p, "confidence": p[lead]}

    def answer(self, state, q: dict) -> tuple[dict, int]:
        """One question. Shares `prepare` with the bundle path."""
        text, keys, marks, legend = self.prepare(state, q)
        enc = self.tok([text], return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            if marks is None:                    # the three answers always have head rows
                z = self.model(**enc).logits[0, :3].float().cpu()
            else:
                h = self.model.model(**enc).last_hidden_state[0, -1].float()
                z = self.torch.tensor([float((self.mark_rows(m) @ h).max()) for m in marks])
        logits = {k: float(v) for k, v in zip(keys, z)}
        out = self.read(self.kind_of(q), logits, legend, marks)
        out["logits"] = logits
        return out, int(enc["attention_mask"].sum())


class QuestionError(ValueError):
    """A question that cannot be asked, and the key it was sent under."""

    def __init__(self, key: str, msg: str):
        super().__init__(msg)
        self.key = key


class Busy(Exception):
    """The waiting room is full; `wait` is how many seconds to suggest."""

    def __init__(self, wait: int):
        super().__init__(f"try again in {wait} s")
        self.wait = wait


class Gate:
    """One forward pass at a time, and a waiting room of a fixed size.

    The model is one. Two requests on the GPU at once do not finish sooner than one after the
    other — they compete for the same memory and the same cores — so a request holds the lock
    for its forward passes and the others wait in line. The line has an end: past `queue` of
    them, the next request is told to come back (529 with `Retry-After`) rather than queued
    without limit, because a queue with no end is a latency with no bound and the client's
    timeout fires anyway, after having done the work. `Retry-After` is the length of the line
    times the average request, which is what the caller would have waited.
    """

    def __init__(self, queue: int = DEFAULT_QUEUE):
        self.queue = max(0, int(queue))
        self.lock = threading.Lock()             # held for the forward passes
        self._mu = threading.Lock()              # guards the counters below
        self.inflight = 0                        # running plus waiting
        self.served = 0
        self.avg = 0.0                           # seconds per request, an exponential mean

    def __enter__(self):
        with self._mu:
            if self.inflight > self.queue:
                raise Busy(max(1, math.ceil(self.inflight * (self.avg or 1.0))))
            self.inflight += 1
        self.lock.acquire()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self._t0
        self.lock.release()
        with self._mu:
            self.inflight -= 1
            self.served += 1
            self.avg = dt if self.served == 1 else 0.9 * self.avg + 0.1 * dt
        return False

    def load(self) -> dict:
        with self._mu:
            return {"busy": self.lock.locked(), "waiting": max(0, self.inflight - 1),
                    "queue": self.queue, "served": self.served,
                    "avg_ms": round(self.avg * 1000, 1)}


def _request_model():
    """The request body is declared at module level, not inside the factory.

    The module has `from __future__ import annotations`, so annotations are strings and FastAPI
    resolves them in the MODULE namespace. A class defined inside a function is invisible there,
    and the request body silently becomes a query parameter: the server answers 422 to a
    perfectly good POST.
    """
    from pydantic import BaseModel

    class Ask(BaseModel):
        # `state` is a string, an object or an array: the API this follows takes all three, and a
        # third of the benchmark's own items send an object. A non-string is rendered as JSON.
        state: str | dict | list
        questions: dict
        model: str | None = None

    return Ask


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("typecastlm")
    except Exception:                            # a source tree that is not installed
        return "0"


def build_app(reader: Reader, api_key: str | list[str] | set[str] = "",
              queue: int = DEFAULT_QUEUE):
    """The Jev API, served from your own weights.

    The routes, the request body, the answer objects and the error codes are that API's; a client
    written against it reaches this service by changing the base URL. What is added rather than
    changed: the question type `tfu`, a `logits` field on every answer, the checkpoint's
    `calibration` on the body, and `/health`.

    `api_key` is one token or several (a list, or one string with commas): any of them opens the
    two `/v1` routes, and without any the service answers anyone who can reach the port.
    `/health` never asks for a key — it says nothing a caller could not learn from the model card,
    and an orchestrator has to be able to ask it.
    """
    from fastapi import FastAPI, Header, HTTPException, Request

    keys = api_key.split(",") if isinstance(api_key, str) else list(api_key)
    keys = {k.strip() for k in keys if k and k.strip()}
    gate = Gate(queue)

    Ask = _request_model()
    globals()["Ask"] = Ask                     # so the string annotation resolves
    app = FastAPI(title="typecastlm", version=_version(),
                  description="A Jev-class decision model behind the Jev API. Send the key as "
                              "`Authorization: Bearer <key>` when the service asks for one.")

    def authorize(header: str) -> None:
        if keys and header.removeprefix("Bearer ").strip() not in keys:
            raise HTTPException(401, "Missing or invalid API key. Check the `Authorization` "
                                     "header.")

    def refuse(msg: str, *loc) -> HTTPException:
        # The shape FastAPI gives a body that does not parse, so a caller reads one form of 422
        # whether the request was malformed or merely could not be asked.
        return HTTPException(422, detail=[{"loc": ["body", *loc], "msg": msg,
                                           "type": "value_error"}])

    @app.middleware("http")
    async def tag(request: Request, call_next):
        # A request id ties a client's log line to the server's; the caller's is kept, one is
        # minted otherwise. The timing is the whole request as the server saw it, queue included.
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    @app.get("/health")
    def health() -> dict:
        return {"model": reader.name, "name": reader.served, "labels": reader.labels,
                "device": str(reader.device), "dtype": str(next(reader.model.parameters()).dtype),
                "prompt": reader.prompt_source, "max_state_tokens": reader.max_state_tokens,
                "calibration": reader.prompt_cfg.get("calibration", {}),
                "auth": bool(keys), "load": gate.load(), "version": app.version,
                "checks": reader.check(strict=False) or "ok"}

    @app.get("/v1/models")
    def models(authorization: str = Header(default="")) -> dict:
        authorize(authorization)
        return {"models": reader.metadata()}

    @app.post("/v1/systemone")
    @app.post("/v1/typecast")                  # the name this service answered to in 1.0.0
    def systemone(req: "Ask", authorization: str = Header(default="")) -> dict:
        authorize(authorization)
        if not req.questions:
            raise refuse("no questions", "questions")
        try:
            with gate:
                answers, tokens = reader.answer_many(req.state, req.questions)
        except Busy as e:
            raise HTTPException(529, f"The service is at capacity ({gate.queue} waiting); "
                                     f"{e}.", headers={"Retry-After": str(e.wait)}) from e
        except QuestionError as e:
            raise refuse(str(e), "questions", e.key) from e
        except ValueError as e:
            raise refuse(str(e)) from e
        # `model`, `answers` and `usage` are the Jev body. `calibration` is ours: it belongs to the
        # checkpoint, and a client should not have to guess the temperature its numbers were read
        # at. Nothing is generated here, so `output_tokens` is zero and stays zero.
        return {"model": reader.served, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "calibration": reader.prompt_cfg.get("calibration", {})}

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="typecastlm-serve",
        description="Serve the Jev API from your own weights. Every flag can also be set as "
                    "TYPECASTLM_<FLAG> in the environment, which is how the Docker image runs.")
    ap.add_argument("--model", default=env("MODEL", DEFAULT_MODEL),
                    help="repo id on the Hub, or a directory")
    ap.add_argument("--host", default=env("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("PORT", "8000")))
    ap.add_argument("--device", default=env("DEVICE", "auto"))
    ap.add_argument("--dtype", default=env("DTYPE", "bfloat16"))
    ap.add_argument("--max-state-tokens", type=int,
                    default=int(env("MAX_STATE_TOKENS", "0")) or None,
                    help="how many state tokens to read in full; longer states are folded in the middle")
    ap.add_argument("--prompt", default=env("PROMPT"),
                    help="your own template instead of the one shipped with the weights (same fields)")
    ap.add_argument("--no-strict", action="store_true", default=bool(env("NO_STRICT")),
                    help="serve a mismatched checkpoint anyway — for debugging only")
    ap.add_argument("--api-key", default=env("API_KEY", ""),
                    help="require this bearer token (several: separated by commas); without it "
                         "the service answers anyone who can reach the port")
    ap.add_argument("--queue", type=int, default=int(env("QUEUE", str(DEFAULT_QUEUE))),
                    help="how many requests may wait for the model before the next one is told "
                         "to retry (529 with Retry-After)")
    a = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:                                          # pragma: no cover
        print("the service needs the extra: pip install \"typecastlm[server]\"\n"
              "to run the model in your own process instead, without HTTP: "
              "pip install \"typecastlm[local]\" and use typecastlm.Reader", file=sys.stderr)
        return 2

    print(f"typecastlm {_version()}: loading {a.model} …", flush=True)
    t0 = time.perf_counter()
    reader = Reader(a.model, device=a.device, dtype=a.dtype, strict=not a.no_strict,
                    prompt=a.prompt, max_state_tokens=a.max_state_tokens)
    print(f"loaded in {time.perf_counter() - t0:.1f} s; prompt: {reader.prompt_source}; "
          f"context {reader.max_state_tokens} tokens", flush=True)
    print(f"checks: {reader.check(strict=False) or 'ok'}", flush=True)
    print(f"ready on {a.host}:{a.port} as {reader.served}; {len(reader.labels)} labels; "
          f"device {reader.device}; auth {'on' if a.api_key else 'OFF'}; queue {a.queue}",
          flush=True)
    uvicorn.run(build_app(reader, api_key=a.api_key, queue=a.queue), host=a.host, port=a.port,
                log_level="info")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
