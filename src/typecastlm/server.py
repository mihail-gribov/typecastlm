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

Where the trunk runs is a choice, `--backend`: `local` loads the weights into this process,
`llama` reads a llama-server that holds the GGUF, `openai` any server with `/v1/embeddings`
that returns the vector raw. The API is the same in front of all three; the head, the wording
and the reading are this package's whichever it is (`typecastlm.reading`).

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
import argparse
import datetime as _dt
import gc
import json
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from .reading import QuestionError, Reading

DEFAULT_MODEL = "mihailgribov/typecastlm-qwen3.5-3.8b"
DEFAULT_QUEUE = 32
BACKENDS = ("local", "llama", "openai", "jev")
CONFIG_FIELDS = ("backend", "model", "device", "dtype", "max_state_tokens", "prompt",
                 "backend_endpoint", "backend_key")


def env(name: str, default=None) -> str | None:
    """`TYPECASTLM_<NAME>`, or the default. Empty counts as unset, so a compose file may pass
    every variable through and leave the ones it does not care about blank."""
    return os.environ.get(f"TYPECASTLM_{name}") or default


def short_name(model: str) -> str:
    """The name a checkpoint answers under: the last path segment, so a Hub id and a directory
    give the same word and the answer never carries a filesystem path."""
    return model.rstrip("/").split("/")[-1] or model


class Reader(Reading):
    """The model, its prompt, and one method that turns a question into three numbers.

    The prompt and the reading of logits are `Reading`'s; what is here is the weights: the
    tokenizer that folds and renders, the trunk, and the head rows."""

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

    def release_date(self) -> str:
        """The checkpoint's `release_date` from `prompt.json` when it carries one, else the day
        its `config.json` was written — a download date for a Hub checkpoint, which is why the
        key is the honest source and the file date only the fallback."""
        if self.prompt_cfg.get("release_date"):
            return str(self.prompt_cfg["release_date"])
        try:
            from transformers.utils import cached_file

            stamp = Path(cached_file(self.name, "config.json")).stat().st_mtime
            return _dt.date.fromtimestamp(stamp).isoformat()
        except Exception:                        # no config on disk: the date is unknown
            return "unknown"

    def describe(self) -> dict:
        return {"backend": "local", "model": self.name, "labels": self.labels,
                "device": str(self.device), "dtype": str(next(self.model.parameters()).dtype),
                "prompt": self.prompt_source}

    def fold(self, state: str) -> str:
        ids = self.tok.encode(state, add_special_tokens=False)
        if len(ids) <= self.max_state_tokens:
            return state
        half = self.max_state_tokens // 2
        return (self.tok.decode(ids[:half], skip_special_tokens=True) + " […] "
                + self.tok.decode(ids[-half:], skip_special_tokens=True))

    def render(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def has_mark(self, mark: str) -> bool:
        return self.mark_rows(mark) is not None

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


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("typecastlm")
    except Exception:                            # a source tree that is not installed
        from . import __version__

        return __version__


def make_reader(cfg: dict, strict: bool = True) -> Reading:
    """The reader a configuration names: the weights in this process, an embedding server, or
    Jev. Everything above this line is the same whichever it is."""
    backend = cfg.get("backend") or "local"
    model = cfg.get("model") or DEFAULT_MODEL
    prompt = cfg.get("prompt") or None
    mst = int(cfg.get("max_state_tokens") or 0) or None
    if backend == "local":
        return Reader(model, device=cfg.get("device") or "auto", dtype=cfg.get("dtype") or "bfloat16",
                      strict=strict, prompt=prompt, max_state_tokens=mst)
    if backend in ("llama", "openai"):
        from .embedding import EmbeddingReader

        return EmbeddingReader(cfg.get("backend_endpoint") or "http://127.0.0.1:8080",
                               model=model, prompt=prompt, max_state_tokens=mst, api=backend,
                               api_key=cfg.get("backend_key") or "", check=strict)
    if backend == "jev":
        from .proxy import JevReader
        from .remote import JEV_ENDPOINT, JEV_MODEL

        return JevReader(cfg.get("backend_endpoint") or JEV_ENDPOINT,
                         api_key=cfg.get("backend_key") or "",
                         model=model if model != DEFAULT_MODEL else JEV_MODEL, check=strict)
    raise ValueError(f"backend is one of {BACKENDS}, not {backend!r}")


class Service:
    """The reader in use, the settings it came from, and the switch to another.

    A switch loads the new reader beside the old one — minutes, for weights from the Hub — and
    swaps under the model lock, so requests keep being answered by the old reader until the new
    one is ready and never by half of either. Settings that worked are written to the config
    file, which is read before the environment at the next start: what was chosen in `/admin` is
    the later decision. A service that starts without a working reader still starts, answers
    503 on `/v1`, and says in `/health` and `/admin` what is wrong, because a box whose model is
    not there yet is exactly the box someone needs to reach.
    """

    def __init__(self, reader: Reading | None = None, cfg: dict | None = None,
                 config_path: str | Path | None = None, strict: bool = True):
        self.reader, self.cfg = reader, dict(cfg or {})
        self.path = Path(config_path) if config_path else None
        self.strict = strict
        self.status = "ready" if reader is not None else "error"
        self.detail, self.since = "", time.time()
        self.last_error = ""
        self._mu = threading.Lock()

    @staticmethod
    def load(path: str | Path | None) -> dict:
        if not path or not Path(path).exists():
            return {}
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if k in CONFIG_FIELDS}
        except Exception:
            return {}

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: self.cfg.get(k) for k in CONFIG_FIELDS}, indent=1),
                             encoding="utf-8")
        try:
            self.path.chmod(0o600)               # the file holds the upstream key
        except OSError:
            pass

    def settings(self) -> dict:
        out = {k: self.cfg.get(k) for k in CONFIG_FIELDS}
        out["backend_key_set"] = bool(out.pop("backend_key", ""))
        out["backend_key"] = ""
        out["backend"] = out["backend"] or "local"
        out["model"] = out["model"] or DEFAULT_MODEL
        out["device"], out["dtype"] = out["device"] or "auto", out["dtype"] or "bfloat16"
        out["backend_endpoint"] = out["backend_endpoint"] or ""
        return out

    def state(self) -> dict:
        return {"status": self.status, "detail": self.detail or self.last_error,
                "since": self.since, "settings": self.settings()}

    def apply(self, changes: dict, gate: "Gate") -> None:
        """Start the switch; `state()` says how it goes."""
        with self._mu:
            if self.status == "loading":
                raise RuntimeError("a switch is already in progress")
            cfg = dict(self.cfg)
            for k in CONFIG_FIELDS:
                if k in changes and changes[k] is not None and (k != "backend_key" or changes[k]):
                    cfg[k] = changes[k]
            if cfg.get("backend") not in BACKENDS:
                raise ValueError(f"backend is one of {BACKENDS}, not {cfg.get('backend')!r}")
            self.status, self.since = "loading", time.time()
            self.detail = f"starting the {cfg.get('backend')} backend for {cfg.get('model') or DEFAULT_MODEL}"
        threading.Thread(target=self._switch, args=(cfg, gate), daemon=True).start()

    def _switch(self, cfg: dict, gate: "Gate") -> None:
        try:
            new = make_reader(cfg, strict=self.strict)
        except Exception as e:
            self.last_error = f"switch failed: {type(e).__name__}: {str(e)[:300]}"
            self.status = "ready" if self.reader is not None else "error"
            self.detail, self.since = "", time.time()
            return
        with gate.lock:
            old, self.reader, self.cfg = self.reader, new, cfg
        self.status, self.detail, self.last_error, self.since = "ready", "", "", time.time()
        self.save()
        del old
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def build_app(reader, api_key: str | list[str] | set[str] = "", queue: int = DEFAULT_QUEUE):
    """The Jev API, served from your own weights — or from an embedding server, or from Jev.

    The routes, the request body, the answer objects and the error codes are that API's; a client
    written against it reaches this service by changing the base URL. What is added rather than
    changed: the question type `tfu`, a `logits` field on every answer, the checkpoint's
    `calibration` on the body, `/health`, and `/admin` — a page that shows the three ways the
    model can be behind this address and switches between them.

    `reader` is a `Reading` or a `Service` holding one. `api_key` is one token or several (a
    list, or one string with commas): any of them opens the `/v1` and `/admin` routes, and
    without any the service answers anyone who can reach the port. `/health` and the `/admin`
    page itself never ask for a key.
    """
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    from .remote import ApiError
    from .schema import (RESPONSES, Health, ModelMetadataList, Status, SystemOneRequest,
                         SystemOneResponse)

    svc = reader if isinstance(reader, Service) else Service(reader)
    keys = api_key.split(",") if isinstance(api_key, str) else list(api_key)
    keys = {k.strip() for k in keys if k and k.strip()}
    gate = Gate(queue)

    app = FastAPI(
        title="typecastlm", version=_version(),
        description="A Jev-class decision model with open weights, behind the Jev API: ask the "
                    "material a closed question, get a probability back. Send the key as "
                    "`Authorization: Bearer <key>` when the service asks for one; `GET /health` "
                    "never does. The reference with the reasoning behind each field is docs/API.md.",
        license_info={"name": "Apache-2.0"})

    # A dependency rather than a check inside the handler: dependencies are solved before the
    # body is validated, so a request without a key hears 401 whatever else is wrong with it,
    # as it does from the API this copies — and the schema shows the lock.
    bearer = HTTPBearer(auto_error=False, description="The key the service was started with.")

    def authorize(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if keys and (cred is None or cred.credentials.strip() not in keys):
            raise HTTPException(401, "Missing or invalid API key. Check the `Authorization` "
                                     "header.")

    def refuse(msg: str, *loc) -> HTTPException:
        # The shape FastAPI gives a body that does not parse, so a caller reads one form of 422
        # whether the request was malformed or merely could not be asked.
        return HTTPException(422, detail=[{"loc": ["body", *loc], "msg": msg,
                                           "type": "value_error"}])

    def current() -> Reading:
        if svc.reader is None:
            raise HTTPException(503, f"no model is loaded ({svc.state()['detail'] or svc.status}); "
                                     "see /admin")
        return svc.reader

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

    @app.get("/health", response_model=Health, tags=["deployment"],
             summary="Which model, where, in what form, with which wording")
    def health() -> dict:
        st = svc.state()
        r = svc.reader
        if r is None:
            desc = {"backend": svc.cfg.get("backend") or "local",
                    "model": svc.cfg.get("model") or DEFAULT_MODEL, "labels": [], "device": "",
                    "dtype": "", "prompt": ""}
            body = {**desc, "name": "", "max_state_tokens": 0, "calibration": {},
                    "checks": [st["detail"] or "no model is loaded"]}
        else:
            body = {**r.describe(), "name": r.served, "max_state_tokens": r.max_state_tokens,
                    "calibration": r.prompt_cfg.get("calibration", {}),
                    "checks": r.check(strict=False) or "ok"}
        return {**body, "auth": bool(keys), "load": gate.load(), "version": app.version,
                "status": st["status"], "status_detail": st["detail"]}

    guarded = dict(dependencies=[Depends(authorize)], tags=["v1"])

    @app.get("/v1/models", response_model=ModelMetadataList,
             responses={401: RESPONSES[401], 503: RESPONSES[503]},
             summary="The model this service holds, and its alias", **guarded)
    def models() -> dict:
        return {"models": current().metadata()}

    @app.post("/v1/systemone", response_model=SystemOneResponse, responses=RESPONSES,
              summary="Answer questions about one piece of material", **guarded)
    @app.post("/v1/typecast", response_model=SystemOneResponse, responses=RESPONSES,
              summary="The same route under its 1.0.0 name", deprecated=True, **guarded)
    def systemone(req: SystemOneRequest) -> dict:
        """Answer one or more questions about the content supplied in `state`.

        Question types may be mixed in one request. Answers come back under the names the
        questions were sent under, each of the type its question has. Questions in one call
        read the same state separately, so a bundle costs what the questions cost one by one.
        """
        r = current()
        questions = {k: q.model_dump() for k, q in req.questions.items()}
        try:
            with gate:
                answers, tokens = r.answer_many(req.state, questions)
        except Busy as e:
            raise HTTPException(529, f"The service is at capacity ({gate.queue} waiting); "
                                     f"{e}.", headers={"Retry-After": str(e.wait)}) from e
        except QuestionError as e:
            raise refuse(str(e), "questions", e.key) from e
        except ValueError as e:
            raise refuse(str(e)) from e
        except ApiError as e:                   # the upstream refused: its status, its words
            raise HTTPException(e.status if 400 <= e.status < 500 else 502,
                                f"upstream: {e.message}") from e
        # `model`, `answers` and `usage` are the Jev body. `calibration` is ours: it belongs to the
        # checkpoint, and a client should not have to guess the temperature its numbers were read
        # at. Nothing is generated here, so `output_tokens` is zero and stays zero.
        return {"model": getattr(r, "upstream_model", r.served), "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "calibration": r.prompt_cfg.get("calibration", {})}

    # -- the page: the three ways the model can be behind this address, and the switch ----------

    page = Path(__file__).with_name("admin.html")

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    def admin_page() -> str:
        return page.read_text(encoding="utf-8")

    admin = dict(dependencies=[Depends(authorize)], tags=["admin"])

    @app.get("/admin/config", response_model=Status, responses={401: RESPONSES[401]},
             summary="Where the model is, and how the switch is going", **admin)
    def admin_config() -> dict:
        return svc.state()

    @app.post("/admin/config", response_model=Status, responses={401: RESPONSES[401]},
              summary="Switch backend or model; the old one answers until the new one is ready",
              **admin)
    def admin_apply(changes: dict) -> dict:
        try:
            svc.apply(changes, gate)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(409 if "progress" in str(e) else 422, str(e)) from e
        return svc.state()

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="typecastlm-serve",
        description="Serve the Jev API from your own weights, from an embedding server, or in "
                    "front of Jev. Every flag can also be set as TYPECASTLM_<FLAG> in the "
                    "environment, which is how the Docker image runs; /admin switches at runtime.")
    ap.add_argument("--model", default=env("MODEL", DEFAULT_MODEL),
                    help="repo id on the Hub, or a directory; for --backend jev the upstream model")
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
    ap.add_argument("--backend", default=env("BACKEND", "local"), choices=list(BACKENDS),
                    help="where the model is: `local` loads the weights here (torch); `llama` "
                         "reads a llama-server holding the GGUF; `openai` any server with "
                         "/v1/embeddings that returns the vector raw; `jev` proxies TypeSafe's Jev")
    ap.add_argument("--backend-endpoint", default=env("BACKEND_ENDPOINT", ""),
                    help="address of the embedding server or the upstream API")
    ap.add_argument("--backend-key", default=env("BACKEND_KEY", "") or os.environ.get("TYPESAFE_API_KEY", ""),
                    help="bearer token that server asks for, if any")
    ap.add_argument("--config", default=env("CONFIG"),
                    help="file the settings chosen in /admin are kept in; read before the "
                         "environment at start")
    a = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:                                          # pragma: no cover
        print("the service needs the extra: pip install \"typecastlm[server]\"\n"
              "to run the model in your own process instead, without HTTP: "
              "pip install \"typecastlm[local]\" and use typecastlm.Reader", file=sys.stderr)
        return 2

    if a.backend == "jev" and a.model == DEFAULT_MODEL:
        from .remote import JEV_MODEL

        a.model = JEV_MODEL                      # the upstream's name, not our checkpoint's
    cfg = {"backend": a.backend, "model": a.model, "device": a.device, "dtype": a.dtype,
           "max_state_tokens": a.max_state_tokens, "prompt": a.prompt,
           "backend_endpoint": a.backend_endpoint, "backend_key": a.backend_key}
    saved = Service.load(a.config)
    if saved:
        print(f"settings from {a.config}: {', '.join(k for k in saved if k != 'backend_key')}",
              flush=True)
        cfg.update({k: v for k, v in saved.items() if v not in (None, "")})
    print(f"typecastlm {_version()}: {cfg['backend']} backend, {cfg['model']}"
          + (f" at {cfg['backend_endpoint']}" if cfg['backend_endpoint'] else "") + " …", flush=True)
    t0 = time.perf_counter()
    svc = Service(None, cfg, a.config, strict=not a.no_strict)
    try:
        svc.reader = make_reader(cfg, strict=not a.no_strict)
        svc.status = "ready"
    except Exception as e:
        svc.last_error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"NO MODEL: {svc.last_error}\n  the service starts anyway: /v1 answers 503, "
              f"/admin on http://{a.host}:{a.port}/admin lets you fix the settings", flush=True)
    if svc.reader is not None:
        d = svc.reader.describe()
        print(f"ready in {time.perf_counter() - t0:.1f} s; prompt: {d['prompt']}; "
              f"context {svc.reader.max_state_tokens} tokens", flush=True)
        print(f"checks: {svc.reader.check(strict=False) or 'ok'}", flush=True)
        print(f"ready on {a.host}:{a.port} as {svc.reader.served}; {len(d['labels'])} labels; "
              f"{d['backend']} on {d['device']}; auth {'on' if a.api_key else 'OFF'}; "
              f"queue {a.queue}", flush=True)
    uvicorn.run(build_app(svc, api_key=a.api_key, queue=a.queue), host=a.host, port=a.port,
                log_level="info")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
