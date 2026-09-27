"""The model behind a launcher: llama-server holds the trunk, this reads it.

The checkpoint is a trunk and a head. The trunk is an ordinary Qwen3.5 stack that any launcher
able to serve embeddings can run from a GGUF; the head is 39 directions — the three verdict rows
and the embedding rows of the marks — and applying them is 39 dot products. So the model can be
installed like any other: `llama-server -m typecastlm-qwen3.5-3.8b-q8_0.gguf --embeddings`, and
this reader does the rest with `requests` alone. The prompt is the one from `prompt.json`, the
head comes from `head.json` beside it, and the answers are the service's, field for field.

    from typecastlm import Client, EmbeddingReader

    c = Client(transport=EmbeddingReader("http://127.0.0.1:8080"))
    c.noul(doc, "Is the claim covered?", true="…", false="…").prob

Two things the server has to do, and both are checked at start: pool the last token (the GGUF
says so in its metadata, `--pooling last` says so on the command line) and hand the vector over
unnormalised. Two routes are known. llama-server's native `/embedding` does so when asked with
`embd_normalize: -1`, and is tried first. The OpenAI-style `/v1/embeddings` is what every other
server speaks — vLLM, TEI, and llama-server itself — and is used when the native route is not
there; whether it normalises is the server's setting (vLLM: `--override-pooler-config
'{"pooling_type": "LAST", "normalize": false}'`; llama-server's always does), and a unit vector
is refused at start, because it keeps the winner and loses the probabilities. Ollama normalises
with no setting to turn it off, which is why it is not a backend here.
"""
import os
import time
from pathlib import Path

from .reading import Reading

DEFAULT_MODEL = "mihailgribov/typecastlm-qwen3.5-3.8b"   # weights, GGUF, head and wording together
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
HUB = "https://huggingface.co/{repo}/resolve/main/{file}"
CACHE = Path(os.environ.get("TYPECASTLM_CACHE", Path.home() / ".cache" / "typecastlm"))

# What the tokenizer's chat template renders for a system line and a user turn, with thinking
# off and the generation prompt on — Qwen3.5's, verbatim. Kept here rather than asked of the
# launcher so the prompt is the same string the weights were measured with; `prompt.json` may
# carry its own `chat_frame`, which wins.
CHAT_FRAME = ("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")


class EmbeddingReader(Reading):
    """Reads the model through a llama-server that serves its GGUF as embeddings."""

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                 head: str | Path | None = None, prompt: str | Path | None = None,
                 max_state_tokens: int | None = None, timeout: float = 300.0,
                 check: bool = True, api: str = "auto", api_key: str = "",
                 served_model: str = ""):
        """`api` is the route the server speaks: `llama` (the native `/embedding`), `openai`
        (`/v1/embeddings`), or `auto`, which takes the native one when the server has it.
        `api_key` goes in the `Authorization` header when the server asks for one, and
        `served_model` is the name the OpenAI route wants in `model` — llama-server ignores it,
        vLLM answers only to the name it was started with."""
        import json

        import requests

        if api not in ("auto", "llama", "openai"):
            raise ValueError(f"api is auto, llama or openai, not {api!r}")
        # `repository:tag` names the GGUF the server runs; the files read here — the wording
        # and the head — come from the repository itself, whatever the tag.
        self.variant = ""
        if ":" in model and not model.startswith(("/", ".", "~")) and model.count("/") == 1:
            model, self.variant = model.rsplit(":", 1)
        self.endpoint, self.name = endpoint.rstrip("/"), model
        self.served = model.rstrip("/").split("/")[-1] or model
        self.served_model = served_model or self.served
        self.timeout, self.api = timeout, api
        self._s = requests.Session()
        if api_key:
            self._s.headers["Authorization"] = f"Bearer {api_key}"
        self._can_tokenize: bool | None = None   # learned on first use
        pf = self._file("prompt.json", prompt)
        self.prompt_cfg = json.loads(Path(pf).read_text("utf-8"))
        self.prompt_source = (f"override: {prompt}" if prompt else
                              "model directory" if Path(self.name).is_dir() else "model repository")
        head_cfg = json.loads(Path(self._file("head.json", head)).read_text("utf-8"))
        self.rows: dict[str, list[list[float]]] = head_cfg["rows"]
        self.labels = list(self.rows)
        self.hidden = int(head_cfg["hidden_size"])
        self._matrix()
        self.frame = self.prompt_cfg.get("chat_frame", CHAT_FRAME)
        self.max_state_tokens = max_state_tokens or self.prompt_cfg["max_state_tokens"]
        missing = [f for f in self.PROMPT_FIELDS if f not in self.prompt_cfg]
        if missing:
            raise ValueError(f"prompt.json is missing {missing}")
        if any(n not in self.rows for n in self.EXPECTED):
            raise ValueError(f"head.json lacks the verdict rows {self.EXPECTED}")
        if check:
            self.check()

    def _file(self, name: str, given: str | Path | None) -> Path:
        """`prompt.json` and `head.json`: the path given, the model directory, or the Hub —
        fetched once into the cache with a plain GET, no Hub library needed."""
        if given:
            return Path(given)
        local = Path(self.name) / name
        if local.exists():
            return local
        cached = CACHE / self.name.replace("/", "--") / name
        mark = cached.with_name(name + ".etag")
        cached.parent.mkdir(parents=True, exist_ok=True)
        # A private repository needs the token; `HF_TOKEN` is where the Hub's own tools read it
        # from, so a machine set up for them is set up for this. The copy kept here is checked
        # against the Hub's by its ETag: a wording that changed upstream must not go on being
        # read from a cache, and a machine that is offline keeps what it has.
        tok = os.environ.get("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {tok}"} if tok else {}
        if cached.exists() and mark.exists():
            headers["If-None-Match"] = mark.read_text().strip()
        try:
            r = self._s.get(HUB.format(repo=self.name, file=name), timeout=self.timeout,
                            headers=headers)
        except Exception:
            if cached.exists():
                return cached
            raise
        if r.status_code == 304 and cached.exists():
            return cached
        if r.status_code != 200:
            if cached.exists():
                return cached
            raise FileNotFoundError(f"{name}: not beside the model and not on the Hub for "
                                    f"{self.name} (HTTP {r.status_code}); pass it explicitly")
        cached.write_bytes(r.content)
        if r.headers.get("ETag"):
            mark.write_text(r.headers["ETag"])
        return cached

    # -- the launcher -------------------------------------------------------------------------

    def _post(self, route: str, body: dict, missing_ok: bool = False):
        r = self._s.post(f"{self.endpoint}{route}", json=body, timeout=self.timeout)
        if r.status_code == 404 and missing_ok:
            return None
        if r.status_code != 200:
            raise RuntimeError(f"{route}: HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def tokenize(self, text: str) -> list[int] | None:
        """Token ids from the server's own tokenizer, or None where the route does not exist
        (the OpenAI API has no tokenizer): the fold is then done by characters and the token
        count taken from the answer's `usage`."""
        if self._can_tokenize is False:
            return None
        got = self._post("/tokenize", {"content": text, "add_special": False}, missing_ok=True)
        self._can_tokenize = got is not None
        if got is None:
            return None
        return got["tokens"] if isinstance(got, dict) else got

    def detokenize(self, ids: list[int]) -> str:
        got = self._post("/detokenize", {"tokens": ids})
        return got["content"] if isinstance(got, dict) else got

    def vector(self, text: str) -> list[float]:
        """The trunk's last hidden state for this text, unnormalised."""
        v, self._tokens = self._embed(text)
        if v is None or len(v) != self.hidden:
            raise RuntimeError(f"the server answered without a vector of {self.hidden} floats; "
                               "is it started with --embeddings and last-token pooling?")
        return v

    def _embed(self, text: str) -> tuple[list[float] | None, int | None]:
        """The vector and, when the route reports it, the tokens read. The native route is
        tried first under `auto` and remembered either way."""
        if self.api != "openai":
            got = self._post("/embedding", {"content": text, "embd_normalize": -1},
                             missing_ok=(self.api == "auto"))
            if got is not None:
                self.api = "llama"
                return _floats(got), None
            self.api = "openai"
        got = self._post("/v1/embeddings", {"model": self.served_model, "input": text,
                                            "encoding_format": "float"})
        data = got.get("data") if isinstance(got, dict) else None
        usage = (got.get("usage") or {}) if isinstance(got, dict) else {}
        v = _floats(data[0]) if data else _floats(got)
        return v, (int(usage["prompt_tokens"]) if "prompt_tokens" in usage else None)

    def describe(self) -> dict:
        return {"backend": self.api, "model": self.name, "labels": self.labels,
                "device": self.endpoint, "dtype": self.variant or "as served",
                "prompt": self.prompt_source}

    def check(self, strict: bool = True) -> list[str]:
        """Is this server the trunk, read the right way? A normalised vector has length one and
        a mean-pooled one is short: the trunk's last state after the final norm is long, and
        that is what the head was built against. With `strict` a mismatch raises; without,
        it is returned, for `/health` to show."""
        try:
            v = self.vector("check")
        except Exception as e:
            if strict:
                raise
            return [f"the embedding server does not answer: {e}"]
        n = sum(x * x for x in v) ** 0.5
        bad = []
        if abs(n - 1.0) < 1e-3:
            bad.append(f"the server returns unit vectors over its {self.api} route")
        missing = [f for f in self.PROMPT_FIELDS if f not in self.prompt_cfg]
        if missing:
            bad.append(f"prompt.json is missing {missing}")
        if bad and strict:
            hint = ("llama-server: start it so the native /embedding route is there"
                    if self.api == "openai" else "check the server's pooling and normalisation")
            raise RuntimeError("; ".join(bad) + f": a normalised vector keeps the winner and "
                               f"loses the probabilities — {hint}; vLLM takes "
                               "--override-pooler-config '{\"pooling_type\": \"LAST\", "
                               "\"normalize\": false}'")
        return bad

    # -- what Reading asks for ------------------------------------------------------------------

    CHARS_PER_TOKEN = 3          # a cautious ratio for the fold when no tokenizer route exists

    def fold(self, state: str) -> str:
        if len(state) <= self.max_state_tokens:     # a token is at least one character
            return state
        ids = self.tokenize(state)
        if ids is None:                              # no tokenizer over this API: by characters
            keep = self.max_state_tokens * self.CHARS_PER_TOKEN
            if len(state) <= keep:
                return state
            return state[: keep // 2] + " […] " + state[-(keep // 2):]
        if len(ids) <= self.max_state_tokens:
            return state
        half = self.max_state_tokens // 2
        return self.detokenize(ids[:half]) + " […] " + self.detokenize(ids[-half:])

    def render(self, system: str, user: str) -> str:
        return self.frame.format(system=system, user=user)

    def has_mark(self, mark: str) -> bool:
        return f"mark_{mark}" in self.rows

    def _matrix(self) -> None:
        """The head as one matrix when numpy is there: every row of every name stacked, and for
        each name the slice of rows that are its. Without numpy the rows stay lists and the
        reading is the same arithmetic in Python — 3 ms a question instead of a fraction of one,
        and the client keeps its single dependency."""
        try:
            import numpy as np
        except ImportError:                      # pragma: no cover - the requests-only install
            self._np, self._W = None, None
            return
        self._np = np
        self._W = np.asarray([r for n in self.labels for r in self.rows[n]], dtype=np.float32)
        self._slice, at = {}, 0
        for n in self.labels:
            self._slice[n] = slice(at, at + len(self.rows[n]))
            at += len(self.rows[n])

    def logits_of(self, h: list[float], names: list[str]) -> list[float]:
        """One number per name: the largest dot product over the name's rows (a mark's two
        surface forms, read by the stronger, never their mean)."""
        if self._np is not None:
            z = self._W @ self._np.asarray(h, dtype=self._np.float32)
            return [float(z[self._slice[n]].max()) for n in names]
        return [max(sum(a * b for a, b in zip(row, h)) for row in self.rows[n]) for n in names]

    def answer(self, state, q: dict) -> tuple[dict, int]:
        text, keys, marks, legend = self.prepare(state, q)
        h = self.vector(text)
        names = list(self.EXPECTED) if marks is None else [f"mark_{m}" for m in marks]
        logits = dict(zip(keys, self.logits_of(h, names)))
        out = self.read(self.kind_of(q), logits, legend, marks)
        out["logits"] = logits
        ids = self.tokenize(text) if self._tokens is None else None
        return out, (self._tokens if self._tokens is not None
                     else len(ids) if ids is not None else len(text) // 4)

    def __call__(self, body: dict) -> dict:
        """The service's answer body for a request body — what `Client(transport=…)` calls."""
        answers, tokens = self.answer_many(body["state"], body["questions"])
        return {"model": self.served, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "calibration": self.prompt_cfg.get("calibration", {})}


def _floats(obj):
    """The first list of numbers in whatever shape the route answered: `[{"embedding": [[…]]}]`
    today, and the same vector under other wrappings across versions."""
    if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            got = _floats(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _floats(v)
            if got is not None:
                return got
    return None
