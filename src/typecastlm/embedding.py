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

Two things the launcher has to do, and both are checked at start: pool the last token (the GGUF
says so in its metadata, `--pooling last` says so on the command line) and hand the vector over
unnormalised — the native `/embedding` route does when asked with `embd_normalize: -1`; the
OpenAI-style `/v1/embeddings` always normalises, and a normalised vector keeps the winner and
loses the probabilities. Ollama normalises too, which is why it is not a backend here.
"""
import os
import time
from pathlib import Path

from .reading import Reading

DEFAULT_MODEL = "mihailgribov/typecastlm-qwen3.5-3.8b"
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
                 check: bool = True):
        import json

        import requests

        self.endpoint, self.name = endpoint.rstrip("/"), model
        self.served = model.rstrip("/").split("/")[-1] or model
        self.timeout = timeout
        self._s = requests.Session()
        self.prompt_cfg = json.loads(Path(self._file("prompt.json", prompt)).read_text("utf-8"))
        head_cfg = json.loads(Path(self._file("head.json", head)).read_text("utf-8"))
        self.rows: dict[str, list[list[float]]] = head_cfg["rows"]
        self.hidden = int(head_cfg["hidden_size"])
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
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            # A private repository needs the token; `HF_TOKEN` is where the Hub's own tools
            # read it from, so a machine set up for them is set up for this.
            tok = os.environ.get("HF_TOKEN", "")
            r = self._s.get(HUB.format(repo=self.name, file=name), timeout=self.timeout,
                            headers={"Authorization": f"Bearer {tok}"} if tok else {})
            if r.status_code != 200:
                raise FileNotFoundError(f"{name}: not beside the model and not on the Hub for "
                                        f"{self.name} (HTTP {r.status_code}); pass it explicitly")
            cached.write_bytes(r.content)
        return cached

    # -- the launcher -------------------------------------------------------------------------

    def _post(self, route: str, body: dict) -> dict | list:
        r = self._s.post(f"{self.endpoint}{route}", json=body, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"{route}: HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def tokenize(self, text: str) -> list[int]:
        got = self._post("/tokenize", {"content": text, "add_special": False})
        return got["tokens"] if isinstance(got, dict) else got

    def detokenize(self, ids: list[int]) -> str:
        got = self._post("/detokenize", {"tokens": ids})
        return got["content"] if isinstance(got, dict) else got

    def vector(self, text: str) -> list[float]:
        """The trunk's last hidden state for this text, unnormalised."""
        got = self._post("/embedding", {"content": text, "embd_normalize": -1})
        v = _floats(got)
        if v is None or len(v) != self.hidden:
            raise RuntimeError(f"/embedding answered {type(got).__name__} without a vector of "
                               f"{self.hidden} floats; is the server started with --embeddings "
                               "and last-token pooling?")
        return v

    def check(self) -> None:
        """Is this server the trunk, read the right way? A normalised vector has length one and
        a mean-pooled one is short: the trunk's last state after the final norm is long, and
        that is what the head was built against."""
        v = self.vector("check")
        n = sum(x * x for x in v) ** 0.5
        if abs(n - 1.0) < 1e-3:
            raise RuntimeError("the server returns unit vectors: it normalises embeddings, and "
                               "the head cannot be applied to a normalised vector — use "
                               "llama-server's native /embedding route")

    # -- what Reading asks for ------------------------------------------------------------------

    def fold(self, state: str) -> str:
        if len(state) <= self.max_state_tokens:     # a token is at least one character
            return state
        ids = self.tokenize(state)
        if len(ids) <= self.max_state_tokens:
            return state
        half = self.max_state_tokens // 2
        return self.detokenize(ids[:half]) + " […] " + self.detokenize(ids[-half:])

    def render(self, system: str, user: str) -> str:
        return self.frame.format(system=system, user=user)

    def has_mark(self, mark: str) -> bool:
        return f"mark_{mark}" in self.rows

    def logits_of(self, h: list[float], names: list[str]) -> list[float]:
        """One number per name: the largest dot product over the name's rows (a mark's two
        surface forms, read by the stronger, never their mean)."""
        return [max(sum(a * b for a, b in zip(row, h)) for row in self.rows[n]) for n in names]

    def answer(self, state, q: dict) -> tuple[dict, int]:
        text, keys, marks, legend = self.prepare(state, q)
        h = self.vector(text)
        names = list(self.EXPECTED) if marks is None else [f"mark_{m}" for m in marks]
        logits = dict(zip(keys, self.logits_of(h, names)))
        out = self.read(self.kind_of(q), logits, legend, marks)
        out["logits"] = logits
        return out, len(self.tokenize(text))

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
