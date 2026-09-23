"""The reader: a truncated trunk plus rows of the model's own output matrix.

One forward pass turns a state and a closed question into probabilities. Nothing is generated: the
answer is read off the last position of the prompt, and only the rows belonging to the answer words
are ever multiplied.

The head is not a trained artefact. It is the model's own language-modelling head narrowed to the
answer words — no temperature, no bias, no fitted numbers. That is why answer words can be chosen
per call: any word is a row of the same matrix.

Two lists are kept apart on purpose:

  * what the prompt SHOWS — the words the model is told to answer with;
  * what the reader READS — the rows whose logits become probabilities.

In the measured configuration they differ: the prompt offers `True` and `False`, while `Unknown`
is read silently. A word the model was never offered still has a row, and its logit still says how
much of the state points that way.

Everything model-specific — where to cut, which words, how the prompt reads — lives in a profile
(`profiles.py`), so a second base model needs a file, not a fork.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .profiles import resolve

DEFAULT_MODEL = "mihailgribov/qwen3-layer30-reader"


@dataclass(frozen=True)
class Answer:
    """One reading: probabilities over the answer words and the logits behind them."""

    p: dict[str, float]
    logits: dict[str, float]

    @property
    def top(self) -> str:
        return max(self.p, key=self.p.get)

    @property
    def confidence(self) -> float:
        """Probability of the leading word. Our definition, not a borrowed one."""
        return max(self.p.values())


class TypecastLM:
    """Loads the trunk once and answers closed questions about states.

    `model` is a Hugging Face repo id or a local directory. A full base model works too: it is
    truncated in memory at the layer the profile names, so the package is useful before any
    packed checkpoint exists.
    """

    def __init__(self, model: str | Path = DEFAULT_MODEL, profile: str | dict | None = None,
                 device: str = "auto", dtype: str = "bfloat16",
                 max_state_tokens: int | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.profile = resolve(model, profile)
        self.max_state_tokens = max_state_tokens or self.profile.get("max_state_tokens", 1280)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model, dtype=getattr(torch, dtype),
                                               device_map=device).eval()
        self.model.requires_grad_(False)
        self._truncate(int(self.profile["cut_layer"]))
        self.device = next(self.model.parameters()).device
        self.answers: dict[str, str] = dict(self.profile["answers"])
        self._rows: dict[str, tuple] = {}
        self._head = self._load_head()

    # --- head ---------------------------------------------------------------------------------
    def _load_head(self) -> dict[str, "torch.Tensor"]:
        """The three vectors that do the reading, carried by the profile.

        They are rows of the model's own output matrix — thirty kilobytes, so the profile can hold
        them outright. Carrying them beats re-deriving them from the tokenizer every time: the id
        of an answer word depends on how the surface is split, and a silent shift there would read
        the wrong rows while still returning plausible numbers.

        The copy is checked against the weights on load. A mismatch means the profile and the
        checkpoint are not from the same model, and that has to fail loudly rather than quietly.
        """
        spec = self.profile.get("head")
        if not spec or not self.profile.get("_dir"):
            return {}
        from safetensors.torch import load_file

        path = Path(self.profile["_dir"]) / spec["file"]
        if not path.exists():
            return {}
        rows = load_file(str(path))["head"].float()
        out = {}
        for name, vec in zip(spec["rows"], rows):
            word = self.answers.get(name, name)
            own = self.row(word)[0]
            d = float((vec.to(own.device) - own).abs().max())
            if d > 0.05:
                raise ValueError(
                    f"profile head does not match these weights: row {name!r} ({word!r}) differs "
                    f"by {d:.3f}. The profile belongs to another checkpoint.")
            out[word] = vec.to(self.device)
        return out

    # --- trunk --------------------------------------------------------------------------------
    def _truncate(self, cut: int) -> None:
        """Keep blocks 0..cut. Above the reading layer the trunk cannot influence the answer."""
        import torch

        blocks = self.model.layers
        if cut < 0 or len(blocks) <= cut + 1:
            return
        self.model.layers = torch.nn.ModuleList(list(blocks[: cut + 1]))
        self.model.config.num_hidden_layers = cut + 1
        # Config keeps per-layer lists beside the count; leaving them long breaks a later reload.
        for key in ("layer_types",):
            v = getattr(self.model.config, key, None)
            if isinstance(v, list) and len(v) > cut + 1:
                setattr(self.model.config, key, v[: cut + 1])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- answer words -------------------------------------------------------------------------
    def row(self, word: str) -> tuple:
        """Output-matrix row for an answer word, and the token id it came from.

        The word is encoded as it appears after the tail, and its FIRST token is taken: the
        decision is bound to the word well below the reading layer, so one token carries it.
        """
        if getattr(self, "_head", None) and word in self._head:
            return (self._head[word], -1)
        if word not in self._rows:
            ids = self.tok.encode(" " + word.strip(), add_special_tokens=False)
            if not ids:
                raise ValueError(f"empty answer word: {word!r}")
            self._rows[word] = (self.model.embed_tokens.weight[ids[0]].float(), ids[0])
        return self._rows[word]

    # --- prompt -------------------------------------------------------------------------------
    def prompt(self, state: str, question: str, shown: dict[str, str]) -> str:
        """`shown` maps the answer word to what choosing it would mean, in order."""
        P = self.profile
        ids = self.tok.encode(state, add_special_tokens=False)
        if len(ids) > self.max_state_tokens:
            half = self.max_state_tokens // 2
            state = (self.tok.decode(ids[:half], skip_special_tokens=True) + " […] "
                     + self.tok.decode(ids[-half:], skip_special_tokens=True))
        words = list(shown)
        head = (P["head_two"].format(a=words[0], b=words[1]) if len(words) == 2
                else P["head_many"].format(words=", ".join(words)))
        lines = "\n".join(f"{w} means: {m}" for w, m in shown.items())
        body = head + P["body"].format(state=state, question=question) + lines
        msgs = [{"role": "system", "content": P["system"]}, {"role": "user", "content": body}]
        try:
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=P.get("enable_thinking", False))
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + P["tail"]

    # --- reading ------------------------------------------------------------------------------
    def probs(self, states: str | list[str], question: str, shown: dict[str, str],
              read: list[str] | None = None, batch_size: int = 4) -> list[Answer]:
        """Probabilities over answer words.

        `shown` is what the prompt offers, `read` is what is read off the matrix — by default the
        shown words themselves. Words in `read` that are not in `shown` are read silently.
        """
        torch = self.torch
        if isinstance(states, str):
            states = [states]
        names = read or list(shown)
        rows, ids = zip(*(self.row(w) for w in names))
        ids = [i for i in ids if i >= 0]                 # строки из профиля идут без id
        if len(set(ids)) != len(ids):
            clash = [w for w, (_, i) in zip(names, (self.row(w) for w in names))
                     if i >= 0 and ids.count(i) > 1]
            raise ValueError(f"answer words share a first token, so they cannot be separated: "
                             f"{clash}")
        D = torch.stack(rows).to(self.device)
        out: list[Answer] = []
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                chunk = states[i: i + batch_size]
                enc = self.tok([self.prompt(s, question, shown) for s in chunk],
                               return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.device)
                h = self.model(**enc).last_hidden_state[:, -1, :].float()
                lg = h @ D.T
                for p_row, l_row in zip(torch.softmax(lg, -1).cpu(), lg.cpu()):
                    out.append(Answer(p=dict(zip(names, (float(v) for v in p_row))),
                                      logits=dict(zip(names, (float(v) for v in l_row)))))
        return out
