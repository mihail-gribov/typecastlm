"""The reader: a truncated Qwen trunk plus rows of its own output matrix.

One forward pass turns a state and a closed question into probabilities. Nothing is generated: the
answer is read off the last position of the prompt, and only the rows that belong to the answer
words are ever multiplied.

The head is not a trained artefact. It is the model's own language-modelling head narrowed to the
tokens of the answer words — no temperature, no bias, no fitted numbers. That is why answer words
can be chosen per call: any word is a row of the same matrix.

Two lists are kept apart on purpose:

  * what the prompt SHOWS — the words the model is told to answer with;
  * what the reader READS — the rows whose logits are turned into probabilities.

They need not coincide, and in the measured configuration they do not: the prompt offers `True` and
`False`, while `Unknown` is read silently. A word the model was never offered still has a row, and
its logit still says how much of the state points that way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "mihailgribov/qwen3-layer30-reader"

SYSTEM = ("You are a decision function inside a program. You judge the state you are given "
          "against one yes/no question and answer with a single word. You never explain.")
HEAD = "Answer format: one word, either {a} or {b}."
HEAD_N = "Answer format: one word, one of: {words}."
BODY = """
<state>
{state}
</state>

Question: {question}
"""
TAIL = "Answer: "


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


class Reader:
    """Loads the trunk once and answers questions about states.

    `model` is a Hugging Face repo id or a local directory holding an ordinary transformers
    checkpoint, so anything that reads Qwen weights reads these.
    """

    def __init__(self, model: str | Path = DEFAULT_MODEL, device: str = "auto",
                 dtype: str = "bfloat16", max_state_tokens: int = 1280):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_state_tokens = max_state_tokens
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model, dtype=getattr(torch, dtype),
                                               device_map=device).eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        cfg = Path(model) / "reader.json"
        self.cfg = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
        self._rows: dict[str, tuple] = {}

    # --- answer words -------------------------------------------------------------------------
    def row(self, word: str) -> tuple:
        """Output-matrix row for an answer word, and the token id it came from.

        The word is encoded as it appears after the tail `"Answer: "`, and its FIRST token is
        taken: the decision is bound to the word well below the reading layer, so one token
        carries it. Two answer words sharing a first token cannot be told apart, and `probs`
        refuses such a set instead of returning numbers that mean nothing.
        """
        if word not in self._rows:
            ids = self.tok.encode(" " + word.strip(), add_special_tokens=False)
            if not ids:
                raise ValueError(f"empty answer word: {word!r}")
            self._rows[word] = (self.model.embed_tokens.weight[ids[0]].float(), ids[0])
        return self._rows[word]

    # --- prompt -------------------------------------------------------------------------------
    def prompt(self, state: str, question: str, shown: dict[str, str]) -> str:
        """`shown` maps the answer word to what choosing it would mean, in order."""
        ids = self.tok.encode(state, add_special_tokens=False)
        if len(ids) > self.max_state_tokens:
            half = self.max_state_tokens // 2
            state = (self.tok.decode(ids[:half], skip_special_tokens=True) + " […] "
                     + self.tok.decode(ids[-half:], skip_special_tokens=True))
        words = list(shown)
        head = (HEAD.format(a=words[0], b=words[1]) if len(words) == 2
                else HEAD_N.format(words=", ".join(words)))
        lines = "\n".join(f"{w} means: {m}" for w, m in shown.items())
        body = head + BODY.format(state=state, question=question) + lines
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + TAIL

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
        if len(set(ids)) != len(ids):
            clash = [w for w, i in zip(names, ids) if ids.count(i) > 1]
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
