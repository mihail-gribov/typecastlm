"""The wrapper: a prompt, a forward pass, three probabilities.

The model carries everything of its own. It is an ordinary three-label classifier — the trunk cut
after its reading layer, and a `score` matrix of three rows over the last position — so this file
holds no weights, no head and no arithmetic beyond a softmax. What it does hold is the prompt,
because a classifier that answers ANY closed question has to be told which one.

    p = softmax( score · norm(h) )

The prompt travels with the weights (`prompt.json`), so a checkpoint and the wording it was
measured with cannot drift apart. The bundled copy is only a fallback for a checkpoint that
predates the file.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import defaults

DEFAULT_MODEL = "mihailgribov/typecastlm-qwen3-4b"


class TypecastLM:
    """Loads the classifier once and answers closed questions about states."""

    def __init__(self, model: str | Path = DEFAULT_MODEL, device: str = "auto",
                 dtype: str = "bfloat16", prompt: dict | None = None,
                 max_state_tokens: int | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name = str(model)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "right"      # голова берёт правейший непадовый токен
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map=device).eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.prompt_cfg = prompt or self._prompt_cfg(model)
        self.max_state_tokens = max_state_tokens or self.prompt_cfg["max_state_tokens"]
        self.last_tokens = 0

    @staticmethod
    def _prompt_cfg(model: str | Path) -> dict:
        p = Path(model) / "prompt.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        try:                                  # у скачанной модели файл лежит в кеше хаба
            from huggingface_hub import hf_hub_download

            return json.loads(Path(hf_hub_download(str(model), "prompt.json"))
                              .read_text(encoding="utf-8"))
        except Exception:
            return defaults()

    # --- prompt -------------------------------------------------------------------------------
    def _fold(self, state: str) -> str:
        ids = self.tok.encode(state, add_special_tokens=False)
        if len(ids) <= self.max_state_tokens:
            return state
        half = self.max_state_tokens // 2
        return (self.tok.decode(ids[:half], skip_special_tokens=True) + " […] "
                + self.tok.decode(ids[-half:], skip_special_tokens=True))

    def prompt(self, state: str, question: str, meanings: tuple[str, str]) -> str:
        C = self.prompt_cfg
        body = (C["format_two"] + C["body"].format(state=self._fold(state), question=question)
                + C["means_two"].format(true=meanings[0], false=meanings[1]))
        msgs = [{"role": "system", "content": C["system"]}, {"role": "user", "content": body}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + C["tail"]

    # --- reading ------------------------------------------------------------------------------
    def ask(self, states: str | list[str], question: str, meanings: tuple[str, str],
            labels: list[str] | None = None, batch_size: int = 4) -> list[dict[str, float]]:
        """Probabilities over the model's labels, one dictionary per state.

        `labels` renames them on the way out; the order is what carries the meaning.
        """
        torch = self.torch
        if isinstance(states, str):
            states = [states]
        names = list(labels) if labels else self.labels
        if len(names) != len(self.labels):
            raise ValueError(f"the model has {len(self.labels)} labels, got {len(names)}")
        prompts = [self.prompt(s, question, meanings) for s in states]
        out: list[dict[str, float]] = []
        self.last_tokens = 0
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                enc = self.tok(prompts[i: i + batch_size], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.device)
                self.last_tokens += int(enc["attention_mask"].sum())
                p = torch.softmax(self.model(**enc).logits.float(), -1).cpu()
                out += [dict(zip(names, (float(v) for v in row))) for row in p]
        return out
