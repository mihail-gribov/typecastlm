"""Read this checkpoint in three decision shapes. Requires `transformers` only.

    ask(state, question, true, false)      -> {"p": {"yes","no"}, "p_unsure", "logits"}
    choice(state, question, options)       -> {"p": {option: prob}, "logits", "marks"}
    scale(state, question, levels)         -> same, for an ordinal rubric
    ask_many(state, questions)             -> list of ask() results, material read once

Options are marked with letters, rubric levels with their own digits when those are single
tokens, otherwise with `0..9A..P`. Marks above ten levels degrade. Probabilities use the
per-mode temperature from `prompt.json`; `logits` are already divided by it.

    from reader import Reader
    r = Reader("mihailgribov/typecastlm-qwen3.5-3.8b")
    r.ask(state, "Is the claim supported?", true="the material supports it",
          false="the material contradicts it")
    r.choice(state, "Which rule applies?", [("a", "…"), ("b", "…")])
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

LETTERS = "ABCDEFGHIJKLMNOP"
ORDINAL = "0123456789ABCDEFGHIJKLMNOP"
CHOICE_SYSTEM = "You answer with exactly one character from the given list."


class Reader:
    def __init__(self, model: str, device: str = "cuda", dtype=torch.bfloat16):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(model)
        cfg_file = (path / "prompt.json") if path.is_dir() else Path(
            hf_hub_download(model, "prompt.json"))
        self.cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, dtype=dtype, device_map=device).eval()
        self.model.requires_grad_(False)
        self.trunk = self.model.model
        self.names = [self.model.config.id2label[i]
                      for i in range(self.model.config.num_labels)]
        self.col = {n: i for i, n in enumerate(self.names)}
        # The head emits 29 raw logits: three answers and one per mark. A softmax over all of
        # them is meaningless — each mode normalises its own subset.
        self.cal = self.cfg.get("calibration", {})

    # -- prompt ---------------------------------------------------------------------------------
    def _fold(self, state: str) -> str:
        """Fold a state longer than `max_state_tokens` in the middle."""
        ids = self.tok.encode(state, add_special_tokens=False)
        n = self.cfg["max_state_tokens"]
        if len(ids) <= n:
            return state
        return (self.tok.decode(ids[: n // 2], skip_special_tokens=True) + " […] "
                + self.tok.decode(ids[-(n // 2):], skip_special_tokens=True))

    def _chat(self, system: str, body: str, tail: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": body}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) + tail

    def _text(self, state: str, question: str, true: str, false: str) -> str:
        C = self.cfg
        body = (C["format_two"] + C["body"].format(state=state, question=question)
                + C["means_two"].format(true=true, false=false))
        return self._chat(C["system"], body, C["tail"])

    # -- marks ----------------------------------------------------------------------------------
    def _has_mark(self, mark: str) -> bool:
        """True if the head carries a row for this mark."""
        return f"mark_{mark}" in self.col

    def _marks_for(self, ids: list[str], ordinal: bool) -> list[str]:
        """Marks for the options: a rubric\'s own digits when usable, else letters or `0..9A..P`."""
        if ordinal:
            own = [str(x) for x in ids]
            if len(own) <= 10 and all(x.isdigit() and len(x) == 1 and self._has_mark(x)
                                      for x in own):
                return own
            row = [c for c in ORDINAL if self._has_mark(c)]
        else:
            row = [c for c in LETTERS if self._has_mark(c)]
        if len(row) < len(ids):
            raise ValueError(f"{len(ids)} options but only {len(row)} single-token marks")
        return row[: len(ids)]

    # -- reading --------------------------------------------------------------------------------
    @torch.no_grad()
    def _last(self, text: str) -> torch.Tensor:
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        return self.trunk(**enc).last_hidden_state[0, -1].float()

    def _temp(self, mode: str) -> float:
        return float((self.cal.get(mode) or {}).get("temperature", 1.0))

    def _verdict(self, z: list[float]) -> dict:
        """Three raw logits to an answer. `p` uses the `verdict` temperature, `p_unsure` the
        `three_answers` one; the two are fitted separately and are not interchangeable."""
        yes, no = (v / self._temp("verdict") for v in z[:2])
        m = max(yes, no)
        ey, en = math.exp(yes - m), math.exp(no - m)
        t3 = self._temp("three_answers")
        w = [v / t3 for v in z]
        m3 = max(w)
        e3 = [math.exp(v - m3) for v in w]
        s3 = sum(e3)
        return dict(logits=dict(zip(self.names[:3], z)),
                    p={"yes": ey / (ey + en), "no": en / (ey + en)},
                    p_unsure=e3[2] / s3)

    @torch.no_grad()
    def ask(self, state: str, question: str, true: str, false: str) -> dict:
        """One closed question. `p` is over yes/no; `p_unsure` is a separate signal to threshold."""
        h = self._last(self._text(self._fold(state), question, true, false))
        z = self.model.score(h.to(self.model.score.weight.dtype)).float()[:3].tolist()
        return self._verdict(z)

    @torch.no_grad()
    def _pick(self, state: str, question: str, options: list[tuple[str, str]], ask: str,
              ordinal: bool) -> dict:
        marks = self._marks_for([o[0] for o in options], ordinal)
        body = (f"<state>\n{self._fold(state)}\n</state>\n\n{question}\n\n"
                + "\n".join(f"{m}. {desc}" for m, (_, desc) in zip(marks, options))
                + f"\n\n{ask}")
        h = self._last(self._chat(CHOICE_SYSTEM, body, "Answer: "))
        full = self.model.score(h.to(self.model.score.weight.dtype)).float()
        z = [float(full[self.col[f"mark_{m}"]]) for m in marks]
        z = [v / self._temp("scale" if ordinal else "choice") for v in z]
        e = [math.exp(v - max(z)) for v in z]
        s = sum(e)
        ids = [o[0] for o in options]
        return dict(p=dict(zip(ids, (v / s for v in e))), logits=dict(zip(ids, z)),
                    marks=dict(zip(ids, marks)))

    def choice(self, state: str, question: str, options: list[tuple[str, str]]) -> dict:
        """Pick one of 2..16 options. `options` are `(name, description)`; names are yours and do
        not reach the prompt."""
        return self._pick(state, question, options, "Answer with one letter.", ordinal=False)

    def scale(self, state: str, question: str, levels: list[tuple[str, str]]) -> dict:
        """Pick a level of an ordinal rubric. `levels` are `(name, description)` in order."""
        marks_are_digits = all(str(k).isdigit() for k, _ in levels)
        ask = ("Answer with the number of the level that rates it." if marks_are_digits
               else "Answer with the mark of the level that rates it.")
        return self._pick(state, question, levels, ask, ordinal=True)

    @staticmethod
    def _snapshot(cache) -> dict:
        """Copy of the cache state after the prefix.

        A hybrid trunk keeps two kinds of state: attention layers hold keys and values per token
        and could be cropped; linear-attention layers hold a recurrent summary that cannot. Both
        are restored from a copy instead.
        """
        return {name: [None if t is None else t.clone() for t in getattr(cache, name)]
                for name in ("conv_states", "recurrent_states", "key_cache", "value_cache")}

    @staticmethod
    def _restore(cache, snap: dict) -> None:
        for name, tensors in snap.items():
            setattr(cache, name, [None if t is None else t.clone() for t in tensors])

    @torch.no_grad()
    def ask_many(self, state: str, questions: list[tuple[str, str, str]]) -> list[dict]:
        """Several `(question, true, false)` about one state. The shared prefix is computed once;
        results equal calling `ask` separately. The split is taken in token space, not in text."""
        state = self._fold(state)
        fulls = [self.tok(self._text(state, q, t, f), add_special_tokens=False)["input_ids"]
                 for q, t, f in questions]
        n_min = min(len(f) for f in fulls)
        n = 0
        while n < n_min and len({f[n] for f in fulls}) == 1:
            n += 1
        # Sharing the prefix across a hybrid trunk is not implemented: restoring the recurrent
        # state of the linear-attention layers from a copy does not reproduce a clean pass.
        return [self.ask(state, q, t, f) for q, t, f in questions]

        cache = self._cache()
        dev = self.model.device
        pre = torch.tensor([fulls[0][:n]], device=dev)
        self.trunk(input_ids=pre, attention_mask=torch.ones_like(pre),
                   past_key_values=cache, use_cache=True,
                   cache_position=torch.arange(n, device=dev))
        snap = self._snapshot(cache)
        out = []
        for full in fulls:
            self._restore(cache, snap)
            ids = torch.tensor([full[n:]], device=dev)
            h = self.trunk(input_ids=ids, past_key_values=cache, use_cache=True,
                           cache_position=torch.arange(n, len(full), device=dev),
                           attention_mask=torch.ones(1, len(full), device=dev, dtype=torch.long))
            z = self.model.score(h.last_hidden_state[0, -1].to(self.model.score.weight.dtype))
            out.append(self._verdict(z.float()[:3].tolist()))
        return out

    def _cache(self):
        """The cache class this trunk needs. A plain `DynamicCache` does not fit a hybrid one."""
        from transformers import DynamicCache

        mod = type(self.model).__module__
        for name in ("Qwen3_5DynamicCache", "Qwen3NextDynamicCache"):
            cls = getattr(__import__(mod, fromlist=[name]), name, None)
            if cls is not None:
                return cls(self.model.config.get_text_config())
        return DynamicCache()
