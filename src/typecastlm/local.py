"""Read this checkpoint in its four modes, one call each. Requires `transformers` only.

    noul(state, question, true, false)     -> {"p": {"yes","no"}, "logits"}
    tfu(state, question, true, false)      -> {"p": {"true","false","unsure"}, "verdict", "logits"}
    choice(state, question, options)       -> {"p": {option: prob}, "logits", "marks"}
    scale(state, question, levels)         -> same, for an ordinal rubric
    noul_many(state, questions)            -> one noul() result per question

The names are the API's, which are Jev's where Jev has the mode; `ask` and `ask_many` still work
as the older names of `noul` and `noul_many`.

Options are marked `A..Z`, rubric levels with their own digits when those are single tokens,
otherwise with `0..9A..Z`. Reading holds to about six options and slips past twelve; rubrics
longer than ten levels read poorly. Probabilities use the per-mode temperature from
`prompt.json`; `logits` come back raw.

    from reader import Reader
    r = Reader("mihailgribov/typecastlm-qwen3.5-3.8b")
    r.noul(state, "Is the claim supported?", true="the material supports it",
           false="the material contradicts it")
    r.choice(state, "Which rule applies?", [("a", "…"), ("b", "…")])
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ORDINAL = "0123456789" + LETTERS
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
        self._rows: dict[str, torch.Tensor | None] = {}      # mark -> the row that reads it
        # The head emits one logit per answer and one per mark. A softmax over all of them is
        # meaningless — each mode normalises its own subset.
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
    def _mark_row(self, mark: str) -> torch.Tensor | None:
        """The direction that reads this mark, or None if the mark is not one token.

        A mark row is the model's own output row for that token — that is how the head was
        packed — so a mark the head does not carry is not a mark the model cannot read: the row
        is right there in the embedding, and taking it from there gives the same number.
        """
        if mark in self._rows:
            return self._rows[mark]
        col = self.col.get(f"mark_{mark}")
        if col is not None:
            row = self.model.score.weight[col].float()
        else:
            ids = [t[0] for t in (self.tok(x, add_special_tokens=False)["input_ids"]
                                  for x in (" " + mark, mark)) if len(t) == 1]
            emb = self.model.get_input_embeddings().weight      # tied, and where marks come from
            row = emb[ids[0]].float() if ids else None
        self._rows[mark] = row
        return row

    def _has_mark(self, mark: str) -> bool:
        return self._mark_row(mark) is not None

    def _marks_for(self, ids: list[str], ordinal: bool) -> list[str]:
        """Marks for the options: a rubric\'s own digits when usable, else letters or `0..9A..Z`."""
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
        """The two answers that decide the question, read against each other at the `verdict`
        temperature. The third output takes no part in this softmax: it is fitted and calibrated
        as its own mode, and mixing the two would read one of them at the other's temperature."""
        yes, no = (v / self._temp("verdict") for v in z[:2])
        m = max(yes, no)
        ey, en = math.exp(yes - m), math.exp(no - m)
        return dict(p={"yes": ey / (ey + en), "no": en / (ey + en)},
                    logits=dict(zip(self.names[:2], z[:2])))

    def _ternary(self, z: list[float]) -> dict:
        """All three answers in one distribution, at the `three_answers` temperature."""
        w = [v / self._temp("three_answers") for v in z]
        m = max(w)
        e = [math.exp(v - m) for v in w]
        s = sum(e)
        p = dict(zip(self.names[:3], (v / s for v in e)))
        return dict(p=p, verdict=max(p, key=p.get), logits=dict(zip(self.names[:3], z)))

    def _read(self, state: str, question: str, true: str, false: str) -> list[float]:
        h = self._last(self._text(self._fold(state), question, true, false))
        return self.model.score(h.to(self.model.score.weight.dtype)).float()[:3].tolist()

    @torch.no_grad()
    def noul(self, state: str, question: str, true: str, false: str) -> dict:
        """One closed question, answered over the two criteria. For the third answer call `tfu`."""
        return self._verdict(self._read(state, question, true, false))

    @torch.no_grad()
    def tfu(self, state: str, question: str, true: str, false: str) -> dict:
        """The same question over three answers: the two criteria and `unsure`, which is what is
        left when neither fits. No third criterion is written for it."""
        return self._ternary(self._read(state, question, true, false))

    @torch.no_grad()
    def _pick(self, state: str, question: str, options: list[tuple[str, str]], ask: str,
              ordinal: bool) -> dict:
        marks = self._marks_for([o[0] for o in options], ordinal)
        body = (f"<state>\n{self._fold(state)}\n</state>\n\n{question}\n\n"
                + "\n".join(f"{m}. {desc}" for m, (_, desc) in zip(marks, options))
                + f"\n\n{ask}")
        h = self._last(self._chat(CHOICE_SYSTEM, body, "Answer: "))
        R = torch.stack([self._mark_row(m) for m in marks])
        z = (R @ h.float()).tolist()
        w = [v / self._temp("scale" if ordinal else "choice") for v in z]
        e = [math.exp(v - max(w)) for v in w]
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

    @torch.no_grad()
    def noul_many(self, state: str, questions: list[tuple[str, str, str]]) -> list[dict]:
        """Several `(question, true, false)` about one state, one `noul` result each.

        Sharing the prefix across a hybrid trunk is not implemented: restoring the recurrent state
        of the linear-attention layers from a copy does not reproduce a clean pass, so the state
        is read again per question and a bundle costs what the questions cost separately.
        """
        state = self._fold(state)
        return [self.noul(state, q, t, f) for q, t, f in questions]

    # The names these two carried in 1.0.0.
    ask = noul
    ask_many = noul_many
