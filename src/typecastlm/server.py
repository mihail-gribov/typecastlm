"""The service: loads the model once, answers questions about documents.

This is the half that carries the weights. It is installed on purpose — `pip install
typecastlm[server]` — because the client must stay light, and the two rarely live on the same
machine.

    typecastlm-serve --model mihailgribov/typecastlm-qwen3.5-3.8b --port 8000

The model name is the only thing it needs: the weights come from the Hub on first start and are
cached, and the prompt comes with them (`prompt.json` beside the weights). That is deliberate —
the wording and the weights were measured together, and keeping the wording here instead would let
the two drift apart.

The body and the answer are the Jev API's, field for field, so a client written against that
interface reaches this service by changing the base URL. Added rather than changed: the question
type `tfu`, `logits` on every answer, and the checkpoint's `calibration` on the body.

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
import json
import math
import os
import sys
from pathlib import Path

DEFAULT_MODEL = "mihailgribov/typecastlm-qwen3.5-3.8b"


class Reader:
    """The model, its prompt, and one method that turns a question into three numbers."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 dtype: str = "bfloat16", max_state_tokens: int | None = None,
                 strict: bool = True, prompt: str | Path | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.name = torch, model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "right"          # the head reads the rightmost non-pad token
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
    CHOICE_SYSTEM = "You answer with exactly one character from the given list."
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
        return text + "Answer: "

    def answer_many(self, state: str, questions: dict) -> tuple[dict, int]:
        """Answer several questions about one state.

        Sharing the prefix across a hybrid trunk is not implemented yet, so the questions are
        answered one by one and a bundle costs what asking them separately costs.
        """
        out, tokens = {}, 0
        for key, q in questions.items():
            ans, t = self.answer(state, q)
            out[key], tokens = ans, tokens + t
        return out, tokens

    def kind_of(self, q: dict) -> str:
        kind = self.ALIASES.get(q.get("type", "noul"), q.get("type", "noul"))
        if kind not in self.KINDS:
            raise ValueError(f"question type {q.get('type')!r} is not implemented; this service "
                             f"answers {self.KINDS}")
        return kind

    def prepare(self, state: str, q: dict) -> tuple[str, list[str], list[str] | None, dict]:
        """Prompt text, answer names, the marks that carry them if any, and the legend.

        `criteria` is a map of names to descriptions, and for `score` it may also be an ordered
        list, which is the form the Jev API documents; a list is read as levels 0, 1, 2 and so on.
        """
        kind = self.kind_of(q)
        crit = q.get("criteria") or self.prompt_cfg.get("criteria_default", {})
        if kind in ("noul", "tfu"):
            if len(crit) != 2:
                raise ValueError("a yes/no question takes exactly two criteria; the third answer "
                                 "is read without being asked for")
            t, f = list(crit.values())
            return self.build(state, q["instructions"], t, f), list(self.EXPECTED), None, {}
        if isinstance(crit, (list, tuple)):
            crit = {str(i): d for i, d in enumerate(crit)}
        if len(crit) < 2:
            raise ValueError(f"a {kind} question takes at least two options")
        options = list(crit.items())
        marks = self.marks_for([k for k, _ in options], ordinal=(kind == "score"))
        text = self.build_marks(state, q["instructions"], options, marks,
                                ordinal=(kind == "score"))
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

    def read(self, kind: str, logits: dict, legend: dict) -> dict:
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
                    "probabilities": p, "confidence": p[lead]}
        if kind == "choice":
            return {"type": "choice", "choice": lead, "probabilities": p, "confidence": p[lead]}
        return {"type": "tfu", "tfu": lead, "probabilities": p, "confidence": p[lead]}

    def answer(self, state: str, q: dict) -> tuple[dict, int]:
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
        out = self.read(self.kind_of(q), logits, legend)
        out["logits"] = logits
        return out, int(enc["attention_mask"].sum())


def _request_model():
    """The request body is declared at module level, not inside the factory.

    The module has `from __future__ import annotations`, so annotations are strings and FastAPI
    resolves them in the MODULE namespace. A class defined inside a function is invisible there,
    and the request body silently becomes a query parameter: the server answers 422 to a
    perfectly good POST.
    """
    from pydantic import BaseModel

    class Ask(BaseModel):
        state: str
        questions: dict
        model: str | None = None

    return Ask


def build_app(reader: Reader, api_key: str = ""):
    """The Jev API, served from your own weights.

    The routes, the request body, the answer objects and the error codes are that API's; a client
    written against it reaches this service by changing the base URL. What is added rather than
    changed: the question type `tfu`, a `logits` field on every answer, and the checkpoint's
    `calibration` on the body.
    """
    from fastapi import FastAPI, Header, HTTPException

    Ask = _request_model()
    globals()["Ask"] = Ask                     # so the string annotation resolves
    app = FastAPI(title="typecastlm", version="1.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"model": reader.name, "labels": reader.labels, "device": str(reader.device),
                "prompt": reader.prompt_source, "max_state_tokens": reader.max_state_tokens,
                "calibration": reader.prompt_cfg.get("calibration", {}),
                "auth": bool(api_key), "checks": reader.check(strict=False) or "ok"}

    @app.post("/v1/systemone")
    @app.post("/v1/typecast")                  # the name this service answered to in 1.0.0
    def systemone(req: "Ask", authorization: str = Header(default="")) -> dict:
        if api_key and authorization.removeprefix("Bearer ").strip() != api_key:
            raise HTTPException(401, "Missing or invalid API key. Check the `Authorization` "
                                     "header.")
        if not req.questions:
            raise HTTPException(422, "no questions")
        try:
            answers, tokens = reader.answer_many(req.state, req.questions)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        # `model`, `answers` and `usage` are the Jev body. `calibration` is ours: it belongs to the
        # checkpoint, and a client should not have to guess the temperature its numbers were read
        # at. Nothing is generated here, so `output_tokens` is zero and stays zero.
        return {"model": reader.name, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
                "calibration": reader.prompt_cfg.get("calibration", {})}

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="typecastlm-serve")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="repo id on the Hub, or a directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-state-tokens", type=int, default=None,
                    help="how many state tokens to read in full; longer states are folded in the middle")
    ap.add_argument("--prompt", default=None,
                    help="your own template instead of the one shipped with the weights (same fields)")
    ap.add_argument("--no-strict", action="store_true",
                    help="serve a mismatched checkpoint anyway — for debugging only")
    ap.add_argument("--api-key", default=os.environ.get("TYPECASTLM_API_KEY", ""),
                    help="require this bearer token; without it the service answers anyone "
                         "who can reach the port")
    a = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:                                          # pragma: no cover
        print("the service needs the extra: pip install \"typecastlm[server]\"\n"
              "to run the model in your own process instead, without HTTP: "
              "pip install \"typecastlm[local]\" and use typecastlm.Reader", file=sys.stderr)
        return 2

    print(f"loading {a.model} …", flush=True)
    reader = Reader(a.model, device=a.device, dtype=a.dtype, strict=not a.no_strict,
                    prompt=a.prompt, max_state_tokens=a.max_state_tokens)
    print(f"prompt: {reader.prompt_source}; context {reader.max_state_tokens} tokens",
          flush=True)
    print(f"checks: {reader.check(strict=False) or 'ok'}", flush=True)
    print(f"ready on {a.host}:{a.port}; labels {reader.labels}; device {reader.device}", flush=True)
    uvicorn.run(build_app(reader, api_key=a.api_key), host=a.host, port=a.port,
                log_level="info")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
