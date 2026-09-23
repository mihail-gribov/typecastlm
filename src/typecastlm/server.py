"""The service: loads the model once, answers questions about documents.

This is the half that carries the weights. It is installed on purpose — `pip install
typecastlm[server]` — because the client must stay light, and the two rarely live on the same
machine.

    typecastlm-serve --model mihailgribov/typecastlm-qwen3-3.5b --port 8000

The model name is the only thing it needs: the weights come from the Hub on first start and are
cached, and the prompt comes with them (`prompt.json` beside the weights). That is deliberate —
the wording and the weights were measured together, and keeping the wording here instead would let
the two drift apart.

The body and the answer are the service's own:

    POST /v1/typecast
    {"state": "...", "questions": {"q": {"type": "noul", "instructions": "...",
                                         "criteria": {"true": "...", "false": "..."}}}}
    -> {"answers": {"q": {"noul": 0.94, "unknown": 0.01}},
        "usage": {"input_tokens": 131}, "model": "..."}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = "mihailgribov/typecastlm-qwen3-3.5b"


class Reader:
    """The model, its prompt, and one method that turns a question into three numbers."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 dtype: str = "bfloat16", max_state_tokens: int | None = None,
                 strict: bool = True, prompt: str | Path | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.name = torch, model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "right"          # голова берёт правейший непадовый токен
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map=device).eval()
        self.model.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
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

        A model with two outputs loses `unknown` without a word, and one whose labels sit in
        another order swaps yes and no — both keep answering, plausibly, and wrongly. So the
        shape is checked once at startup and the process refuses to serve a mismatch.
        """
        bad = []
        if not hasattr(self.model, "score"):
            bad.append(f"not a sequence classifier: {type(self.model).__name__} has no `score`")
        n = len(self.labels)
        if n != 3:
            bad.append(f"the clients read three outputs, this model has {n}: {self.labels}")
        elif tuple(l.lower() for l in self.labels) != self.EXPECTED:
            bad.append(f"labels are {self.labels}, expected {list(self.EXPECTED)} in that order — "
                       "the order is what carries the meaning")
        missing = [f for f in self.PROMPT_FIELDS if f not in self.prompt_cfg]
        if missing:
            bad.append(f"prompt.json is missing {missing}")
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
        nothing and invalidates everything measured. `examples/prompt.json` is there to be copied
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
                f"--prompt (see examples/prompt.json in the typecastlm repository)") from e

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

    KINDS = ("noul", "tfu")

    def answer(self, state: str, q: dict) -> tuple[dict, int]:
        kind = q.get("type", "noul")
        if kind not in self.KINDS:
            raise ValueError(f"question type {kind!r} is not implemented; this service answers "
                             f"{self.KINDS}")
        crit = q.get("criteria") or self.prompt_cfg.get("criteria_default", {})
        if len(crit) != 2:
            raise ValueError("a question takes exactly two criteria; the third answer is read "
                             "without being asked for")
        t, f = list(crit.values())
        prompt = self.build(state, q["instructions"], t, f)
        enc = self.tok([prompt], return_tensors="pt", add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            z = self.model(**enc).logits[0].float()
            p = self.torch.softmax(z, -1).cpu().tolist()
        # Ответ один на оба типа вопроса: сервер даёт три числа, а всякое чтение — `noul` по
        # двум, `tfu` по трём, любая температура — выводится из них на стороне спрашивающего.
        # Считать здесь то же самое дважды значило бы завести второе место, где это можно
        # разойтись.
        logits = dict(zip(self.labels, (float(v) for v in z.cpu())))
        decided = p[0] + p[1]
        return ({"noul": p[0] / decided if decided > 0 else 0.5,
                 "unknown": p[2] if len(p) > 2 else 0.0, "logits": logits},
                int(enc["attention_mask"].sum()))


def _request_model():
    """Тело запроса объявляется на уровне модуля, а не внутри фабрики.

    В модуле стоит `from __future__ import annotations`, поэтому аннотации — строки, и FastAPI
    разрешает их в пространстве имён МОДУЛЯ. Класс, объявленный внутри функции, там не виден, и
    тело запроса молча превращается в параметр строки запроса: сервер отвечает 422 на совершенно
    правильный POST.
    """
    from pydantic import BaseModel

    class Ask(BaseModel):
        state: str
        questions: dict
        model: str | None = None

    return Ask


def build_app(reader: Reader):
    from fastapi import FastAPI, HTTPException

    Ask = _request_model()
    globals()["Ask"] = Ask                     # чтобы аннотация-строка разрешилась
    app = FastAPI(title="typecastlm", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"model": reader.name, "labels": reader.labels, "device": str(reader.device),
                "prompt": reader.prompt_source, "max_state_tokens": reader.max_state_tokens,
                "calibration": reader.prompt_cfg.get("calibration", {}),
                "checks": reader.check(strict=False) or "ok"}

    @app.post("/v1/typecast")
    def typecast(req: "Ask") -> dict:
        if len(req.questions) != 1:
            raise HTTPException(400, "version zero answers one question per call")
        key, q = next(iter(req.questions.items()))
        try:
            ans, tokens = reader.answer(req.state, q)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        # Тело ровно такое, как у сервиса: `answers`, `usage`, `model`. Время меряет клиент —
        # лишнее поле здесь было бы нашим, а не общим.
        # Калибровка едет с ответом: она свойство чекпойнта, и клиент не должен её угадывать.
        return {"answers": {key: ans}, "usage": {"input_tokens": tokens}, "model": reader.name,
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
                    help="сколько токенов состояния читать целиком; длиннее складывается серединой")
    ap.add_argument("--prompt", default=None,
                    help="свой шаблон вместо того, что приехал с весами (JSON тех же полей)")
    ap.add_argument("--no-strict", action="store_true",
                    help="не отказываться от несовпавшей модели — только для отладки")
    a = ap.parse_args(argv)

    import uvicorn

    print(f"loading {a.model} …", flush=True)
    reader = Reader(a.model, device=a.device, dtype=a.dtype, strict=not a.no_strict,
                    prompt=a.prompt, max_state_tokens=a.max_state_tokens)
    print(f"prompt: {reader.prompt_source}; context {reader.max_state_tokens} tokens",
          flush=True)
    print(f"checks: {reader.check(strict=False) or 'ok'}", flush=True)
    print(f"ready on {a.host}:{a.port}; labels {reader.labels}; device {reader.device}", flush=True)
    uvicorn.run(build_app(reader), host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
