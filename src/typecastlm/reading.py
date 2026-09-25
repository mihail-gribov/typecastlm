"""How a question becomes a prompt and how the model's numbers become an answer.

This is the part of the service that needs no torch: the wording from `prompt.json`, the marks
that stand for options, the folding of a state too long to read, and the reading of logits at
each mode's temperature. Two readers share it — `typecastlm.server.Reader`, which holds the
weights, and `typecastlm.embedding.EmbeddingReader`, which asks a llama-server for the trunk's
vector and applies the head itself — so that whichever produces the vector, the prompt is the
same string and the answer the same numbers. `tests/same_reading.py` checks that character by
character against the reader shipped beside the weights.

A subclass supplies four things: `fold` (a state cut to `max_state_tokens`, keeping both ends),
`render` (the chat template around a system line and a user turn), `has_mark` (whether a mark
is a single token the head can read) and `answer` (the numbers).
"""
import json
import math


class QuestionError(ValueError):
    """A question that cannot be asked, and the key it was sent under."""

    def __init__(self, key: str, msg: str):
        super().__init__(msg)
        self.key = key


class Reading:
    EXPECTED = ("true", "false", "unsure")
    PROMPT_FIELDS = ("system", "format_two", "means_two", "body", "tail", "max_state_tokens")
    KINDS = ("noul", "tfu", "choice", "score")
    ALIASES = {"scale": "score"}   # the client calls it `scale`, the wire has always said `score`
    CHOICE_SYSTEM = "You answer with exactly one letter from the given list."
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ORDINAL = "0123456789" + LETTERS

    prompt_cfg: dict
    max_state_tokens: int

    # -- what a subclass supplies -------------------------------------------------------------

    def fold(self, state: str) -> str:
        """The state cut to `max_state_tokens`: the middle dropped, both ends kept."""
        raise NotImplementedError

    def render(self, system: str, user: str) -> str:
        """The chat template around a system line and a user turn, generation prompt included."""
        raise NotImplementedError

    def has_mark(self, mark: str) -> bool:
        raise NotImplementedError

    def answer(self, state, q: dict) -> tuple[dict, int]:
        """One question: the answer body and the tokens read."""
        raise NotImplementedError

    def describe(self) -> dict:
        """What `/health` says about this reader: where the model is and in what form."""
        raise NotImplementedError

    def check(self, strict: bool = True) -> list[str]:
        """What does not fit the interface; raises when `strict` and something does not."""
        raise NotImplementedError

    def release_date(self) -> str:
        return str(self.prompt_cfg.get("release_date") or "unknown")

    served: str

    def metadata(self) -> list[dict]:
        """What `GET /v1/models` lists: the checkpoint under its own name, and the alias
        `typecastlm-latest`, both in the shape the Jev API gives a model — name, description,
        release date. A process serves one checkpoint, so the list has one model and one alias,
        and a request naming either (or anything else) is answered by it."""
        date = self.release_date()
        base = self.prompt_cfg.get("base", "")
        desc = ("Jev-class decision model with open weights: noul, tfu, choice and score, one "
                "forward pass each" + (f"; derived from {base}" if base else "") + ".")
        return [{"name": self.served, "description": desc, "release_date": date},
                {"name": "typecastlm-latest", "release_date": date,
                 "description": f"Alias of {self.served}, the one checkpoint this service holds."}]

    # -- the prompt -----------------------------------------------------------------------------

    def build(self, state: str, instructions: str, true: str, false: str) -> str:
        C = self.prompt_cfg
        body = (C["format_two"] + C["body"].format(state=self.fold(state), question=instructions)
                + C["means_two"].format(true=true, false=false))
        return self.render(C["system"], body) + C["tail"]

    def marks_for(self, keys: list[str], ordinal: bool) -> list[str]:
        """Marks for the options: a rubric's own digits when usable, else `0..9A..Z`; letters
        for a choice. A mark the head lacks is taken from the embedding it was copied from."""
        if ordinal:
            own = [str(k) for k in keys]
            if len(own) <= 10 and all(x.isdigit() and len(x) == 1 and self.has_mark(x)
                                      for x in own):
                return own
            row = [c for c in self.ORDINAL if self.has_mark(c)]
        else:
            row = [c for c in self.LETTERS if self.has_mark(c)]
        if len(row) < len(keys):
            raise ValueError(f"{len(keys)} options, but only {len(row)} marks are single tokens "
                             "for this model")
        return row[: len(keys)]

    def build_marks(self, state: str, instructions: str, options: list[tuple[str, str]],
                    marks: list[str], ordinal: bool) -> str:
        C = self.prompt_cfg.get("marks") or {}
        if not ordinal:
            ask = C.get("choice_ask", "Answer with one letter.")
        elif marks[0].isdigit():
            ask = C.get("score_ask_digits", "Answer with the number of the level that rates it.")
        else:
            ask = C.get("score_ask_marks", "Answer with the letter of the level that rates it.")
        body = (f"<state>\n{self.fold(state)}\n</state>\n\n{instructions}\n\n"
                + "\n".join(f"{m}. {d}" for m, (_, d) in zip(marks, options))
                + f"\n\n{ask}")
        return self.render(C.get("system", self.CHOICE_SYSTEM), body) + self.prompt_cfg["tail"]

    # -- the question ---------------------------------------------------------------------------

    def kind_of(self, q: dict) -> str:
        kind = self.ALIASES.get(q.get("type", "noul"), q.get("type", "noul"))
        if kind not in self.KINDS:
            raise ValueError(f"question type {q.get('type')!r} is not implemented; this service "
                             f"answers {self.KINDS}")
        return kind

    @staticmethod
    def as_text(value) -> str:
        """A field as text. The API this follows takes a string, an object, an array or nothing
        wherever prose is expected — the state, the instructions, a criterion — and a third of a
        public benchmark's own items send objects. An object or an array becomes JSON, which is
        what the readers were measured on; nothing becomes an empty string, for the caller to
        decide what that means."""
        if value is None:
            return ""
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def prepare(self, state, q: dict) -> tuple[str, list[str], list[str] | None, dict]:
        """Prompt text, answer names, the marks that carry them if any, and the legend.

        `criteria` is a map of names to descriptions, and for `score` it may also be an ordered
        list, which is the form the Jev API documents; a list is read as levels 0, 1, 2 and so on.
        A description may be missing — the API allows it — and then the name stands in for it:
        a choice without a description is interpreted by its name alone. A yes/no question
        without criteria is asked against the wording in `prompt.json`.
        """
        kind = self.kind_of(q)
        state = self.as_text(state)
        ask = self.as_text(q.get("instructions"))
        crit = q.get("criteria")
        if kind in ("noul", "tfu"):
            default = self.prompt_cfg.get("criteria_default", {})
            crit = crit or default              # absent or empty: the shipped wording
            if not isinstance(crit, dict) or len(crit) != 2:
                raise ValueError("a yes/no question takes exactly two criteria; the third answer "
                                 "is read without being asked for")
            t, f = [self.as_text(v) or default.get(str(k), str(k)) for k, v in crit.items()]
            return self.build(state, ask, t, f), list(self.EXPECTED), None, {}
        if crit is None:
            raise ValueError(f"a {kind} question takes `criteria`: the options it chooses among")
        if isinstance(crit, (list, tuple)):
            crit = {str(i): d for i, d in enumerate(crit)}
        if not isinstance(crit, dict) or len(crit) < 2:
            raise ValueError(f"a {kind} question takes at least two options")
        options = [(str(k), self.as_text(d) or str(k)) for k, d in crit.items()]
        marks = self.marks_for([k for k, _ in options], ordinal=(kind == "score"))
        text = self.build_marks(state, ask, options, marks, ordinal=(kind == "score"))
        return text, [k for k, _ in options], marks, dict(options)

    # -- the answer -----------------------------------------------------------------------------

    def temp(self, mode: str) -> float:
        return float((self.prompt_cfg.get("calibration", {}).get(mode) or {})
                     .get("temperature", 1.0))

    def _soft(self, logits: dict, mode: str) -> dict:
        v = {k: z / self.temp(mode) for k, z in logits.items()}
        m = max(v.values())
        e = {k: math.exp(x - m) for k, x in v.items()}
        s = sum(e.values())
        return {k: x / s for k, x in e.items()}

    def read(self, kind: str, logits: dict, legend: dict, marks: list[str] | None = None) -> dict:
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
                    "probabilities": p, "confidence": p[lead],
                    "marks": dict(zip(p, marks)) if marks else {}}
        if kind == "choice":
            return {"type": "choice", "choice": lead, "probabilities": p, "confidence": p[lead],
                    "marks": dict(zip(p, marks)) if marks else {}}
        return {"type": "tfu", "tfu": lead, "probabilities": p, "confidence": p[lead]}

    def answer_many(self, state, questions: dict) -> tuple[dict, int]:
        """Answer several questions about one state.

        Sharing the prefix across a hybrid trunk is not implemented yet, so the questions are
        answered one by one and a bundle costs what asking them separately costs. A question
        that cannot be asked names itself: the error carries the caller's key, so a bundle of
        twenty is refused with the one that is wrong.
        """
        out, tokens = {}, 0
        for key, q in questions.items():
            if not isinstance(q, dict):
                raise QuestionError(key, "a question is an object with `type`, `instructions` "
                                         "and `criteria`")
            try:
                ans, t = self.answer(state, q)
            except ValueError as e:
                raise QuestionError(key, str(e)) from e
            out[key], tokens = ans, tokens + t
        return out, tokens
