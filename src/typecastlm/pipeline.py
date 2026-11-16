"""The `decision` task of transformers' `pipeline`.

Importing this module registers the task, so the checkpoint is used the way any Hub model is:

    import typecastlm.pipeline                      # pip install "typecastlm[local]"
    from transformers import pipeline

    decide = pipeline("decision", model="inola-ai/decider-open")
    decide("The invoice was paid on 3 May.", "Was the invoice paid?")
    # {"type": "noul", "noul": 0.99, "logits": {...}}

    decide(state, {"type": "choice", "instructions": "Which rule applies?",
                   "criteria": {"vacancy": "…", "seepage": "…"}})
    decide([{"state": s, "question": q} for s, q in rows])

A question is a string (asked yes/no against the criteria in `prompt.json`) or the question
object of the API: `type` (`noul`, `tfu`, `choice`, `score`), `instructions`, `criteria`. The
answer is the API's answer body with the raw `logits` beside it.

The pipeline adds no reading of its own: the prompt, the rows of each mode and the calibration
are `typecastlm.server.Reader`'s, built around the weights the pipeline loaded. `import
typecastlm` registers the task too when transformers is already imported.
"""
from __future__ import annotations

from transformers import AutoModelForSequenceClassification, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY

from .server import Reader

TASK = "decision"


class DecisionPipeline(Pipeline):
    """One forward pass per question; `Reader` writes the prompt and reads the head."""

    def __init__(self, *args, prompt=None, max_state_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader = Reader.around(self.model, self.tokenizer, prompt=prompt,
                                    max_state_tokens=max_state_tokens)

    def _sanitize_parameters(self, **kwargs):
        if kwargs:
            raise TypeError(f"the {TASK} pipeline takes no call parameters, got {sorted(kwargs)}; "
                            "the question type and its criteria go into the question object")
        return {}, {}, {}

    def __call__(self, state, question=None, **kwargs):
        """`decide(state, question)`, `decide({"state": …, "question": …})` or a list of those."""
        if question is not None:
            state = {"state": state, "question": question}
        return super().__call__(state, **kwargs)

    @staticmethod
    def _question(item) -> tuple[object, dict]:
        if not isinstance(item, dict) or "state" not in item or "question" not in item:
            raise ValueError('an input is {"state": …, "question": …}; the question is a string '
                             "or an object with `type`, `instructions` and `criteria`")
        q = item["question"]
        return item["state"], (q if isinstance(q, dict) else {"type": "noul", "instructions": q})

    def preprocess(self, item):
        state, q = self._question(item)
        text, keys, marks, legend = self.reader.prepare(state, q)
        enc = self.tokenizer([text], return_tensors="pt", add_special_tokens=False)
        return {"enc": enc, "q": q, "keys": keys, "marks": marks, "legend": legend}

    def _forward(self, x):
        enc = x.pop("enc")
        x["h"] = self.model.model(**enc).last_hidden_state[0, -1].float()
        x["tokens"] = int(enc["attention_mask"].sum())
        return x

    def postprocess(self, x):
        h = x["h"].to(self.reader.device)
        out = self.reader._answer_from(h, x["q"], x["keys"], x["marks"], x["legend"])
        out["tokens"] = x["tokens"]
        return out


def register() -> None:
    """Add the task to transformers' registry; a second call changes nothing."""
    if TASK not in PIPELINE_REGISTRY.get_supported_tasks():
        PIPELINE_REGISTRY.register_pipeline(TASK, pipeline_class=DecisionPipeline,
                                            pt_model=AutoModelForSequenceClassification,
                                            type="text")


register()
