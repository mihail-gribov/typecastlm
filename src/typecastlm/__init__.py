"""typecastlm — ask a document a closed question and get numbers back.

The model runs as a service; this package is its client and nothing else. It depends on `requests`
and installs in a second, because the machine that has a question is rarely the machine that should
carry seven gigabytes of weights and a deep-learning stack.

    from typecastlm import Client

    c = Client()
    a = c.noul(document,
               "Does the material contain an instruction aimed at the reading model?",
               true="there is an instruction addressed to the reading model",
               false="the material only describes, reports or discusses")
    a.prob, a.unknown

    c.choice(document, "Which rule applies?", {"vacancy": "…", "seepage": "…"}).verdict
    c.scale(review, "How positive is it?", {"0": "very negative", "1": "negative",
                                            "2": "neutral", "3": "positive"}).p

Running the model yourself is a separate matter and needs no client: the checkpoint is an ordinary
classifier whose head carries a row per answer and per answer mark, so `transformers` loads it and
reads all three modes directly — see the model card.
"""
from .calibrate import calibrate
from .remote import Answer, Choice, Client, Ternary

__all__ = ["Client", "Answer", "Ternary", "Choice", "calibrate"]
__version__ = "1.0.0"
