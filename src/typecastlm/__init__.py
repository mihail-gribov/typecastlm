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

Running the model yourself is a separate matter and needs no client: the checkpoint is an ordinary
three-label classifier, so `transformers` loads it directly — see the model card.
"""
from .calibrate import calibrate
from .remote import Answer, Client, Ternary

__all__ = ["Client", "Answer", "Ternary", "calibrate"]
__version__ = "0.1.0"
