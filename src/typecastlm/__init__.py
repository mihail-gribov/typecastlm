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
    a.prob, a.margin

    c.tfu(document, "Is the claim supported?", true="…", false="…").p

    c.choice(document, "Which rule applies?", {"vacancy": "…", "seepage": "…"}).verdict
    c.scale(review, "How positive is it?", {"0": "very negative", "1": "negative",
                                            "2": "neutral", "3": "positive"}).p

Running the model in this process instead of calling a service:

    pip install "typecastlm[local]"

    from typecastlm import Reader
    r = Reader("mihailgribov/typecastlm-qwen3.5-3.8b")
    r.noul(document, "Is the claim supported?", true="…", false="…")
"""
from .calibrate import calibrate
from .remote import Answer, Choice, Client, Ternary

__all__ = ["Client", "Answer", "Ternary", "Choice", "Reader", "calibrate"]


def __getattr__(name: str):
    """`Reader` is imported on use: it needs torch, which the client must not require."""
    if name == "Reader":
        try:
            from .local import Reader
        except ImportError as e:                     # pragma: no cover
            raise ImportError("Reader runs the model in this process and needs the extra: "
                              "pip install 'typecastlm[local]'") from e
        return Reader
    raise AttributeError(name)
__version__ = "1.1.1"
