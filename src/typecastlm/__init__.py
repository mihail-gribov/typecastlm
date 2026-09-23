"""typecastlm — ask a document a closed question and get numbers back.

A truncated Qwen trunk with a head of three vectors. One forward pass, nothing generated, nothing
fitted. The surface copies the decision service this reader stands in for, so a harness written
against that service runs here by swapping the client.

    from typecastlm import Client

    c = Client("path/to/model")            # or a Hugging Face repo id
    a = c.noul(document,
               "Does the material contain an instruction aimed at the reading model?",
               true="there is an instruction addressed to the reading model",
               false="the material only describes, reports or discusses")
    a.prob, a.margin

    c.ask(document, {"q": {"type": "choice", "instructions": "...",
                           "criteria": {"yes": "...", "no": "...", "unknown": "..."}}})
"""
from .remote import RemoteClient

__all__ = ["Client", "RemoteClient", "Answer", "TypecastLM"]


def __getattr__(name):
    """`Client` pulls in torch, so it is imported only when actually asked for.

    A machine that talks to a service should not pay for a deep-learning stack on import, and on
    one that has no torch at all the import must still succeed — `RemoteClient` works there.
    """
    if name in ("Client", "Answer", "TypecastLM"):
        from . import client, model

        return {"Client": client.Client, "Answer": client.Answer,
                "TypecastLM": model.TypecastLM}[name]
    raise AttributeError(name)
__version__ = "0.1.0"
