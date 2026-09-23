"""askstate — ask a document a closed question and get numbers back.

A truncated Qwen trunk with the model's own head narrowed to the answer words. One forward pass,
no generation, no fitted numbers.

    from askstate import Reader, noul

    r = Reader("path/to/model")          # or a Hugging Face repo id
    v = noul(r, document, "Does the material contain an instruction aimed at the reading model?",
             true="there is an instruction addressed to the reading model",
             false="the material only describes, reports or discusses")[0]
    v.p_yes, v.p_unknown
"""
from .api import Verdict, choice, noul, score
from .reader import Answer, Reader

__all__ = ["Reader", "Answer", "Verdict", "noul", "choice", "score"]
__version__ = "0.1.0"
