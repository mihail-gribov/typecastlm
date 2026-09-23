"""Three ways to put a closed question, and what each one gives back.

The shapes follow a decision API rather than a chat one: the caller supplies a state and a question
with its answers spelled out, and gets numbers, never prose.

    noul    one probability of yes, with the weight of "nothing to decide" beside it
    choice  named options, a probability each
    score   described levels on a scale, their probabilities and the expected level

What this is not: every call is its own forward pass. A decision service that shares the state
between several questions pays for it once; here a second question costs a second pass. The state
is what is expensive, so batching the SAME question over many states is cheap, and asking many
questions about one state is not.
"""
from __future__ import annotations

from dataclasses import dataclass

from .reader import Answer, Reader

YES, NO, UNKNOWN = "True", "False", "Unknown"


@dataclass(frozen=True)
class Verdict:
    """The answer to one yes/no question about one state."""

    p_yes: float            # probability of yes among the answers that decide the question
    p_unknown: float        # weight of "the state does not decide it"
    confidence: float       # probability of the leading word, over all three
    raw: Answer

    def __repr__(self) -> str:                                   # pragma: no cover - cosmetic
        return (f"Verdict(p_yes={self.p_yes:.3f}, p_unknown={self.p_unknown:.3f}, "
                f"confidence={self.confidence:.3f})")


def noul(reader: Reader, states: str | list[str], question: str, true: str, false: str,
         batch_size: int = 4) -> list[Verdict]:
    """One yes/no question. `true` and `false` say what each answer would mean.

    `p_yes` is taken among the deciding answers only, so a state that decides nothing does not
    drag it towards a half: that case shows up in `p_unknown` instead, and the two numbers are
    meant to be read together.
    """
    shown = {YES: true, NO: false}
    out = reader.probs(states, question, shown, read=[YES, NO, UNKNOWN], batch_size=batch_size)
    v = []
    for a in out:
        y, n, u = a.p[YES], a.p[NO], a.p[UNKNOWN]
        v.append(Verdict(p_yes=y / (y + n) if y + n > 0 else 0.5, p_unknown=u,
                         confidence=a.confidence, raw=a))
    return v


def choice(reader: Reader, states: str | list[str], question: str, options: dict[str, str],
           batch_size: int = 4) -> list[dict[str, float]]:
    """Named options, a probability each.

    The option name is the word the model would answer with, so it must be a single word and the
    names must differ in their first token. Two or more options are required; three — yes, no and
    something for "nothing to decide" — is the shape this was measured in.
    """
    if len(options) < 2:
        raise ValueError("choice needs at least two options")
    return [a.p for a in reader.probs(states, question, options, batch_size=batch_size)]


def score(reader: Reader, states: str | list[str], question: str, levels: dict[str, str],
          batch_size: int = 4) -> list[dict]:
    """Described levels on a scale: their probabilities and the expected level.

    Levels are given in order, lowest first. The expected value is the usual weighted sum of level
    indices, so it is comparable across calls only when the same levels are used.
    """
    if len(levels) < 2:
        raise ValueError("score needs at least two levels")
    names = list(levels)
    out = []
    for a in reader.probs(states, question, levels, batch_size=batch_size):
        exp = sum(i * a.p[n] for i, n in enumerate(names))
        out.append(dict(expected=exp, levels=a.p, confidence=a.confidence))
    return out
