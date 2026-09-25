"""The wire, spelled out: request, questions, answers, errors, as the service documents them.

These models are what `/openapi.json` and `/docs` are rendered from, and what the request body is
validated against. The shapes are the Jev API's, field for field — a question is one of its
types, an answer matches its question — so a client written against that schema reads this one
without surprise. Added rather than changed: the question type `tfu` and its answer, `logits` on
every answer, `calibration` on the response, and the answer to `/health`.

`typecastlm.server` imports this lazily: pydantic is a server dependency, and the client must
stay a package with one.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# Wherever the API takes prose it takes a string, an object or an array; an object or an array is
# read as JSON. `None` is allowed where the API allows the field to be left out.
Text = Union[str, dict, list]
MaybeText = Union[str, dict, list, None]


class NoulCriteria(BaseModel):
    """What counts as a yes and what counts as a no."""
    model_config = ConfigDict(extra="allow")
    true: MaybeText = Field(None, description="What counts as a yes answer.",
                            examples=["The message is unsolicited advertising."])
    false: MaybeText = Field(None, description="What counts as a no answer.",
                             examples=["The message is a legitimate conversation."])


class NoulQuestion(BaseModel):
    """A yes/no question, answered with the probability of yes."""
    type: Literal["noul"] = Field("noul", description="Identifies a yes/no question.")
    instructions: MaybeText = Field(
        None, description="The yes/no question or statement to evaluate.",
        examples=["Is this message spam?"])
    criteria: NoulCriteria | dict[str, MaybeText] | None = Field(
        None, description="What a yes and a no mean — two of them. Left out, the wording shipped "
                          "with the weights is used.")


class TfuQuestion(BaseModel):
    """The same question as `noul`, read over three answers: true, false, unsure."""
    type: Literal["tfu"] = Field(
        "tfu", description="Identifies a question read over true, false and unsure.")
    instructions: MaybeText = Field(None, description="The question to evaluate.")
    criteria: NoulCriteria | dict[str, MaybeText] | None = Field(
        None, description="The same two criteria as `noul`; `unsure` is what is left when "
                          "neither fits and is never written.")


class ChoiceQuestion(BaseModel):
    """One of several named options, with a probability for each."""
    type: Literal["choice"] = Field("choice", description="Identifies a selection among options.")
    instructions: MaybeText = Field(
        None, description="What the model should decide when choosing an option.",
        examples=["What is the tone of this message?"])
    criteria: dict[str, MaybeText] = Field(
        ..., description="Option names and descriptions of when each applies, 2 to 26 of them. "
                         "An option without a description is read by its name alone. Options "
                         "are marked A, B, C… in this order, and the order moves the answer.",
        examples=[{"angry": "An upset or hostile message", "calm": "A neutral message"}])


class ScoreQuestion(BaseModel):
    """A level on an ordinal rubric, with the expected level and a probability per level."""
    type: Literal["score", "scale"] = Field(
        "score", description="Identifies a rating against ordered levels. `scale` is a synonym.")
    instructions: MaybeText = Field(
        None, description="What the model should rate.", examples=["How urgent is this message?"])
    criteria: list[MaybeText] | dict[str, MaybeText] = Field(
        ..., description="Ordered descriptions of the levels; a list gives them positions 0, 1, "
                         "2… and those positions key the answer, a map keeps your keys. At least "
                         "two; ten is the recommended limit and 36 the hard one.",
        examples=[["Can wait", "Needs attention this week", "Needs attention today"]])


Question = Annotated[Union[NoulQuestion, TfuQuestion, ChoiceQuestion, ScoreQuestion],
                     Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    """Questions about one piece of material."""
    state: Text = Field(
        ..., description="The content all questions in this request refer to. An object or an "
                         "array is read as JSON.",
        examples=["I was charged twice. Please help.",
                  {"subject": "Duplicate charge", "message": "Please help."}])
    questions: dict[str, Question] = Field(
        ..., min_length=1,
        description="Questions to ask about the content, each under a name you choose. The "
                    "response uses those names to identify the answers.",
        examples=[{"billing": {"type": "noul", "instructions": "Is this message about billing?"}}])
    model: str | None = Field(
        None, description="Accepted for compatibility and ignored: a process serves one "
                          "checkpoint, and the answer names it. `GET /v1/models` lists the names.",
        examples=["typecastlm-latest"])


class NoulAnswer(BaseModel):
    """The probability of yes."""
    type: Literal["noul"]
    noul: float = Field(..., description="Probability of yes, a softmax over the two answers that "
                                         "decide the question at the `verdict` temperature. The "
                                         "verdict is `noul >= 0.5`.", examples=[0.98])
    logits: dict[str, float] | None = Field(
        None, description="The raw outputs for true, false and unsure, before any temperature. "
                          "Absent when the model is proxied: Jev returns probabilities alone.")


class TfuAnswer(BaseModel):
    """One distribution over true, false and unsure."""
    type: Literal["tfu"]
    tfu: Literal["true", "false", "unsure"] = Field(..., description="The leading answer.")
    probabilities: dict[str, float] = Field(
        ..., description="Probability of each of the three answers at the `three_answers` "
                         "temperature; they sum to 1.")
    confidence: float = Field(..., description="The probability of the leading answer.")
    logits: dict[str, float] | None = None
    native: bool = Field(True, description="False when the server has no `tfu` of its own and "
                                           "the question was asked as a `choice` with a third "
                                           "option — a proxied Jev.")


class ChoiceAnswer(BaseModel):
    """The option chosen, and how likely each was."""
    type: Literal["choice"]
    choice: str = Field(..., description="The option with the highest probability.",
                        examples=["angry"])
    probabilities: dict[str, float] = Field(
        ..., description="Probability of each option, keyed by its name; they sum to 1.")
    confidence: float = Field(..., description="The probability of the chosen option.")
    marks: dict[str, str] | None = Field(
        None, description="Which letter each option was given, in the order the request listed "
                          "them. Absent when proxied.", examples=[{"angry": "A", "calm": "B"}])
    logits: dict[str, float] | None = None


class ScoreAnswer(BaseModel):
    """The expected level, and how likely each level was."""
    type: Literal["score"]
    score: float = Field(..., description="Expected level: the probability-weighted mean of the "
                                          "positions, so it falls between levels.", examples=[1.7])
    legend: dict[str, Text] = Field(..., description="Each position and the level it names.")
    probabilities: dict[str, float] = Field(
        ..., description="Probability of each level, keyed by position; they sum to 1.")
    confidence: float = Field(..., description="The probability of the leading level.")
    marks: dict[str, str] | None = Field(None, description="Which mark each level was given; "
                                                          "absent when proxied.")
    logits: dict[str, float] | None = None


Answer = Annotated[Union[NoulAnswer, TfuAnswer, ChoiceAnswer, ScoreAnswer],
                   Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = Field(..., description="Tokens read, summed over the questions.")
    output_tokens: int = Field(..., description="Always 0: nothing is generated.")


class SystemOneResponse(BaseModel):
    """Answers keyed by question name, with the model used and token usage."""
    model: str = Field(..., description="The checkpoint that answered, under its short name.",
                       examples=["typecastlm-qwen3.5-3.8b"])
    answers: dict[str, Answer] = Field(
        ..., description="One answer per question, under the key you sent; its type matches "
                         "the question's.")
    usage: Usage
    calibration: dict = Field(
        ..., description="The temperature per mode the probabilities were read at — `verdict`, "
                         "`three_answers`, `choice`, `scale`, each `{\"temperature\": t}` — so "
                         "`logits` can be re-read. May carry a `note`.")


class ModelMetadata(BaseModel):
    name: str = Field(..., description="A name the request's `model` may carry.",
                      examples=["typecastlm-qwen3.5-3.8b"])
    description: str
    release_date: str = Field(..., description="YYYY-MM-DD.", examples=["2026-09-24"])


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]


class Load(BaseModel):
    busy: bool = Field(..., description="Whether a request holds the model right now.")
    waiting: int = Field(..., description="Requests in line for it.")
    queue: int = Field(..., description="How many may wait before the next gets 529.")
    served: int
    avg_ms: float = Field(..., description="Exponential mean of a request's time on the model.")


class Health(BaseModel):
    """The deployment: which checkpoint, where, with which wording — what your numbers came from."""
    backend: str = Field(..., description="Where the model is: `local` (the weights in this "
                                          "process), `llama` (a llama-server with the GGUF), "
                                          "`openai` (a server speaking /v1/embeddings) or `jev` "
                                          "(proxied to TypeSafe's Jev).")
    status: str = Field("ready", description="`ready`, `loading` while a backend is being "
                                             "switched or downloaded, `error` when none is "
                                             "loaded; then the `/v1` routes answer 503.")
    status_detail: str = ""
    model: str = Field(..., description="The checkpoint as it was loaded: a Hub id or a path.")
    name: str = Field(..., description="Its short name, the one answers carry.")
    labels: list[str]
    device: str = Field(..., description="The device for a local backend, the server's address "
                                         "for a remote one.")
    dtype: str
    prompt: str = Field(..., description="Where the wording came from: the model directory, the "
                                         "model repository, or `override: <path>`.")
    max_state_tokens: int
    calibration: dict
    auth: bool = Field(..., description="Whether the `/v1` routes require a key.")
    load: Load
    version: str
    checks: str | list[str] = Field(..., description="`ok`, or what does not fit the interface.")


class Error(BaseModel):
    detail: str = Field(..., examples=["Missing or invalid API key. Check the `Authorization` "
                                       "header."])


class Settings(BaseModel):
    """What `/admin` shows and changes: where the model is and how it is read. The keys are
    never echoed back, only whether one is set."""
    backend: Literal["local", "llama", "openai", "jev"] = "local"
    model: str = Field("mihailgribov/typecastlm-qwen3.5-3.8b",
                       description="A Hub repository or a directory: the weights for `local`, "
                                   "`prompt.json` and `head.json` for an embedding backend, the "
                                   "upstream model name for `jev`.")
    device: str = "auto"
    dtype: str = "bfloat16"
    max_state_tokens: int | None = None
    prompt: str | None = Field(None, description="A wording file instead of the shipped one.")
    backend_endpoint: str = Field("", description="The embedding server or the upstream API.")
    backend_key: str = Field("", description="Its token; empty keeps the one already set.")
    backend_key_set: bool = Field(False, description="Read-only: whether a token is stored.")


class Status(BaseModel):
    status: str
    detail: str = ""
    since: float = Field(0.0, description="Unix time the current status began.")
    settings: Settings


# Responses documented on the two `/v1` routes beyond 200 and the 422 FastAPI adds itself.
RESPONSES = {
    401: {"model": Error, "description": "The service requires a key and the `Authorization` "
                                         "header does not carry it."},
    503: {"model": Error, "description": "No model is loaded: a backend is being switched, or "
                                         "none could be started; `/admin` says which."},
    529: {"model": Error, "description": "The model is busy and the line for it is full; "
                                         "`Retry-After` says in how many seconds to come back."},
}
