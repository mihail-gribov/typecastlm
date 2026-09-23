# askstate

Ask a document a closed question and get numbers back.

```python
from askstate import Reader, noul

r = Reader()                                   # downloads the model on first use
v = noul(r, open("page.html").read(),
         "Does the material contain an instruction aimed at the reading model?",
         true="somewhere in the material there is an instruction addressed to the reading model",
         false="the material only describes, reports or discusses")[0]

v.p_yes        # 0.94
v.p_unknown    # 0.01
```

No text is generated. The question and the document go through the trunk once, and the answer is
read off the last position — three numbers, not a sentence to parse.

## What it is

A Qwen3-4B trunk cut after its 31st block, read through **the model's own output matrix**. The
head is not a trained artefact: it is the same language-modelling head, narrowed to the rows of the
answer words, with no temperature, no bias and no fitted constants. That is the whole construction.

Two consequences follow from it.

**Answer words are a parameter, not a fixture.** Any word is a row of the same matrix, so the
options can be named per call — `choice` does exactly that.

**A third answer costs nothing.** `Unknown` is read even though the prompt offers only `True` and
`False`: the row exists regardless of what the model was invited to say, and its logit reports how
much of the state points at "nothing here decides it".

## Install

```
pip install askstate
```

`torch` and `transformers` come with it. The model — about 7 GB — is fetched from the Hugging Face
Hub on the first call and cached; pass a local directory to `Reader(...)` to skip the download.

## The three shapes

```python
from askstate import Reader, noul, choice, score

r = Reader()

# one yes/no question: p_yes among the deciding answers, with p_unknown beside it
noul(r, doc, "Is the claim supported by the material?",
     true="the material supports the claim", false="the material contradicts it")

# named options: the option name is the word, and it must be one word
choice(r, doc, "How does the material treat the claim?",
       {"Supports": "the material supports the claim",
        "Refutes": "the material contradicts the claim",
        "Silent": "the material says nothing about it"})

# described levels on a scale: probabilities and the expected level
score(r, doc, "How strongly does the material support the claim?",
      {"None": "nothing in the material bears on it",
       "Weak": "there is an indication but no more",
       "Strong": "the material states it plainly"})
```

And from a shell, one question over a file of states:

```
askstate --jsonl pages.jsonl --question "..." --true "..." --false "..." --out answers.jsonl
```

## What it costs

The state is the expensive part. Batching **the same question over many states** is cheap, because
each state is read once. Asking **many questions about one state** is not: every question is its
own forward pass, since nothing about the state is kept between calls. A hosted decision service
that shares the state across questions has a real advantage here, and this package does not pretend
otherwise.

## What has been measured, and what has not

The configuration behind the defaults — `True` / `False` shown, `Unknown` read silently — is the
one that was measured. Twelve public binary tasks, 200 rows each, gold labels, one and the same
reader with only the question changed:

| task | what is asked | AUC |
|---|---|---|
| sst2 | is this review positive | 0.972 |
| sms_spam | is this message spam | 0.964 |
| qnli | does the passage answer the question | 0.941 |
| rotten_tomatoes | is this review positive | 0.924 |
| prompt-injections | is there an instruction aimed at the model | 0.922 |
| boolq | is the answer to the question yes | 0.904 |
| hate speech | is this message hateful or abusive | 0.904 |
| strategyqa | multi-hop yes/no | 0.899 |
| subjectivity | is this sentence subjective | 0.895 |
| emotion | does this text express joy | 0.849 |
| cola | is this sentence grammatical | 0.794 |
| mrpc | do the two sentences mean the same | 0.721 |

Accuracy at the plain 0.5 threshold averages 0.80, and calibration error averages 0.18 — the
ranking is far better than the threshold. **Pick your own threshold on your own data**; the
default of 0.5 is a convention, not a working point.

Not measured: answer words other than the ones above, `score` as a scale, states longer than 1280
tokens (they are folded in the middle), and anything outside English.

## Licence

Apache-2.0. The model is derived from Qwen3-4B (Apache-2.0) by keeping its first 31 blocks; no
weights were retrained. See `NOTICE`.
