"""Requests from many callers, read together in shared forward passes.

A GPU reading one prompt at a time is mostly idle: a forward pass over one short prompt costs
nearly what a pass over a dozen does, because the weights are streamed from memory either way.
So the service lets requests meet. Callers hand their question to a single worker thread and
wait; the worker takes whatever has queued up by the time the GPU is free — nothing is held back
for a window, so a lone request goes straight through and a crowd fills the next pass — sorts the
prompts by length, cuts them into batches under a token budget and reads each batch in one
right-padded pass. The trunk is causal, so padding after a row's last token cannot reach it, and
every row is read at its own last token, as `shared_state` reads its tails.

A request with several questions about one state keeps its own path (`answer_many`, the state
read once): its questions share a long prefix, which a batch of strangers does not. Only
single-question requests are pooled — the common shape, and the one that was paying a whole pass
for one prompt.

The head is read for the whole batch at once: the verdict rows through the checkpoint's own
`score` layer, exactly as one question reads them, and every mark row from one matrix built at
the first marks question. A request's answer is the same as alone up to the rounding of a bf16
pass at another batch shape.
"""
from __future__ import annotations

import queue
import threading

from .reading import QuestionError

# padded tokens per pass (rows x the longest row): the activations of a pass grow with it, and
# past a few thousand rows' worth the GPU is full anyway, so a larger budget only adds memory
BATCH_TOKENS = 32768


class _Job:
    __slots__ = ("state", "questions", "done", "result", "error")

    def __init__(self, state, questions: dict):
        self.state, self.questions = state, questions
        self.done = threading.Event()
        self.result = None
        self.error: BaseException | None = None


class Batcher:
    """One worker on the model; any number of callers. `batcher(state, questions)` blocks until
    the answer is read and returns what `reader.answer_many` would: `(answers, tokens)`."""

    def __init__(self, reader, max_rows: int = 64, batch_tokens: int = BATCH_TOKENS):
        self.reader = reader
        self.max_rows = max(1, int(max_rows))
        self.batch_tokens = max(1, int(batch_tokens))
        self._q: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self.passes = 0                          # forward passes run, and the rows they read:
        self.rows = 0                            # rows / passes is the batch the load produced
        threading.Thread(target=self._loop, name="typecastlm-batcher", daemon=True).start()

    def __call__(self, state, questions: dict):
        job = _Job(state, questions)
        self._q.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def close(self):
        """Stop the worker and let go of the reader: a switch frees the old weights only when
        nothing holds them. Called with every slot of the gate held, so no caller is waiting."""
        self._q.put(None)

    # -- the worker -----------------------------------------------------------------------------

    def _loop(self):
        while True:
            jobs = [self._q.get()]
            while True:                          # everything that queued while the GPU was busy
                try:
                    jobs.append(self._q.get_nowait())
                except queue.Empty:
                    break
            if None in jobs:
                for j in jobs:
                    if j is not None:
                        self._finish(j, error=RuntimeError("the model was switched; ask again"))
                self.reader = None
                return
            try:
                self._run(jobs)
            except BaseException as e:           # never leave a caller waiting
                for j in jobs:
                    if not j.done.is_set():
                        j.error = e
                        j.done.set()

    def _finish(self, job: _Job, result=None, error: BaseException | None = None):
        job.result, job.error = result, error
        job.done.set()

    def _run(self, jobs: list[_Job]):
        r = self.reader
        rows = []                                # (job, key, kind, text, names, marks, legend)
        for job in jobs:
            if len(job.questions) != 1:          # a bundle reads its state once, on its own path
                try:
                    self._finish(job, r.answer_many(job.state, job.questions))
                except BaseException as e:
                    self._finish(job, error=e)
                continue
            (key, q), = job.questions.items()
            if not isinstance(q, dict):
                self._finish(job, error=QuestionError(key, "a question is an object with `type`, "
                                                           "`instructions` and `criteria`"))
                continue
            try:
                kind = r.kind_of(q)
                rows.append((job, key, kind, *r.prepare(job.state, q)))
            except ValueError as e:
                self._finish(job, error=QuestionError(key, str(e)))
        if not rows:
            return
        ids = r.tok([x[3] for x in rows], add_special_tokens=False)["input_ids"]
        order = sorted(range(len(rows)), key=lambda i: len(ids[i]))
        batch: list[int] = []
        for i in order:                          # ascending length: the last row sets the width
            if batch and ((len(batch) + 1) * len(ids[i]) > self.batch_tokens
                          or len(batch) >= self.max_rows):
                self._pass([rows[k] for k in batch], [ids[k] for k in batch])
                batch = []
            batch.append(i)
        self._pass([rows[k] for k in batch], [ids[k] for k in batch])

    def _pass(self, rows: list, ids: list[list[int]]):
        try:
            Z = self.reader.read_rows(ids, [x[5] for x in rows])
        except BaseException as e:
            for x in rows:
                self._finish(x[0], error=e)
            return
        self.passes += 1
        self.rows += len(rows)
        for (job, key, kind, _, names, marks, legend), z, n in zip(rows, Z, map(len, ids)):
            logits = {name: float(v) for name, v in zip(names, z)}
            try:
                out = self.reader.read(kind, logits, legend, marks) | {"logits": logits}
                self._finish(job, ({key: out}, n))
            except BaseException as e:
                self._finish(job, error=e)

    def load(self) -> dict:
        return {"passes": self.passes, "rows": self.rows,
                "rows_per_pass": round(self.rows / self.passes, 2) if self.passes else 0.0}
