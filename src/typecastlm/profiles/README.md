# Profiles

One file per base model. A profile says everything that is not in the weights: where to cut the
trunk, which words the answers are, and how the prompt is put together.

Answer words are stored as **surfaces**, not token ids. Ids differ between tokenizers, and a
profile that carried them would silently read the wrong rows on another model; the surface is
resolved through the tokenizer at load time, and two answers whose first token coincides are
refused outright.

Fields:

| field | meaning |
|---|---|
| `base_model` | the checkpoint this profile was written for |
| `cut_layer` | last block kept; the trunk is truncated in memory when a full model is loaded |
| `answers` | role → the word the model would write (`yes`, `no`, `unknown`, or any names) |
| `system`, `head_two`, `head_many`, `body`, `tail` | the prompt, piece by piece |
| `max_state_tokens` | longer states are folded in the middle |
| `enable_thinking` | passed to the chat template where it exists |
| `measured` | what was actually measured with this profile, and on what |

A model repository may ship its own `reader.json` next to the weights. That file wins: weights and
the way they are read belong together, and a bundled profile is only the fallback for a bare base
model.

Adding a base model means writing a profile and **measuring it**. The `measured` block is not
decoration: without it nobody knows whether the answer words of this tokenizer carry the verdict at
the layer the profile cuts at.
