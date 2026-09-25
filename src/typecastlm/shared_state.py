"""Several questions about one state, the state read once.

Every prompt puts the state before the question, so the prompts of a bundle share a long token
prefix. The prefix is run once with a cache, and the tails — each question with its criteria and
the answer cue — are read as one batch continuing from copies of that cache. Tails are
right-padded: the trunk is causal, so padding after a row's last token cannot reach it, and each
row is read at its own last token.

The trunk is hybrid. transformers continues its linear-attention (gated delta) layers from a
cache only one token at a time: a longer chunk restarts the convolution and the recurrence from
zero and reads something else than a clean pass. `install` replaces that layer's forward with one
that continues both from the cached state for any chunk length; everything else — the attention
layers, the head — is transformers' own. Against clean passes in bf16, 140 Quadrat-IPI documents x
3 yes/no questions: mean |dp| 0.005, p95 0.02, max 0.04, AUC within 0.01. transformers' own
token-by-token continuation drifts from a clean pass as much (0.06 at worst on 30 prompts): a
bf16 hybrid trunk does not read a split state bit for bit, whoever splits it.
"""
from __future__ import annotations

import sys

import torch

# rows x prefix tokens per batch: the attention layers' cache is copied once per row, so a long
# state splits the bundle into several batches rather than one that does not fit
ROW_TOKENS = 65536


def common_prefix(ids: list[list[int]]) -> int:
    """Length of the longest shared token prefix, leaving every row at least its last token."""
    n = min(len(x) for x in ids) - 1
    for k in range(n):
        t = ids[0][k]
        if any(x[k] != t for x in ids):
            return k
    return n


def last_hidden(trunk, ids: list[list[int]], pad: int, device) -> tuple[torch.Tensor, int]:
    """The trunk's last hidden state at each row's last token, and the tokens actually read."""
    mod = sys.modules[type(trunk).__module__]
    if not hasattr(mod, "Qwen3_5DynamicCache"):
        raise NotImplementedError(f"{type(trunk).__name__}: no shared-state reading for this "
                                  "architecture")
    install(mod)
    n = common_prefix(ids)
    base = mod.Qwen3_5DynamicCache(trunk.config)
    trunk(input_ids=torch.tensor([ids[0][:n]], device=device), past_key_values=base,
          use_cache=True)
    step = max(1, ROW_TOKENS // max(n, 1))
    out = []
    for i in range(0, len(ids), step):
        tails = [x[n:] for x in ids[i:i + step]]
        width = max(map(len, tails))
        batch = torch.tensor([t + [pad] * (width - len(t)) for t in tails], device=device)
        cache = _copy(base, mod, trunk.config, len(tails))
        h = trunk(input_ids=batch, past_key_values=cache, use_cache=True).last_hidden_state
        out.append(h[torch.arange(len(tails), device=device),
                     torch.tensor([len(t) - 1 for t in tails], device=device)])
        del cache, h
    return torch.cat(out), n + sum(len(x) - n for x in ids)


def _copy(cache, mod, config, rows: int):
    """The batch-1 cache repeated to `rows` rows; the original stays as it is for the next batch."""
    new = mod.Qwen3_5DynamicCache(config)
    for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        src, dst = getattr(cache, name), getattr(new, name)
        for i, t in enumerate(src):
            if t is not None:
                dst[i] = t.expand(rows, *t.shape[1:]).contiguous()
    return new


def install(mod) -> None:
    """Let the gated delta layers of `mod` continue a chunk of any length from a cache."""
    cls = mod.Qwen3_5GatedDeltaNet
    if getattr(cls, "_continues_chunks", False):
        return
    plain = cls.forward
    F = torch.nn.functional

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        B, L, _ = hidden_states.shape
        if (cache_params is None or L == 1 or not cache_params.has_previous_state
                or cache_params.conv_states[self.layer_idx] is None):
            return plain(self, hidden_states, cache_params=cache_params,
                         cache_position=cache_position, attention_mask=attention_mask)
        i, K = self.layer_idx, self.conv_kernel_size
        mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(B, L, -1, self.head_v_dim)
        b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
        # the convolution sees the last K-1 inputs of the prefix, as it would in one pass
        x = torch.cat([cache_params.conv_states[i][:, :, -(K - 1):].to(mixed.dtype), mixed], -1)
        cache_params.conv_states[i] = x[:, :, -K:].contiguous()
        mixed = F.silu(F.conv1d(x, self.conv1d.weight, None, groups=self.conv_dim)).transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, L, -1, self.head_k_dim)
        k = k.reshape(B, L, -1, self.head_k_dim)
        v = v.reshape(B, L, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        out, state = self.chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=cache_params.recurrent_states[i],
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        cache_params.recurrent_states[i] = state
        out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(B, L, -1))

    cls.forward = forward
    cls._continues_chunks = True
