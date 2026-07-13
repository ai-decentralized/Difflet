# CP Ulysses mode — design

**Date:** 2026-07-13
**Status:** approved for implementation

## Goal

Add `cp_mode="ulysses"` (DeepSpeed-Ulysses / all-to-all context parallelism) alongside
the existing `gather_kv` and `ring` CP modes, wire it through the CLI, and verify it
runs and is numerically correct on device.

## Why Ulysses

The two existing modes trade the same thing differently:

- **`gather_kv`** — every rank all-gathers the full K,V. Communication is
  `O(S · H_local · d)` per rank *per attention*, and each rank then attends its
  `S/cp` queries against the **full** S keys. Simple, exact, but the KV all-gather
  grows with sequence length and every rank redundantly materializes all of K,V.
- **`ring`** — K,V shards rotate around the cp ring; partials merge by online
  softmax. Communication is the same total volume but overlapped/pipelined, and
  memory is `O(S/cp)` per hop. Requires the experimental
  `nkilib.experimental.attention.ring_attention_fwd` kernel, and the joint variant
  is a hand-rolled `collective_permute` ring with an online-softmax merge.

**Ulysses** instead shards the *head* axis during attention: an all-to-all converts
the `[B, H_local, S/cp, d]` sequence-sharded layout into `[B, H_local/cp, S, d]`
head-sharded-full-sequence, runs one **ordinary dense attention** over the full
sequence, then a second all-to-all converts back. Its advantages here:

1. **Comm volume is `O(S/cp · H_local · d)` per rank** — a factor of `cp` less than
   the KV all-gather, and it moves Q,K,V once rather than K,V-per-hop.
2. **It needs no special attention kernel.** The inner call is the plain
   `attention_cte` every non-CP path already uses, so there is no online-softmax
   merge, no experimental nkilib dependency, and no TRN1 fallback (`ring` today
   warns and falls back to gather-KV on TRN1 — see `modeling_flux.py:1374`).
3. **It is exact by construction** — a single dense softmax over the full sequence,
   the same math as `gather_kv`. No partial-merge numerics to validate.

Its cost is the constraint `H_local % cp == 0` (below).

## The core transform

Given this rank's `q,k,v` of shape `[B, H_local, S/cp, d]` where `H_local = H/tp`
and the sequence is **contiguous-block sharded** over the cp axis (rank `r` holds
sequence block `r` — which is exactly how CP scatter/gather works today):

```
a2a(split_dim=1 (heads), concat_dim=2 (seq), split_count=cp)   →  [B, H_local/cp, S, d]
dense attention over the full S                                →  [B, H_local/cp, S, d]
a2a(split_dim=2 (seq),  concat_dim=1 (heads), split_count=cp)  →  [B, H_local, S/cp, d]
```

XLA `AllToAll` semantics make this correct without any rank-dependent indexing: on the
forward a2a, rank `r` receives head-block `r` from **every** rank `r'`, and the
receives are concatenated along seq in replica-group order — and rank `r'` holds
sequence block `r'`, so the concatenation reproduces the *global* sequence order.
The inverse a2a restores the original head order by the same argument. This is
plain SPMD: one traced graph, identical on every rank.

`xm.all_to_all(value, split_dimension, concat_dimension, split_count, groups)` is
available in the installed torch_xla and takes replica `groups`, so it maps onto
the cp mesh (`get_cp_mesh()`) exactly like the existing ring collectives.

### Constraint: `H_local % cp == 0`

Heads are sharded twice — first by TP, then by CP inside attention. So the model's
`num_attention_heads` must be divisible by `tp * cp`. At the tp2/cp2 verification
target all four CP models satisfy this comfortably:

| model | heads | `H_local` @tp2 | `H_local/cp` @cp2 |
|---|---|---|---|
| wan | 40 | 20 | 10 |
| flux | 24 | 12 | 6 |
| hunyuan_video | 24 | 12 | 6 |
| qwen_image | 24 | 12 | 6 |

This is validated with a clear error at attention-construction time rather than
failing deep inside the compiler.

## Two ops, mirroring the ring pair

The models split into exactly two shapes, so Ulysses needs the same two entry points
`ring` already has (`difflet/ops/attention.py`):

### `ulysses_attention(q, k, v, *, scale, causal=False)`
For a **uniformly sharded** sequence. Used by **wan** (plain self-attention) and
**flux** (which shards the *joint* text+image sequence uniformly, so its q/k/v are a
single sharded sequence — it already reuses plain `ring_attention` for the same
reason). Straight application of the core transform above.

### `joint_ulysses_attention(q_img, q_txt, image_k, image_v, text_k, text_v, *, scale, causal=False) -> (img_out, txt_out)`
For **hunyuan_video** and **qwen_image**, where the image stream is sequence-sharded
but the text stream is **replicated** on every rank.

- **Image q,k,v** (sharded): a2a heads→seq, as above.
- **Text q,k,v** (replicated): each rank needs the text restricted to *its* head
  block. Rather than a rank-dependent slice (which would need SPMDRank plumbing into
  two more attention modules), apply the *same* a2a to the replicated text: because
  every rank holds identical text, rank `r` receives `cp` **identical** copies of
  head-block `r` concatenated along seq, giving `[B, H_local/cp, cp·S_txt, d]`.
  Slicing the first `S_txt` yields exactly this rank's head-block text — a **static,
  rank-agnostic** slice. Cost is a small extra all-to-all on the short text tensor.
- Attend the concatenated `[image ‖ text]` keys with the concatenated query in one
  dense attention.
- **Image output** → inverse a2a back to `[B, H_local, S_img/cp, d]`.
- **Text output** is `[B, H_local/cp, S_txt, d]` and must be restored to the
  replicated `[B, H_local, S_txt, d]` the callers expect → all-gather along the head
  dim over the cp group.

Returning `(img_out, txt_out)` separately (rather than one concatenated joint tensor
like `joint_ring_attention` does) is deliberate: hunyuan concatenates `[img ‖ txt]`
while qwen concatenates `[txt ‖ img]`, and the two streams have *different* shardings
under Ulysses, so the op cannot take a pre-concatenated query. Each caller
reassembles in its own order.

**CPU backend** (`cp_degree == 1`): both ops degenerate to plain dense attention,
mirroring the existing CPU `ring_attention` / `joint_ring_attention` references.

## Wiring

1. **`ops/attention.py`** — declare the two new ops; **`backends/cpu/ops_impl/attention.py`**
   and **`backends/trainium/ops_impl/attention.py`** implement them; register in
   `ops/__init__.py`.
2. **Models** — add a `cp_mode == "ulysses"` branch beside each existing `"ring"`
   branch: `modeling_wan.py:545`, `modeling_flux.py:1302`,
   `modeling_hunyuan_video.py:466`, `backends/trainium/qwen_image/transformer.py:372`.
   Like ring, Ulysses rejects `attention_mask` (no caller uses one on the CP path).
3. **`pipeline/parallel_config.py`** — accept `"ulysses"` in the `cp_mode` validity
   set and require `cp_degree > 1` for it (same rule as ring). The compile-cache key
   needs **no** change: `to_cache_dict()` elides `cp_mode` only at its `gather_kv`
   default, so `"ulysses"` is emitted and hashes to a distinct artifact automatically.
4. **CLI** — add `ulysses` to the `--cp-mode` choices in **both** parsers
   (`cli/main.py:42` and `cli/stage.py:43`; the literal is hand-duplicated today).

### Staged-artifact collision — a real bug this change would otherwise hit

The staged compiled-artifact directory names (e.g. `WanOrchestrator._stage_compiled_dir`
→ `wan_transformer_tp{tp}cp{cp}{cfg}{sp}_h{h}w{w}f{f}`) **do not include `cp_mode`**.
Two compiles that differ only in `cp_mode` therefore collide in `~/.cache/difflet`
even though their compile-cache hashes differ — a gather_kv artifact would be silently
reused for a ulysses run. This already latently affects `ring`; adding a third mode
makes it acute. **Fold `cp_mode` into the staged dir name** (non-default modes only, so
existing gather_kv cache dirs keep their current names and stay valid).

## Verification

Two layers, matching how ring was verified:

1. **Numerical parity (correctness/accuracy).** Extend the existing
   `scripts/*_ring_parity_smoke.py|.sh` runners, which already run **one cp_mode per
   process** (NxD's `parallel_state` initializes once per process) and compare saved
   output latents across processes. Add `ulysses` to the `--mode` choices and compare
   **ulysses vs gather_kv** — Ulysses is exact, so this must be lossless
   (cosine ≥ 0.999, same gate the ring smokes use).
2. **End-to-end CLI liveness.** Add a `tp2cp2ulysses` cell to
   `scripts/verify_cli.py`'s `PARALLEL_CONFIGS`
   (`--tp-degree 2 --cp-degree 2 --cp-mode ulysses`, world_size 4 — satisfies the
   script's asserted `world_size == 4` invariant). `verify_cli.py` checks
   plumbing/liveness (exit code + artifact existence), **not** numerics — that is what
   layer 1 is for. The drift-guard unit tests in `tests/unit/cli/test_verify_cli.py`
   pin the config list, flags, skip table and cell counts exactly, so they are updated
   in the same commit.

## Out of scope

- **ltx_2** — has no context parallelism at all; nothing to add a mode to.
- **hunyuan_video_15** — CP-unsupported scaffold (already `CP_UNSUPPORTED` in verify_cli).
- Combining Ulysses with SP or CFG parallel — CP remains mutually exclusive with both,
  unchanged.
- Performance benchmarking / making Ulysses the default. This lands the mode,
  correct and reachable; ranking it against ring is a separate benchmark task.
