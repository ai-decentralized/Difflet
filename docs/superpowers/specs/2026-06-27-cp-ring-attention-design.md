# Context-Parallel Ring Attention — Design

**Date:** 2026-06-27
**Status:** Approved design, pending implementation plan
**Scope:** Ring-attention context parallelism for self-attention across the 4 DiT
models that already have context parallelism (Wan, HunyuanVideo, Qwen-Image, Flux).
LTX-2 is deferred to a follow-up (it has no CP foundation today; adding ring there
requires building CP scatter/gather/wiring from scratch first).

## Summary

Difflet already supports context parallelism (CP) for several models via a
**gather-KV** strategy: each rank holds the query shard for its slice of the
sequence, all-gathers the full K,V along the sequence dimension, and runs one
full attention. Gather-KV is simple and fast when the full K,V fits in device
memory, but it materializes the entire-sequence K,V on every rank — the memory
ceiling that blocks long-sequence work (720p, longer-frame video).

This effort adds **ring attention** as an opt-in alternative CP strategy. Ring
attention keeps K,V sharded (memory O(S/cp), never the full sequence), rotates
K,V shards around the CP ring, and merges per-step partial attention via online
softmax. It is numerically lossless versus gather-KV (same softmax, reassociated).

We implement ring attention by calling the existing, production NKI kernel
`nkilib.experimental.attention.ring_attention_fwd.ring_attention_spmd_fwd`,
not by hand-rolling the ring loop in Difflet. That kernel already does exactly
what we need, built on the same `attention_cte` primitive Difflet already uses.

## Critical constraint: joint vs image-only self-attention

The prebuilt `ring_attention_spmd_fwd` runs a *uniform* ring over equally-sharded
K,V — a clean drop-in only when self-attention is over a single sharded sequence.

- **Wan** self-attention is **image-only** (text is a separate cross-attention).
  Ring is a clean drop-in. ✅
- **HunyuanVideo, Qwen-Image, Flux** use **joint (MMDiT) self-attention**: image
  (sharded) and text (replicated on every rank) tokens are concatenated and
  attended in one softmax. The prebuilt kernel has no notion of a replicated
  block every rank must attend, so ring is **not** a thin drop-in there — it needs
  a dedicated joint-ring design (e.g. image-ring + separate text attention merged
  by LSE; or a one-time text `k_prior/v_prior`; or scattering the whole joint
  sequence). That is its own design problem, deferred to a follow-up.

## Phasing

1. **This plan (increment 1):** shared `ring_attention` op + `cp_mode` config/CLI +
   **Wan** adapter + tests. Ships ring end-to-end for the one clean model.
2. **Follow-up:** joint-MMDiT ring design + plan for HunyuanVideo, Qwen-Image, Flux.
3. **Follow-up:** LTX-2 CP foundation + ring.

## Goals

- Add ring attention as a selectable CP mode for **self-attention** on the 4
  models with existing CP: Wan (2.1/2.2), HunyuanVideo (+1.5), Flux, Qwen-Image.
- Keep the validated gather-KV path as the default; ring is opt-in.
- One shared ring code path; per-model differences limited to tensor-layout glue.
- Leave the existing gather-KV path as-is (per-model, validated, untouched). Ring
  consolidation does not refactor gather-KV.
- User-selectable via a CLI flag and a config field.
- Numerically lossless versus gather-KV (parity gate cosine ≥ 0.999, target ~1.0).

## Non-goals (deferred)

- **LTX-2 entirely** — it has no CP foundation today; a separate follow-up plan
  builds LTX-2 CP scaffolding + ring (incl. its audio↔video cross-attention).
- Cross-attention generally: KV come from the (un-scattered) text encoder, so
  ring does not apply; cross-attention stays as-is.
- Auto-selecting ring vs gather-KV by memory heuristic — explicit opt-in only.
- Training / backward pass — Difflet is inference-only (`training=False`).

## Background: how CP works today

Per model (e.g. Wan `difflet/models/wan/modeling_wan.py`):

1. At DiT input, `hidden_states` and the RoPE (cos, sin) are scattered along the
   sequence dim once (`scatter_to_process_group_spmd`, partition_dim=1) across
   the CP group. Each rank keeps its `S/cp` shard.
2. In each block's self-attention, Q is the local shard; K,V are all-gathered to
   full sequence (`gather_from_tensor_model_parallel_region_with_dim`), then one
   full `attention_cte` runs.
3. At DiT output, `hidden_states` is gathered back along the sequence dim.

**The gather-KV all-gather in step 2 is the only site ring attention changes.**
Input scatter and output gather are unchanged.

## Why the nkilib kernel (feasibility findings)

`attention_cte` (the kernel all models already use) exposes the exact ring
building blocks: `cache_softmax=True` returns `(output, neg_max, sum)`, and
`skip_output_normalization=True` returns the *unnormalized* output plus the raw
softmax denominator S — its docstring names "ring attention" as the intended
caller.

`nkilib.experimental.attention.ring_attention_fwd.ring_attention_spmd_fwd`
composes these into a complete forward ring kernel. Verified importable on the
current box (`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/...`). It:

- Calls `attention_cte(..., cache_softmax=True, skip_output_normalization=True)`
  once per ring step.
- Rotates K,V with `nki.collectives.collective_permute_implicit`, **pipelined one
  step ahead** with ping-pong buffers (comm/compute overlap is built in).
- Performs the online-softmax reduction and a single final normalization in-kernel.
- Supports **non-causal** attention (`use_causal_mask=False`) — our exact
  bidirectional video/image DiT self-attention case — and causal/striped.
- `training=False` → inference only, returns `o: [b, h, seqlen, d]`.

Two alternatives were rejected:
- **NXD `nki_ring_attn_func`** (`neuronx_distributed/kernels/`): its underlying
  `neuronxcc.nki._pre_prod`/`_private` kernels are not importable on this box
  (falls back to a stub that raises); also training/causal-shaped.
- **Hand-rolled Python ring loop** over `attention_cte` + `xm.collective_permute`:
  works, but more code, slower (cp_degree separately-traced kernel+permute pairs),
  and re-implements what the nkilib kernel already optimizes.

## Design

### 1. Activation: `cp_mode` config + CLI

- **CLI:** add `--cp-mode {gather_kv,ring}` (default `gather_kv`) to
  `difflet/cli/main.py` and `difflet/cli/stage.py`, alongside `--cp-degree`.
- **Config:** `DiffletParallelConfig` (`difflet/pipeline/parallel_config.py`)
  gains `cp_mode: str = "gather_kv"`. `__post_init__` validates the value is one
  of `{"gather_kv", "ring"}` and that `cp_mode == "ring"` implies `cp_degree > 1`.
- **Compile cache:** `cp_mode` is added to `to_cache_dict()`. Ring and gather-KV
  compile to different NEFF graphs, so the mode must be part of the cache key to
  avoid an incorrect cache hit.
- **Threading:** the 5 orchestrators (`difflet/cli/orchestrators/{wan,
  hunyuan_video,qwen_image,flux,ltx_2}.py`) pass `cp_mode=args.cp_mode` into
  `DiffletParallelConfig`; the staged orchestrators (wan, hunyuan, qwen) re-emit
  `--cp-mode` to their subprocesses. `cp_mode` then reaches the model config
  alongside `context_parallel_enabled`, selecting the ring op vs the gather branch.

### 2. Shared ring op (thin wrapper)

A single op in `difflet/backends/trainium/ops_impl/attention.py`, exposed via
`difflet/ops/attention.py`:

```python
def ring_attention(q, k, v, *, scale, cp_mesh, cp_degree, tp_q, tp_k, causal=False):
    # q,k,v: [B, H, S_local, d]; tp_q/tp_k indicate transposed vs not, per attention_cte.
    return ring_attention_spmd_fwd(
        q, k, v,
        replica_groups=cp_mesh,    # get_cp_group_mesh(tp_degree, cp_degree)
        num_workers=cp_degree,
        softmax_scale=scale,
        use_causal_mask=causal,    # False for these DiT self-attentions
        training=False,            # inference: no LSE / backward
        tp_q=tp_q, tp_k=tp_k,
    )
```

The op guards the experimental import with a clear, actionable error if the
kernel is unavailable (mirroring NXD's `_ring_attn_placeholder` pattern), so a
missing/renamed kernel fails loudly at setup rather than mysteriously at compile.

### 3. CP ring topology

`replica_groups` is the ring membership, which must be **the same group the model
scatters Q with** so K,V rotate consistently. The models scatter using
`get_data_parallel_group()` (from `neuronx_distributed.parallel_layers.parallel_state`),
so the ring uses that same group's replica list:

- `cp_mesh = get_data_parallel_group(as_list=True)` (== `get_data_parallel_replica_groups()`,
  returns `_DATA_PARALLEL_GROUP_SPMD` — list-of-lists of global ranks).
- `num_workers = get_data_parallel_size()`.

This is consistent with the scatter **by construction** (same group object), so no
reconciliation is needed. NOTE: do *not* use `get_cp_group_mesh()` from
`attention_process_groups.py` — that helper describes the core attention's
TP-shared CP scheme (`world = tp_degree`), which is a different partitioning from
the model-level CP used here (`world = tp_degree × cp_degree`,
`dp_rank = global_rank // tp_degree`).

Because the mesh and size come from these global accessors (which the models
already call to get `data_parallel_group`), the only thing that must be threaded
through config to the attention module is the **mode** (`cp_mode`).

### 4. Per-model adapters (~10 lines each)

**Shared vs per-model (design decision):** the ring *algorithm* is fully shared —
the kernel call, the `replica_groups`/`num_workers` topology, and the constraint
guards (MHA, head_dim ≤ 128, seqlen % 128) all live in the single `ring_attention`
op. No model re-implements ring. The only per-model code is tensor-layout
normalization into the kernel's `[B,H,S_local,d]` form and back, because each
model's per-head tensors are laid out differently. This is strictly more shared
than gather-KV (which is duplicated across models). The gather-KV path is left
per-model and untouched; this effort does not unify the two CP modes into one op.


Each model's self-attention CP branch, when `cp_mode == "ring"`, replaces its
inline `stacked_kv = gather(...)` + full attention with: arrange Q,K,V into the
kernel's `[B, H, S_local, d]` layout (using the model's existing per-head
tensors), call `ring_attention(...)`, and arrange the result back. The current
gather-KV branch remains for `cp_mode == "gather_kv"`.

Per-model layout origins (today's gather axis indicates each layout):
- Wan `[2,B,heads,S/cp,d]`, HunyuanVideo `[2,B,S/cp,H,d]`, Flux `[S/cp,B*H,d]`,
  plus Qwen-Image and LTX-2 self-attention.

The ring op, topology, and constraint checks are written once and shared.

### 5. Correctness and masking

- Self-attention here is **bidirectional** → `use_causal_mask=False`. None of the
  causal-CP machinery (cp_offset / strided / striped) is needed.
- Sequence sharding is unchanged from gather-KV (same `S/cp` shards).
- Ring is numerically lossless versus gather-KV: identical softmax, reassociated
  via online-softmax merge in the kernel.

### 6. Kernel constraints (adapters guard / pad)

The kernel asserts these; adapters must satisfy or guard them:
- **Trainium2+** (trn3 target ✓).
- **MHA only**: q_heads == kv_heads per rank — true for these DiT self-attentions
  (no GQA). Add an explicit guard.
- **head_dim ≤ 128** (Wan/Hunyuan = 128 ✓).
- **seqlen-per-rank divisible by 128** (and the kernel's K tile size). Document
  and enforce the shard/padding requirement; surface a clear error otherwise.

### 7. Performance

Comm/compute overlap is implemented inside the kernel (one-step-ahead collective
pipeline + ping-pong buffers). No Difflet-side overlap work. Ring trades
`cp_degree` serialized attention steps for O(S/cp) K,V memory; its win over
gather-KV is memory headroom for long sequences, not raw speed at short sequences
— which is why gather-KV stays the default.

## Validation (TDD)

1. **Unit (host):** validate the ring topology/mesh construction and the
   `cp_mode` config validation + cache-key inclusion without a device.
2. **Device parity:** ring vs gather-KV, per model, same inputs — cosine ≥ 0.999
   (expect ~1.0, lossless). This is the primary correctness gate.
3. **End-to-end:** at least one model (Wan) full denoise trajectory with
   `cp_degree > 1, cp_mode=ring`, cosine vs the existing validated path.
4. Reuse the existing numerical harness patterns under `tests/numerical/`.

## File change list

- `difflet/cli/main.py`, `difflet/cli/stage.py` — `--cp-mode` flag.
- `difflet/pipeline/parallel_config.py` — `cp_mode` field, validation, cache key.
- `difflet/cli/orchestrators/{wan,hunyuan_video,qwen_image,flux,ltx_2}.py` —
  pass `cp_mode`; re-emit `--cp-mode` in staged orchestrators.
- `difflet/backends/trainium/ops_impl/attention.py` — `ring_attention` wrapper +
  guarded import.
- `difflet/ops/attention.py` — expose ring entry.
- Model self-attention files (Wan, HunyuanVideo(+1.5), LTX-2, Flux, Qwen-Image) —
  per-model adapter at the existing gather site, selected by `cp_mode`.
- `tests/` — config/topology unit tests; per-model device parity; Wan e2e.

## Open items for the implementation plan

- Reconcile/verify scatter group (`get_data_parallel_group`) vs ring
  `replica_groups` (`get_cp_group_mesh`) ring membership and ordering.
- Confirm each model's self-attention is MHA at the per-rank level (no GQA).
- Confirm seqlen-per-rank is a multiple of 128 for the targeted resolutions, or
  define the padding strategy.
- Pin/guard the experimental nkilib import against SDK version drift.
