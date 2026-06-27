# Joint-MMDiT Ring Attention (HunyuanVideo, Qwen-Image, Flux) — Design

**Date:** 2026-06-27
**Status:** Approved design, pending implementation plan
**Scope:** Extend the opt-in `cp_mode=ring` context-parallel attention (shipped for
Wan) to the three **joint-MMDiT** DiT models that already have CP gather-KV:
HunyuanVideo (+1.5), Qwen-Image, Flux. LTX-2 remains deferred (no CP foundation).

This is **increment 2** of the ring-attention effort. Increment 1 (Wan, image-only
self-attention) is described in
`docs/superpowers/specs/2026-06-27-cp-ring-attention-design.md` and shipped in
commit `7114975`.

## Summary

The Wan increment added ring attention for **image-only** self-attention: a clean
drop-in of the prebuilt NKI kernel `ring_attention_spmd_fwd`, because Wan attends a
single uniformly-sharded sequence.

HunyuanVideo, Qwen-Image, and Flux use **joint (MMDiT) self-attention**: image and
text tokens are concatenated and attended in **one softmax**. But they split into
**two structurally different cases** by how text is partitioned under CP — a
distinction verified by reading each model's input scatter and gather-KV site:

- **Text-replicated (HunyuanVideo (+1.5), Qwen-Image):** only image tokens are
  sequence-sharded; text is replicated on every rank. The joint query
  (`[full text ‖ image shard]`) is **longer** than the rotating image shard, and the
  prebuilt kernel has **no slot for the replicated text block** every query must also
  attend. This needs a *joint-ring decomposition* (new op) — the core of this design.
- **Text-sharded (Flux):** Flux scatters **both** streams along the sequence dim
  (`modeling_flux.py:428-436`, `split_along_dim` on `hidden_states` *and*
  `encoder_hidden_states`). Its joint query and its per-rank K,V shard are the **same**
  joint length (`[text shard ‖ image shard]`), so a *uniform* ring over the
  concatenated shard is a direct equivalent — it **reuses the existing
  `ring_attention` op** (the Wan path), with no replicated-text block, no LSE merge,
  and no spike.

This design adds ring as an opt-in CP strategy for all three models, keeping
gather-KV as the untouched default. It is numerically lossless versus gather-KV
(same softmax, reassociated via online-softmax merge / key reordering under a
non-causal mask).

## Goals

- Extend `cp_mode=ring` to self-attention on HunyuanVideo (+1.5), Qwen-Image, Flux.
- Keep the validated gather-KV path as the default; ring is opt-in and additive.
- For text-replicated models (HunyuanVideo, Qwen-Image): one shared
  `joint_ring_attention` op; per-model differences limited to tensor-layout glue
  (~10–15 lines each), mirroring the Wan increment's structure.
- For Flux (text-sharded): reuse the existing `ring_attention` op; per-model work is
  the layout glue that concatenates the two streams into one joint shard.
- Numerically lossless versus gather-KV (parity gate cosine ≥ 0.999, target ~1.0).

## Non-goals (deferred)

- **seqlen-%-128 padding / long-sequence memory demonstration.** Parity-only
  success bar this increment, matching the Wan increment. The memory win (O(S/cp)
  K,V) is real but proving it on an OOM-under-gather-KV config — which forces a
  padding/masking strategy for arbitrary shard lengths — is a follow-up.
- **Performance characterization** (ring's serialized steps vs gather-KV latency).
- **LTX-2** — no CP foundation today (entry guard raises on `cp_degree > 1`); dual
  video+audio streams + audio↔video cross-attention. Separate follow-up.
- **Cross-attention** — K,V come from the un-scattered text encoder, so ring does
  not apply; cross-attention stays as-is.
- **Training / backward pass** — inference only (`training=False` semantically; see
  the LSE note in §2).

## Background: joint-MMDiT self-attention today (per model)

Every model scatters **image** tokens along the sequence dim at DiT input
(`scatter_to_process_group_spmd`, partition_dim = sequence) across the CP
(data-parallel) group, all-gathers image K,V to full sequence in self-attention, runs
one full joint attention, and gathers the image sequence back at output. The two
cases differ in **whether text is also sharded**:

| Model | self-attn site | text under CP | joint concat | image gather-KV layout | head_dim |
|---|---|---|---|---|---|
| HunyuanVideo (+1.5) | `difflet/models/hunyuan_video/modeling_hunyuan_video.py` `HunyuanVideoAttention.forward` (`dual_stream_attention`) | **replicated** | `cat([image, text], dim=1)` pre-projection | `[2, B, S/cp, H, d]` (gather dim 2) | 128 |
| Qwen-Image | `difflet/backends/trainium/qwen_image/transformer.py` `_QwenImageTrainiumAttnProcessor` | **replicated** | `cat([text, image], dim=1)` post-projection | `[2, B, S/cp, H, d]` (gather dim 2) | config |
| Flux | `difflet/models/flux/modeling_flux.py` `NeuronFluxAttention.forward` | **sharded** (both streams, `split_along_dim` at `:428-436`) | `cat([text, image], dim=2)` post-projection | `[2, B, H, S/cp, D]` (gather dim 3) | 64 |

All three are **MHA** per rank (q_heads == kv_heads, no GQA). RoPE is applied after
the scatter and before the gather in every model.

**Flux's two block types are both uniform-ring-friendly:**
- *Double-stream* (`NeuronFluxTransformerBlock`, `encoder_hidden_states is not None`):
  image and text both sharded; gather-KV builds full joint K,V while the query stays
  the local joint shard `cat([text_shard, image_shard])`. Query length == per-rank K,V
  shard length, so a uniform ring over the concatenated shard is a direct equivalent.
- *Single-stream* (`NeuronFluxSingleTransformerBlock`, `encoder_hidden_states is None`):
  one already-concatenated sharded sequence;
  `attention_wrapper_context_parallel_single_transformer` (`:133`) gathers K,V with a
  sharded query — identical shape to the Wan ring. (Flux's TRN1 / masked-SDPA branches
  stay gather-KV; the plan enumerates exactly which branches get ring.)

## Why the text-replicated case needs more than the prebuilt kernel

`nkilib.experimental.attention.ring_attention_fwd.ring_attention_spmd_fwd`:

- Rotates an equally-sharded K,V around the ring and merges per-step partials via
  online softmax internally. Built for self-attention over **one uniformly-sharded
  sequence** — so it is a **direct drop-in for Flux** (text sharded; query and K,V
  shard share the joint length), exactly as it was for Wan.
- For the **text-replicated** models it is **not** a drop-in: it exposes no
  `k_prior`/`v_prior` (no slot for the non-rotated replicated text block) and **does
  not return softmax stats** in inference — only the normalized output `o`. It returns
  `lse` (`[b, h, 128, seqlen/128]`) **only when `training=True`**.

The underlying primitive `nkilib.core.attention.attention_cte` **does** expose the
building blocks the text-replicated case needs: `k_prior`/`v_prior` (an extra KV block
folded into the same softmax) and `cache_softmax=True` (returns `(output, neg_max,
sum)`; `skip_output_normalization=True` makes the output unnormalized and the third
element the raw denominator S). These are how a separate text partial gets merged.

So for HunyuanVideo/Qwen-Image, ring is feasible but **not** by calling
`ring_attention_spmd_fwd` as a black box: the new op reuses it for the **image** ring
and merges the **text** partial outside the kernel (§2). Flux skips all of this and
reuses the existing `ring_attention` op (§3).

## Design

### 1. Text-replicated case — the joint-ring decomposition (Approach B)

Applies to **HunyuanVideo (+1.5) and Qwen-Image** (text replicated). Split the joint
softmax over keys `[image | text]` into two partials and flash-merge
them. Online softmax is associative, so this is numerically identical to one joint
softmax over the concatenated keys.

For each rank, with `q_joint_local` = this rank's **full** local query set (its image
shard **plus** the replicated text queries):

1. **Image partial (ring).**
   `ring_attention_spmd_fwd(q_joint_local, image_K_sharded, image_V_sharded,
   use_causal_mask=False, training=True)` → normalized `o_image` + `lse_image`.
   The ring rotates only the image K,V, so every local query — image and text alike
   — attends all image keys. `training=True` is used **solely** to get `lse_image`
   back out; no backward pass is run.
2. **Text partial (local).**
   `attention_cte(q_joint_local, text_K, text_V, cache_softmax=True)` →
   `o_text` + `lse_text`. Text K,V is replicated, so this is rank-local — no
   collective.
3. **Merge.**
   `lse = logaddexp(lse_image, lse_text)`;
   `o = o_image · e^(lse_image − lse) + o_text · e^(lse_text − lse)`.
   Result is the full joint output for the local queries.

The model's existing output-gather reassembles the image sequence. Text-query
output is identical on every rank (replicated text Q, and every rank's ring sees all
image K), consistent with gather-KV today.

### 2. Risk retired first: joint-Q seqlen + inference LSE

Approach B rests on one unverified kernel behavior:

- **Joint-Q seqlen.** The ring rotates an image K shard of length `S_img/cp`, but
  `q_joint_local` has length `S_img/cp + S_txt`. `ring_attention_spmd_fwd` is built
  for self-attention where Q and the per-rank K shard share a length. For our
  **non-causal** case there is no positional masking, so attention is a plain
  `[Sq × Sk_shard]` per step — but the kernel may still assume `Sq == Sk_shard` in
  its tiling. This must be confirmed on device.
- **Inference LSE.** Confirm `training=True` can be used purely to retrieve
  `lse_image` in our inference path (cost/shape acceptable), without triggering a
  backward.

**Mitigation:** a device spike is **task 1** of the implementation plan, gating all
model wiring. If the kernel rejects `Sq ≠ Sk_shard` or the `training=True` LSE is
unusable, fall back to **Approach A**: a hand-rolled ring over `attention_cte` with
the replicated text as `k_prior`/`v_prior` and `cache_softmax` /
`skip_output_normalization` for the cross-step merge — same decomposition, different
kernel mechanics, more low-level code. The op's public interface (§3) is identical
under either mechanism, so the per-model adapters do not change.

### 3. Shared op + per-model adapters

**One shared op**, Trainium impl + CPU reference, exposed via `difflet.ops`
(mirroring the `ring_attention` op added for Wan):

```python
def joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale, causal=False):
    # q                [B, H, S_img/cp + S_txt, d]   this rank's joint local queries
    # image_k, image_v [B, H, S_img/cp, d]           sharded — rotated by the ring
    # text_k,  text_v  [B, H, S_txt,    d]           replicated — local partial
    # returns          [B, H, S_img/cp + S_txt, d]
```

The op owns the **entire algorithm** — the ring call, the local text `attention_cte`,
the LSE merge, and the kernel constraint guards (MHA, head_dim ≤ 128, per-rank shard
divisible by 128). No model re-implements ring. Because image and text K are passed
as **separate arguments**, the op is concat-order-agnostic.

- **Trainium impl** (`difflet/backends/trainium/ops_impl/attention.py`): the §1
  decomposition, with a guarded experimental import mirroring the existing
  `ring_attention` op (clear, actionable error if the kernel is unavailable).
- **CPU reference** (`difflet/backends/cpu/ops_impl/attention.py`): with cp=1 the
  image is not sharded, so this is plain full joint attention over
  `cat([image_k, text_k])` / `cat([image_v, text_v])` — the unit-test oracle for the
  merge math.

**HunyuanVideo/Qwen-Image adapters (~10–15 lines each)** at the existing gather site,
selected when `context_parallel_enabled and not is_cross_attention and cp_mode ==
"ring"`. Their only job is normalizing each model's per-head layout (table in
Background) into the op's `[B, H, S, d]` contract and arranging the result back,
splitting `o` by the same indices used to assemble `q`. The current gather-KV branch
remains for `cp_mode == "gather_kv"`.

### 3a. Text-sharded case — Flux uniform-ring (reuse existing op)

Applies to **Flux** (both streams sharded). No new op, no decomposition, no spike —
because Flux's query and per-rank K,V shard share the joint length, ring is the same
uniform drop-in used for Wan. The adapter, when `cp_mode == "ring"`, replaces the
gather-KV branch with a call to the **existing** `difflet.ops.ring_attention`:

- **Double-stream blocks** (`NeuronFluxTransformerBlock`): instead of gathering image
  K,V and text K,V to full, build the local joint K,V/Q
  `cat([text_shard, image_shard], dim=seq)` and call `ring_attention(q, k, v,
  scale=1/√d)`. Output is the local joint shard; the existing
  `[:enc_len] / [enc_len:]` split into image/text outputs is unchanged.
- **Single-stream blocks** (`NeuronFluxSingleTransformerBlock`): the sequence is
  already one sharded joint tensor; replace the
  `attention_wrapper_context_parallel_single_transformer` gather with a direct
  `ring_attention(q, k, v, scale=1/√d)` over the sharded Q/K/V.
- **Untouched branches:** Flux's masked-SDPA and TRN1 paths stay gather-KV; ring wires
  only into the non-masked, non-TRN1 CP paths above (the plan enumerates them).

### 4. CP ring topology

Identical to the Wan increment for both ops: `replica_groups` / `num_workers` are
resolved **inside** `ring_attention` / `joint_ring_attention` from
`get_data_parallel_group(as_list=True)` / `get_data_parallel_size()` — the same group
each model scatters with, so K,V rotate consistently by construction. The only thing
threaded through config to the attention module is the **mode** (`cp_mode`).

### 5. Config, CLI, and threading

The activation machinery already exists from the Wan increment and is reused
unchanged:

- `DiffletParallelConfig.cp_mode` (`{"gather_kv", "ring"}`, default `gather_kv`,
  additive cache key) and `--cp-mode` on `difflet/cli/main.py` /
  `difflet/cli/stage.py`.

This increment adds threading for the three models, following the Wan pattern:

- Orchestrators `difflet/cli/orchestrators/{hunyuan_video, hunyuan_video_15,
  qwen_image, flux}.py` pass `cp_mode=args.cp_mode` into `DiffletParallelConfig`;
  staged orchestrators re-emit `--cp-mode` to their subprocesses.
- Each model config + attention module carries `cp_mode` (default `"gather_kv"`) and
  selects the ring branch, mirroring how `context_parallel_enabled` is already
  threaded.

**LNC2 launch:** the ring kernel requires `NEURON_RT_VIRTUAL_CORE_SIZE=2` for the
SPMD grid (same as Wan ring). The design keeps the existing convention — the value
is supplied via the environment and the runner's `setdefault` leaves a user-set
value untouched — rather than introducing new orchestrator logic.

### 6. Correctness and masking

- Joint self-attention is **bidirectional** → `use_causal_mask=False`. No causal-CP
  machinery (cp_offset / strided / striped) is needed.
- Sequence sharding is unchanged from gather-KV (HunyuanVideo/Qwen: `S_img/cp` image
  shards, text replicated; Flux: both streams sharded).
- Lossless versus gather-KV: identical softmax — reassociated via the online-softmax
  LSE merge (text-replicated case) or via key reordering under a non-causal mask
  (Flux uniform ring).

### 7. Kernel constraints (op guards)

Both ops guard these (kernel asserts them):

- **Trainium2+**.
- **MHA only** (q_heads == kv_heads per rank) — true for all three (no GQA).
- **head_dim ≤ 128** (Hunyuan 128, Flux 64, Qwen config-driven — guard).
- **Per-rank shard divisible by 128** — the rotating image shard (HunyuanVideo/Qwen)
  or the joint `[text ‖ image]` shard (Flux). Document and surface a clear error if
  violated. (Padding to satisfy this is a non-goal this increment; at supported
  resolutions the shards are expected to already satisfy it — the parity tests confirm
  per model.)

## Validation (TDD, parity-only)

1. **Unit (host):** `joint_ring_attention` CPU equivalence vs. full joint attention
   (the merge math, no device); `cp_mode` threading per model.
2. **Device spike (text-replicated case, before Hunyuan/Qwen wiring):** confirm the
   joint-Q-seqlen + inference-LSE behavior (§2). Flux needs no spike.
3. **Device parity:** per-model env-gated NEFF tests
   (`DIFFLET_RUN_<MODEL>_RING_NEFF=1`), ring vs gather-KV, cosine ≥ 0.999. This is
   the primary correctness gate, run independently per model.
4. **End-to-end:** at least one model (e.g. HunyuanVideo) short denoise trajectory
   with `cp_degree > 1, cp_mode=ring`, cosine vs the validated gather-KV path.
5. Reuse the existing `tests/numerical/` harness patterns (env-var-gated NEFF, as in
   `test_wan_ring_attention_neff.py`).

## File change list

- `difflet/backends/trainium/ops_impl/attention.py` — `joint_ring_attention` impl +
  guarded import (text-replicated case). `ring_attention` is reused as-is for Flux.
- `difflet/backends/cpu/ops_impl/attention.py` — `joint_ring_attention` CPU reference.
- `difflet/ops/attention.py`, `difflet/ops/__init__.py` — expose
  `joint_ring_attention`.
- `difflet/models/hunyuan_video/modeling_hunyuan_video.py`,
  `difflet/backends/trainium/qwen_image/transformer.py` — `joint_ring_attention`
  adapter at the gather site, selected by `cp_mode`; thread `cp_mode` to the module.
- `difflet/models/flux/modeling_flux.py` — `ring_attention` branch in the
  double-stream and single-stream CP paths, selected by `cp_mode`; thread `cp_mode`.
- `difflet/cli/orchestrators/{hunyuan_video, hunyuan_video_15, qwen_image,
  flux}.py` — pass `cp_mode`; re-emit `--cp-mode` in staged orchestrators.
- Model configs / backbone config carriers (per model) — carry `cp_mode`.
- `tests/unit/` — op CPU-equivalence + per-model threading tests.
- `tests/numerical/` — per-model device parity + one e2e trajectory.

## Open items for the implementation plan

- **Spike (Hunyuan/Qwen only):** joint-Q seqlen support and inference-LSE usability of
  `ring_attention_spmd_fwd`; choose Approach B (reuse) vs A (hand-rolled) before
  wiring HunyuanVideo/Qwen. Flux is unaffected and can land first.
- Enumerate Flux's branches precisely (double-stream CP, single-stream non-masked
  non-TRN1 CP get ring; masked-SDPA and TRN1 stay gather-KV).
- Confirm per-rank shard is 128-divisible for each model's supported resolutions —
  image shard (Hunyuan/Qwen), joint `[text ‖ image]` shard (Flux) (else clear error;
  padding deferred).
- Confirm Qwen-Image `head_dim ≤ 128` for its shipped config.
- Confirm whether HunyuanVideo 1.5 shares `modeling_hunyuan_video.py`'s attention
  module (one adapter serves both) or needs its own adapter.
- Pin/guard the experimental nkilib import against SDK drift (as in increment 1).
