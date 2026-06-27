# Joint-MMDiT & Flux Ring Attention (increment 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the opt-in `cp_mode=ring` context-parallel self-attention (shipped for Wan) to the three remaining CP-enabled DiT models — Flux (uniform-ring reuse), HunyuanVideo (+1.5), and Qwen-Image (joint-ring decomposition) — keeping gather-KV the default.

**Architecture:** Two cases split by how text is partitioned under CP. **Flux** shards *both* streams, so its joint query and per-rank K,V shard share the joint length → reuse the existing `difflet.ops.ring_attention` op over the concatenated `[text ‖ image]` shard (a Wan-style uniform ring). **HunyuanVideo / Qwen-Image** replicate text, so the joint query is longer than the rotating image shard → a new shared `joint_ring_attention` op rings the sharded image K,V (via `ring_attention_spmd_fwd(training=True)` for the LSE) and merges a local replicated-text partial (`attention_cte(cache_softmax=True)`) by online softmax. Per-model work is thin layout glue selected by `cp_mode`; gather-KV is untouched.

**Tech Stack:** Python, PyTorch, `torch_neuronx`, `neuronx_distributed` (parallel state / collectives), `nkilib.experimental.attention.ring_attention_fwd.ring_attention_spmd_fwd`, `nkilib.core.attention.attention_cte`, pytest.

## Global Constraints

- **Scope:** Flux, HunyuanVideo (+1.5), Qwen-Image self-attention only. LTX-2 deferred (no CP foundation). Cross-attention untouched (KV from un-scattered text encoder).
- **Default is `gather_kv`:** ring activates only when `context_parallel_enabled` (cp_degree > 1) AND `cp_mode == "ring"` AND not cross-attention. The `cp_mode` config field, `--cp-mode` CLI flag, and additive compile-cache key already exist from the Wan increment — do NOT re-add them.
- **Success bar: parity-only.** Ring vs gather-KV cosine ≥ 0.999 per model. seqlen-%-128 padding / long-sequence memory demo and perf characterization are OUT of scope.
- **Backend-neutral models:** `modeling_*.py` / Trainium processors reach kernels only via `difflet.ops`; never import `nkilib` / `neuronx_distributed` / `torch_neuronx` into model code beyond what already exists.
- **Kernel constraints (Trainium2+):** MHA (q_heads == kv_heads per rank), `head_dim ≤ 128` (Hunyuan 128, Flux 64, Qwen config), per-rank shard divisible by 128, non-causal (`use_causal_mask=False`).
- **Scale** must equal each model's existing scale: `1.0 / math.sqrt(head_dim)`.
- **LNC2 launch:** ring requires `NEURON_RT_VIRTUAL_CORE_SIZE=2` for the SPMD grid (same as Wan); ops select `ring_attention_spmd_fwd[2]` / `attention_cte[2]` when `vc_size == 2`.
- TDD: host-runnable tests run in normal `pytest`; device parity tests follow the env-var-gated NEFF pattern (`tests/numerical/test_wan_ring_attention_neff.py`).
- Commit after each task.

---

## File Structure

- `difflet/backends/trainium/ops_impl/attention.py` — add `joint_ring_attention` (Approach B) + `__all__`. `ring_attention` reused as-is for Flux. (Task 2)
- `difflet/backends/cpu/ops_impl/attention.py` — add `joint_ring_attention` CPU reference. (Task 2)
- `difflet/ops/attention.py`, `difflet/ops/__init__.py` — expose `joint_ring_attention`. (Task 2)
- `difflet/models/flux/modeling_flux.py` — `ring_attention` branch in the double-stream and single-stream CP paths; carry/thread `cp_mode`. (Task 1)
- `difflet/cli/orchestrators/flux.py` (+ config carrier) — thread `cp_mode`. (Task 1)
- `difflet/models/hunyuan_video/modeling_hunyuan_video.py` — `joint_ring_attention` adapter; thread `cp_mode`. (Task 4)
- `difflet/cli/orchestrators/{hunyuan_video,hunyuan_video_15}.py` (+ config carrier) — thread `cp_mode`. (Task 4)
- `difflet/backends/trainium/qwen_image/transformer.py` — `joint_ring_attention` adapter; thread `cp_mode`. (Task 5)
- `difflet/cli/orchestrators/qwen_image.py` (+ config carrier) — thread `cp_mode`. (Task 5)
- `tests/unit/test_joint_ring_attention_op.py` — op CPU-equivalence. (Task 2)
- `tests/unit/test_flux_cp_mode_threading.py`, `tests/unit/test_hunyuan_cp_mode_threading.py`, `tests/unit/test_qwen_cp_mode_threading.py` — threading. (Tasks 1/4/5)
- `tests/numerical/test_flux_ring_attention_neff.py`, `test_joint_ring_attention_neff.py`, `test_hunyuan_ring_attention_neff.py`, `test_qwen_ring_attention_neff.py` — device parity/spike. (Tasks 1/3/4/5)

---

## Task 1: Flux uniform-ring (reuse existing `ring_attention`)

Flux lands first: it needs no new op and no spike. Both Flux CP paths are uniform rings.

**Files:**
- Modify: `difflet/models/flux/modeling_flux.py` — double-stream CP branch (`:1139-1178`), single-stream CP branch (`:1206-1210`), and `cp_mode` carry (`NeuronFluxAttention.__init__`, `NeuronFluxTransformerBlock` / `NeuronFluxSingleTransformerBlock` ctor threading, `FluxBackboneInferenceConfig:1264`)
- Modify: `difflet/cli/orchestrators/flux.py` — pass `cp_mode` into `DiffletParallelConfig`; re-emit `--cp-mode`
- Test: `tests/unit/test_flux_cp_mode_threading.py`, `tests/numerical/test_flux_ring_attention_neff.py`

**Interfaces:**
- Consumes: `difflet.ops.ring_attention(q, k, v, *, scale, causal=False)` (exists, Wan increment); `DiffletParallelConfig.cp_mode` (exists).
- Produces: `NeuronFluxAttention` instances carry `self.cp_mode: str`; when `cp_mode == "ring"` both Flux CP paths call `ring_attention` instead of gather-KV.

- [ ] **Step 1: Write the failing threading test**

Create `tests/unit/test_flux_cp_mode_threading.py`:

```python
import os

os.environ["DIFFLET_BACKEND"] = "cpu"

from difflet.models.flux.modeling_flux import NeuronFluxAttention


def test_flux_attention_stores_cp_mode_default():
    attn = NeuronFluxAttention(query_dim=64, heads=2, dim_head=32)
    assert attn.cp_mode == "gather_kv"


def test_flux_attention_stores_cp_mode_ring():
    attn = NeuronFluxAttention(query_dim=64, heads=2, dim_head=32, cp_mode="ring")
    assert attn.cp_mode == "ring"
```

(If `NeuronFluxAttention.__init__` requires other mandatory args, pass the minimal set its signature already documents — the assertion is only about `cp_mode`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_flux_cp_mode_threading.py -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'cp_mode'`).

- [ ] **Step 3: Thread `cp_mode` through Flux**

Mirror exactly how `context_parallel_enabled` is already threaded in `modeling_flux.py`. At each site that passes/stores/reads `context_parallel_enabled`, add a sibling `cp_mode` (str, default `"gather_kv"`):

- `NeuronFluxAttention.__init__` (`:890-895`): add `cp_mode: str = "gather_kv"` to the signature; store `self.cp_mode = cp_mode`.
- `NeuronFluxTransformerBlock.__init__` (`:539-543`) and `NeuronFluxSingleTransformerBlock.__init__` (`:645-649`): add `cp_mode: str = "gather_kv"`; pass `cp_mode=cp_mode` to the `NeuronFluxAttention` they build (`:587`, `:666`).
- `NeuronFluxTransformer2DModel.__init__` (`:210`): read `self.cp_mode = getattr(self.config, 'cp_mode', 'gather_kv')` next to `context_parallel_enabled`; pass `cp_mode=self.cp_mode` to both block lists (`:249`, `:262`).
- `FluxBackboneInferenceConfig.__init__` (`:1264-1268`): accept `cp_mode: str = "gather_kv"` and store `self.cp_mode = cp_mode`.

- [ ] **Step 4: Run threading test to verify it passes**

Run: `pytest tests/unit/test_flux_cp_mode_threading.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the ring branch to both Flux CP paths**

In `modeling_flux.py`, import the op next to the existing `attention` import:

```python
from difflet.ops import ring_attention
```

**Double-stream path.** The gather-KV block at `:1139-1178` ends by building joint `query/key/value = cat([text, image], dim=2)`. Add a ring branch that bypasses the gather and rings the concatenated joint shard. Replace the `if self.context_parallel_enabled:` gather block (`:1139-1172`) + the joint concat (`:1176-1178`) with:

```python
            # CP joint attention. Flux shards BOTH streams, so [text ‖ image] is a
            # uniform per-rank joint shard: ring rotates it (cp_mode=ring) instead of
            # all-gathering K,V (cp_mode=gather_kv). Non-causal → equivalent.
            if self.context_parallel_enabled and self.cp_mode == "ring":
                if rotary_emb_text is not None:
                    encoder_hidden_states_query_proj = apply_rotary_emb(
                        encoder_hidden_states_query_proj, rotary_emb_text
                    )
                    encoder_hidden_states_key_proj = apply_rotary_emb(
                        encoder_hidden_states_key_proj, rotary_emb_text
                    )
                q_joint = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                k_joint = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                v_joint = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
                hidden_states = ring_attention(
                    q_joint, k_joint, v_joint, scale=1.0 / math.sqrt(head_dim), causal=False
                )
                query = key = value = None  # joint attention already computed
            else:
                if self.context_parallel_enabled:
                    if rotary_emb_text is not None:
                        encoder_hidden_states_query_proj = apply_rotary_emb(
                            encoder_hidden_states_query_proj, rotary_emb_text
                        )
                        encoder_hidden_states_key_proj = apply_rotary_emb(
                            encoder_hidden_states_key_proj, rotary_emb_text
                        )
                    stacked_kv = torch.stack([key, value], dim=0)
                    stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                        stacked_kv, gather_dim=3, process_group=self.data_parallel_group
                    )
                    key, value = torch.unbind(stacked_kv, dim=0)
                    stacked_kv_enc = torch.stack(
                        [encoder_hidden_states_key_proj, encoder_hidden_states_value_proj], dim=0
                    )
                    stacked_kv_enc = gather_from_tensor_model_parallel_region_with_dim(
                        stacked_kv_enc, gather_dim=3, process_group=self.data_parallel_group
                    )
                    encoder_hidden_states_key_proj, encoder_hidden_states_value_proj = torch.unbind(
                        stacked_kv_enc, dim=0
                    )
                query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
```

Then guard the downstream attention call (`:1218-1227`) so it is skipped when the ring branch already produced `hidden_states`:

```python
        if query is None:
            pass  # double-stream ring already set hidden_states
        elif attention_mask is not None or _HARDWARE == hardware.TRN1:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            if not self.context_parallel_enabled or encoder_hidden_states is not None:
                hidden_states = attention_wrapper_sharded_without_swap(query, key, value)
```

`ring_attention` returns `[B, H, Sq_joint, d]`, matching what the existing `hidden_states.transpose(1, 2).reshape(...)` at `:1229` expects, so the image/text output split at `:1232-1237` is unchanged.

**Single-stream path.** Replace the `else` (non-masked, non-TRN1) wrapper call (`:1206-1210`) with a ring branch:

```python
                elif self.cp_mode == "ring":
                    # Single sharded joint sequence → uniform ring (q/k/v same shard).
                    hidden_states = ring_attention(
                        query, key, value, scale=1.0 / math.sqrt(head_dim), causal=False
                    )
                else:
                    hidden_states = attention_wrapper_context_parallel_single_transformer(
                        query, key, value, self.data_parallel_group
                    )
```

`query/key/value` here are `[B, H, S/cp, d]` (key already transposed for the wrapper — `ring_attention` takes `tp_k=True` internally, matching). Confirm `value`'s layout in the spike test (Step 7); the single-stream wrapper passes `value` as `[S, B*H, d]`, so the adapter must hand `ring_attention` `value` as `[B, H, S/cp, d]` — reshape if the local `value` is not already in that form.

- [ ] **Step 6: Thread `cp_mode` through the Flux orchestrator**

In `difflet/cli/orchestrators/flux.py`, where it builds `DiffletParallelConfig(...)`, add `cp_mode=getattr(self.args, "cp_mode", "gather_kv")` (mirroring `difflet/cli/orchestrators/wan.py`). If Flux is a staged orchestrator, add `"--cp-mode", str(getattr(a, "cp_mode", "gather_kv"))` to its `_shared_cli_args` parts list. Ensure the `cp_mode` reaches `FluxBackboneInferenceConfig(..., cp_mode=parallel.cp_mode)` at the config build site.

- [ ] **Step 7: Write + run the device parity gate (gated, on Trn2 LNC2, cp_degree>1)**

Create `tests/numerical/test_flux_ring_attention_neff.py` mirroring `tests/numerical/test_wan_ring_attention_neff.py` (skip unless `DIFFLET_RUN_FLUX_RING_NEFF=1`; run the Flux backbone once with `cp_mode=gather_kv` and once with `cp_mode=ring` on identical sharded inputs; assert cosine ≥ 0.999). Provide a `scripts/flux_ring_parity_smoke.sh` analogous to `scripts/wan_ring_parity_smoke.sh` (`NEURON_RT_VIRTUAL_CORE_SIZE=2`, `NEURON_RT_NUM_CORES=tp*cp`).

Run host suite: `pytest tests/unit/test_flux_cp_mode_threading.py -v` → PASS.
Run collection: `pytest tests/numerical/test_flux_ring_attention_neff.py -v` → SKIPPED (gate unset).
Run on device: `DIFFLET_RUN_FLUX_RING_NEFF=1 bash scripts/flux_ring_parity_smoke.sh` → PASS, cosine ≥ 0.999.

- [ ] **Step 8: Commit**

```bash
git add difflet/models/flux/modeling_flux.py difflet/cli/orchestrators/flux.py \
        tests/unit/test_flux_cp_mode_threading.py \
        tests/numerical/test_flux_ring_attention_neff.py scripts/flux_ring_parity_smoke.sh
git commit -m "feat(flux): cp_mode=ring uniform-ring reuse for both CP attention paths"
```

---

## Task 2: Shared `joint_ring_attention` op (Trainium B + CPU reference)

**Files:**
- Modify: `difflet/backends/trainium/ops_impl/attention.py` (add `joint_ring_attention` + extend `__all__`)
- Modify: `difflet/backends/cpu/ops_impl/attention.py` (add CPU reference + extend `__all__`)
- Modify: `difflet/ops/attention.py` (public passthrough), `difflet/ops/__init__.py` (dispatch map)
- Test: `tests/unit/test_joint_ring_attention_op.py`

**Interfaces:**
- Consumes: `ring_attention_spmd_fwd` (guarded import, exists), `attention_cte` (exists).
- Produces: `difflet.ops.joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False)`. Inputs `[B, H, S_*, d]`; returns `[B, H, S_q, d]` where `S_q == q.shape[2]`. CPU reference computes plain full joint attention over `cat([image_k, text_k], dim=2)`.

- [ ] **Step 1: Write the failing test (CPU equivalence — the merge oracle)**

Create `tests/unit/test_joint_ring_attention_op.py`:

```python
import math
import os

import torch


def test_joint_ring_attention_cpu_matches_full_joint_attention():
    os.environ["DIFFLET_BACKEND"] = "cpu"
    from difflet.ops import attention, joint_ring_attention

    torch.manual_seed(0)
    b, h, s_img, s_txt, d = 1, 2, 128, 64, 64
    q = torch.randn(b, h, s_img + s_txt, d)
    image_k = torch.randn(b, h, s_img, d)
    image_v = torch.randn(b, h, s_img, d)
    text_k = torch.randn(b, h, s_txt, d)
    text_v = torch.randn(b, h, s_txt, d)
    scale = 1.0 / math.sqrt(d)

    full_k = torch.cat([image_k, text_k], dim=2)
    full_v = torch.cat([image_v, text_v], dim=2)
    ref = attention(
        q.reshape(b * h, s_img + s_txt, d),
        full_k.reshape(b * h, s_img + s_txt, d),
        full_v.reshape(b * h, s_img + s_txt, d),
        scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False,
    ).reshape(b, h, s_img + s_txt, d)

    out = joint_ring_attention(q, image_k, image_v, text_k, text_v, scale=scale, causal=False)
    assert out.shape == (b, h, s_img + s_txt, d)
    assert torch.allclose(ref.float(), out.float(), atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_joint_ring_attention_op.py -v`
Expected: FAIL (`ImportError: cannot import name 'joint_ring_attention'`).

- [ ] **Step 3: Expose the op in `difflet.ops`**

In `difflet/ops/__init__.py`, add to `_EXPORTS` next to `ring_attention`:

```python
    "joint_ring_attention": ("attention", "joint_ring_attention"),
```

In `difflet/ops/attention.py`, add after `ring_attention`:

```python
def joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False):
    """Joint-MMDiT ring self-attention for text-replicated models.

    Rings the sequence-sharded image K,V and merges a replicated text partial via
    online softmax. ``q`` is this rank's joint local query
    ``[B, H, S_img/cp + S_txt, d]``; ``image_*`` are the sharded image K,V;
    ``text_*`` are the replicated text K,V. Returns ``[B, H, q.shape[2], d]``.
    """

    return _load("joint_ring_attention")(
        q, image_k, image_v, text_k, text_v, scale=scale, causal=causal
    )
```

- [ ] **Step 4: Write the CPU reference**

In `difflet/backends/cpu/ops_impl/attention.py`, add (and append `"joint_ring_attention"` to `__all__`):

```python
def joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False):
    # cp_degree == 1 reference: the ring degenerates to plain full joint attention
    # over the concatenated [image, text] keys. This is the merge-math oracle.
    full_k = torch.cat([image_k, text_k], dim=2)
    full_v = torch.cat([image_v, text_v], dim=2)
    b, h, s_q, d = q.shape
    s_k = full_k.shape[2]
    out = attention(
        q.reshape(b * h, s_q, d),
        full_k.reshape(b * h, s_k, d),
        full_v.reshape(b * h, s_k, d),
        scale=scale, causal=causal, tp_q=True, tp_k=True, tp_out=False,
    )
    return out.reshape(b, h, s_q, d)
```

- [ ] **Step 5: Write the Trainium implementation (Approach B)**

In `difflet/backends/trainium/ops_impl/attention.py`, add `joint_ring_attention` and extend `__all__` to include it. The structure (the LSE-merge reshape between the two kernels' stat layouts is the spike's deliverable in Task 3 — see the marked block):

```python
def joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale: float, causal: bool = False):
    """Joint-MMDiT ring self-attention: ring the sharded image K,V (via
    ring_attention_spmd_fwd, training=True for its LSE) and merge a replicated text
    partial (attention_cte, cache_softmax) by online softmax. Lossless vs gather-KV.

    q                [B, H, S_img/cp + S_txt, d]   this rank's joint local queries
    image_k,image_v  [B, H, S_img/cp, d]           sharded — rotated by the ring
    text_k, text_v   [B, H, S_txt,    d]           replicated — local partial
    returns          [B, H, S_img/cp + S_txt, d]
    """
    if ring_attention_spmd_fwd is None:
        raise RuntimeError(
            "joint ring attention requires nkilib.experimental.attention.ring_attention_fwd "
            f"(import failed: {_RING_IMPORT_ERROR!r}). Upgrade neuronx-cc / nkilib, or "
            "use cp_mode=gather_kv."
        )
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_group,
        get_data_parallel_size,
    )

    mesh = get_data_parallel_group(as_list=True)
    num_workers = get_data_parallel_size()
    replica_groups = tuple(tuple(int(r) for r in grp) for grp in mesh)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    ring_kernel = ring_attention_spmd_fwd[2] if vc_size == 2 else ring_attention_spmd_fwd
    cte_kernel = attention_cte[2] if vc_size == 2 else attention_cte

    # Image partial: every local query (image + text) attends ALL image keys via the
    # ring. training=True returns lse (used only for the merge; no backward).
    o_image, lse_image = ring_kernel(
        q, image_k, image_v,
        replica_groups=replica_groups, num_workers=num_workers,
        softmax_scale=float(scale), use_causal_mask=causal,
        training=True, tp_q=True, tp_k=True,
    )
    # Text partial: replicated text K,V, rank-local (no collective).
    o_text, neg_max_text, sum_text = cte_kernel(
        q, text_k, text_v,
        scale=float(scale), causal_mask=causal,
        tp_q=True, tp_k=True, tp_out=False,
        cache_softmax=True,
    )

    # --- ONLINE-SOFTMAX MERGE OF TWO NORMALIZED PARTIALS (spike-validated, Task 3) ---
    # Math (per query row): with lse_A/lse_B the log-sum-exp of each partial and
    # o_A/o_B the NORMALIZED partial outputs,
    #     m   = max(lse_image, lse_text)
    #     wA  = exp(lse_image - m); wB = exp(lse_text - m)
    #     o   = (wA*o_image + wB*o_text) / (wA + wB)
    # lse_text is derived from attention_cte stats: lse_text = -neg_max_text + log(sum_text).
    # The ONLY device-specific unknown is reshaping ring's lse_image
    # [B,H,128,Sq/128] and attention_cte's [bs,128,num_grps] stats into per-query
    # tensors broadcastable against o_* [B,H,Sq,d]; Task 3's spike pins the exact
    # reshape, after which this block is concrete.
    return _merge_joint_partials(o_image, lse_image, o_text, neg_max_text, sum_text)
```

Add a `_merge_joint_partials(...)` helper implementing the formula above once the spike (Task 3) fixes the stat reshape. If the spike rejects Approach B, replace the body with Approach A (hand-rolled `collective_permute` ring over `attention_cte` with `text_k/text_v` as `k_prior`/`v_prior`, `cache_softmax=True, skip_output_normalization=True`); the public signature and the CPU reference are unchanged.

- [ ] **Step 6: Run test to verify it passes (host)**

Run: `pytest tests/unit/test_joint_ring_attention_op.py -v`
Expected: PASS (1 passed) — the CPU reference path, independent of the device merge.

- [ ] **Step 7: Commit**

```bash
git add difflet/ops/__init__.py difflet/ops/attention.py \
        difflet/backends/trainium/ops_impl/attention.py \
        difflet/backends/cpu/ops_impl/attention.py \
        tests/unit/test_joint_ring_attention_op.py
git commit -m "feat(ops): add joint_ring_attention op (trainium image-ring + text LSE merge; cpu ref)"
```

---

## Task 3: Device spike — `joint_ring_attention` parity (Hunyuan/Qwen go/no-go)

Retires the joint-Q-seqlen + inference-LSE risk on synthetic input before wiring any text-replicated model. Gated; runs on Trn2 LNC2 with cp_degree > 1.

**Files:**
- Test: `tests/numerical/test_joint_ring_attention_neff.py`
- Possibly modify: `difflet/backends/trainium/ops_impl/attention.py` (`_merge_joint_partials` reshape, or fall back to Approach A)

**Interfaces:**
- Consumes: `difflet.ops.joint_ring_attention` (Task 2).
- Produces: confirmed device-correct `joint_ring_attention` (cosine ≥ 0.999 vs a gather-KV reference on the same sharded input), OR a committed Approach-A body if B is rejected.

- [ ] **Step 1: Write the gated parity spike**

Create `tests/numerical/test_joint_ring_attention_neff.py` (skip unless `DIFFLET_RUN_JOINT_RING_NEFF=1`). Construct a synthetic joint input where image is sharded (`S_img/cp`) and text is replicated (`S_txt`), `S_q = S_img/cp + S_txt`, per-rank image shard a multiple of 128, head_dim ≤ 128. Compute the reference by all-gathering image K,V and running one full joint `attention`; compute the candidate via `joint_ring_attention`; assert cosine ≥ 0.999. Provide `scripts/joint_ring_spike.sh` mirroring `scripts/wan_ring_parity_smoke.sh` (`NEURON_RT_VIRTUAL_CORE_SIZE=2`).

- [ ] **Step 2: Run to verify it collects/skips**

Run: `pytest tests/numerical/test_joint_ring_attention_neff.py -v`
Expected: SKIPPED (gate unset).

- [ ] **Step 3: Run on device; resolve the merge or fall back**

Run: `DIFFLET_RUN_JOINT_RING_NEFF=1 bash scripts/joint_ring_spike.sh`
- If it passes: `_merge_joint_partials` reshape is correct — Approach B confirmed.
- If `ring_attention_spmd_fwd` rejects `Sq ≠ Sk_shard` or `training=True` LSE is unusable: implement Approach A in `joint_ring_attention` (hand-rolled ring with text `k_prior`/`v_prior`), rerun until cosine ≥ 0.999.
Expected: PASS, cosine ≥ 0.999.

- [ ] **Step 4: Commit**

```bash
git add tests/numerical/test_joint_ring_attention_neff.py scripts/joint_ring_spike.sh \
        difflet/backends/trainium/ops_impl/attention.py
git commit -m "test(ops): device parity spike for joint_ring_attention (confirm Approach B / fall back to A)"
```

---

## Task 4: HunyuanVideo `joint_ring_attention` adapter + threading

**Files:**
- Modify: `difflet/models/hunyuan_video/modeling_hunyuan_video.py` — `HunyuanVideoAttention.forward` (`:393-413`), `cp_mode` carry (`HunyuanVideoAttention.__init__`, `HunyuanVideoTransformerBlock`, model ctor, backbone config)
- Modify: `difflet/cli/orchestrators/hunyuan_video.py`, `difflet/cli/orchestrators/hunyuan_video_15.py` — thread `cp_mode`
- Test: `tests/unit/test_hunyuan_cp_mode_threading.py`, `tests/numerical/test_hunyuan_ring_attention_neff.py`

**Interfaces:**
- Consumes: `difflet.ops.joint_ring_attention` (Task 2/3); `DiffletParallelConfig.cp_mode`.
- Produces: `HunyuanVideoAttention` carries `self.cp_mode`; ring branch selected when `context_parallel_enabled and not self.is_cross_attention and cp_mode == "ring"`.

- [ ] **Step 1: Write the failing threading test**

Create `tests/unit/test_hunyuan_cp_mode_threading.py`:

```python
import os

os.environ["DIFFLET_BACKEND"] = "cpu"

from difflet.models.hunyuan_video.modeling_hunyuan_video import (
    HunyuanVideoAttention,
    HunyuanVideoTransformerBlock,
)


def test_hunyuan_attention_stores_cp_mode_default():
    attn = HunyuanVideoAttention(dim=128, heads=4, head_dim=32)
    assert attn.cp_mode == "gather_kv"


def test_hunyuan_block_threads_cp_mode_to_attention():
    block = HunyuanVideoTransformerBlock(
        num_attention_heads=4, attention_head_dim=32, mlp_ratio=4.0, cp_mode="ring"
    )
    assert block.attn.cp_mode == "ring"
```

(Adjust the constructor kwargs to the real minimal signatures; the assertions are only about `cp_mode`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_hunyuan_cp_mode_threading.py -v`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'cp_mode'`).

- [ ] **Step 3: Thread `cp_mode` (mirror `context_parallel_enabled`)**

In `modeling_hunyuan_video.py`, at each site passing/storing/reading `context_parallel_enabled`, add a sibling `cp_mode: str = "gather_kv"`: `HunyuanVideoAttention.__init__` (store `self.cp_mode`), `HunyuanVideoTransformerBlock.__init__` (pass to its attention), the model `__init__` (read `getattr(config, 'cp_mode', 'gather_kv')`, pass to blocks), and the backbone `InferenceConfig` carrier (accept + store `cp_mode`, mirroring its `context_parallel_enabled` guard).

- [ ] **Step 4: Run threading test to verify it passes**

Run: `pytest tests/unit/test_hunyuan_cp_mode_threading.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the ring branch in `HunyuanVideoAttention.forward`**

Import the op near the top: `from difflet.ops import joint_ring_attention` (alongside the existing attention import).

Replace the CP gather block (`:397-402`) and the `dual_stream_attention` call (`:404-413`) with a `cp_mode` branch. `latent_q/k/v` are `[B, S, H, d]` and `context_q/k/v` are `[B, S_txt, H, d]`; the op wants `[B, H, S, d]`, so transpose in and out:

```python
        if self.context_parallel_enabled and self.cp_mode == "ring":
            # Joint ring: image (latent) K,V sharded → rotated; text (context) K,V
            # replicated → local partial. q is the joint [image_shard ‖ text].
            scale = 1.0 / math.sqrt(self.head_dim)
            q_img = latent_q.transpose(1, 2)      # [B, H, S_img/cp, d]
            q_txt = context_q.transpose(1, 2)     # [B, H, S_txt, d]
            q_joint = torch.cat([q_img, q_txt], dim=2)  # match dual_stream concat order
            o_joint = joint_ring_attention(
                q_joint,
                latent_k.transpose(1, 2), latent_v.transpose(1, 2),
                context_k.transpose(1, 2), context_v.transpose(1, 2),
                scale=scale, causal=False,
            )  # [B, H, S_img/cp + S_txt, d]
            o_joint = o_joint.transpose(1, 2)     # [B, S, H, d]
            img_len = latent_q.shape[1]
            hidden_states = o_joint[:, :img_len]
            encoder_hidden_states = o_joint[:, img_len:]
        else:
            if self.context_parallel_enabled:
                stacked_kv = torch.stack([latent_k, latent_v], dim=0)  # [2, B, S/cp, H, d]
                stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                    stacked_kv, gather_dim=2, process_group=self.data_parallel_group
                )  # [2, B, S, H, d]
                latent_k, latent_v = torch.unbind(stacked_kv, dim=0)
            hidden_states, encoder_hidden_states = dual_stream_attention(
                latent_q, latent_k, latent_v, context_q, context_k, context_v,
                attention_mask=attention_mask, scale=1.0 / math.sqrt(self.head_dim),
            )
```

Confirm `dual_stream_attention`'s output ordering (image vs context) and per-head shape, and match `q_joint`'s concat order to it so the `[:img_len]/[img_len:]` split is correct. The subsequent `hidden_states.flatten(2, 3)` (`:414-415`) is unchanged (`o_joint` slices are `[B, S, H, d]`).

- [ ] **Step 6: Thread `cp_mode` through both orchestrators**

In `difflet/cli/orchestrators/hunyuan_video.py` and `hunyuan_video_15.py`, add `cp_mode=getattr(self.args, "cp_mode", "gather_kv")` at the `DiffletParallelConfig(...)` build, `"--cp-mode", str(getattr(a, "cp_mode", "gather_kv"))` in `_shared_cli_args`, and `cp_mode=parallel.cp_mode` into the backbone config build (mirroring `wan.py`).

- [ ] **Step 7: Write + run the device parity gate**

Create `tests/numerical/test_hunyuan_ring_attention_neff.py` + `scripts/hunyuan_ring_parity_smoke.sh` mirroring the Wan parity harness (skip unless `DIFFLET_RUN_HUNYUAN_RING_NEFF=1`).

Run host: `pytest tests/unit/test_hunyuan_cp_mode_threading.py -v` → PASS.
Run collection: `pytest tests/numerical/test_hunyuan_ring_attention_neff.py -v` → SKIPPED.
Run device: `DIFFLET_RUN_HUNYUAN_RING_NEFF=1 bash scripts/hunyuan_ring_parity_smoke.sh` → PASS, cosine ≥ 0.999.

- [ ] **Step 8: Commit**

```bash
git add difflet/models/hunyuan_video/modeling_hunyuan_video.py \
        difflet/cli/orchestrators/hunyuan_video.py difflet/cli/orchestrators/hunyuan_video_15.py \
        tests/unit/test_hunyuan_cp_mode_threading.py \
        tests/numerical/test_hunyuan_ring_attention_neff.py scripts/hunyuan_ring_parity_smoke.sh
git commit -m "feat(hunyuan): cp_mode=ring joint-ring self-attention adapter"
```

---

## Task 5: Qwen-Image `joint_ring_attention` adapter + threading

**Files:**
- Modify: `difflet/backends/trainium/qwen_image/transformer.py` — `_QwenImageTrainiumAttnProcessor.__call__` (`:363-413`), `cp_mode` carry (`__init__:307-314`, processor construction, transformer module, config)
- Modify: `difflet/cli/orchestrators/qwen_image.py` — thread `cp_mode`
- Test: `tests/unit/test_qwen_cp_mode_threading.py`, `tests/numerical/test_qwen_ring_attention_neff.py`

**Interfaces:**
- Consumes: `difflet.ops.joint_ring_attention`; `DiffletParallelConfig.cp_mode`.
- Produces: `_QwenImageTrainiumAttnProcessor` carries `self.cp_mode`; ring branch when `context_parallel_enabled and cp_mode == "ring"`.

- [ ] **Step 1: Write the failing threading test**

Create `tests/unit/test_qwen_cp_mode_threading.py`:

```python
import os

os.environ["DIFFLET_BACKEND"] = "cpu"

from difflet.backends.trainium.qwen_image.transformer import _QwenImageTrainiumAttnProcessor


def test_qwen_processor_stores_cp_mode_default():
    proc = _QwenImageTrainiumAttnProcessor()
    assert proc.cp_mode == "gather_kv"


def test_qwen_processor_stores_cp_mode_ring():
    proc = _QwenImageTrainiumAttnProcessor(context_parallel_enabled=True, cp_mode="ring")
    assert proc.cp_mode == "ring"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_qwen_cp_mode_threading.py -v`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'cp_mode'`).

- [ ] **Step 3: Thread `cp_mode`**

In `transformer.py`, add `cp_mode: str = "gather_kv"` to `_QwenImageTrainiumAttnProcessor.__init__` (`:307-314`) and store `self.cp_mode = cp_mode`. At the site that constructs the processor (passing `context_parallel_enabled`/`data_parallel_group`), pass `cp_mode=...`. Thread `cp_mode` from the Qwen transformer module / its `InferenceConfig` carrier alongside `context_parallel_enabled`.

- [ ] **Step 4: Run threading test to verify it passes**

Run: `pytest tests/unit/test_qwen_cp_mode_threading.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Add the ring branch in `_QwenImageTrainiumAttnProcessor.__call__`**

Import the op: `from difflet.ops import joint_ring_attention` (alongside `attention`).

Replace the CP gather block (`:367-372`) + the joint concat + unmasked attention (`:374-402`) with a `cp_mode` branch. `img_*`/`txt_*` are `[B, S, H, d]`; the op wants `[B, H, S, d]`, and the joint concat order is `[txt, img]`:

```python
        head_dim = txt_query.shape[-1]
        if self.context_parallel_enabled and self.cp_mode == "ring":
            scale = 1.0 / math.sqrt(head_dim)
            q_joint = torch.cat([txt_query, img_query], dim=1).transpose(1, 2)  # [B,H,Sq,d]
            o_joint = joint_ring_attention(
                q_joint,
                img_key.transpose(1, 2), img_value.transpose(1, 2),     # sharded image
                txt_key.transpose(1, 2), txt_value.transpose(1, 2),     # replicated text
                scale=scale, causal=False,
            )  # [B, H, Sq, d]
            joint_hidden_states = o_joint.transpose(1, 2)  # [B, Sq, H, d]
        else:
            if self.context_parallel_enabled:
                stacked_kv = torch.stack([img_key, img_value], dim=0)  # [2, B, S/cp, H, d]
                stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                    stacked_kv, gather_dim=2, process_group=self.data_parallel_group
                )  # [2, B, S, H, d]
                img_key, img_value = torch.unbind(stacked_kv, dim=0)
            joint_query = torch.cat([txt_query, img_query], dim=1)
            joint_key = torch.cat([txt_key, img_key], dim=1)
            joint_value = torch.cat([txt_value, img_value], dim=1)
            if attention_mask is None:
                q = joint_query.transpose(1, 2)
                k = joint_key.transpose(1, 2)
                v = joint_value.transpose(1, 2)
                b, h, s_q, d = q.shape
                s_k = k.shape[2]
                attn_out = attention(
                    q.reshape(b * h, s_q, d), k.reshape(b * h, s_k, d), v.reshape(b * h, s_k, d),
                    scale=1.0 / math.sqrt(head_dim), causal=False, attention_mask=None,
                    tp_q=True, tp_k=True, tp_out=False,
                )
                joint_hidden_states = attn_out.reshape(b, h, s_q, d).transpose(1, 2)
            else:
                joint_hidden_states = dispatch_attention_fn(
                    joint_query, joint_key, joint_value, attn_mask=attention_mask,
                    dropout_p=0.0, is_causal=False, backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
```

The downstream `joint_hidden_states.flatten(2, 3)` and `[:seq_txt]/[seq_txt:]` split (`:415-419`) are unchanged because `joint_hidden_states` is `[B, Sq, H, d]` with the same `[txt, img]` order.

- [ ] **Step 6: Thread `cp_mode` through the orchestrator**

In `difflet/cli/orchestrators/qwen_image.py`: add `cp_mode=getattr(self.args, "cp_mode", "gather_kv")` to the `DiffletParallelConfig(...)` build, `"--cp-mode", str(getattr(a, "cp_mode", "gather_kv"))` to `_shared_cli_args` if staged, and `cp_mode` into the transformer config build (mirroring `wan.py`).

- [ ] **Step 7: Write + run the device parity gate**

Create `tests/numerical/test_qwen_ring_attention_neff.py` + `scripts/qwen_ring_parity_smoke.sh` (skip unless `DIFFLET_RUN_QWEN_RING_NEFF=1`).

Run host: `pytest tests/unit/test_qwen_cp_mode_threading.py -v` → PASS.
Run collection: `pytest tests/numerical/test_qwen_ring_attention_neff.py -v` → SKIPPED.
Run device: `DIFFLET_RUN_QWEN_RING_NEFF=1 bash scripts/qwen_ring_parity_smoke.sh` → PASS, cosine ≥ 0.999.

- [ ] **Step 8: Commit**

```bash
git add difflet/backends/trainium/qwen_image/transformer.py difflet/cli/orchestrators/qwen_image.py \
        tests/unit/test_qwen_cp_mode_threading.py \
        tests/numerical/test_qwen_ring_attention_neff.py scripts/qwen_ring_parity_smoke.sh
git commit -m "feat(qwen): cp_mode=ring joint-ring self-attention adapter"
```

---

## Task 6: End-to-end trajectory parity + README

**Files:**
- Test: `tests/numerical/test_hunyuan_ring_attention_neff.py` (add an e2e case) or a sibling
- Modify: `README.md` (context-parallelism status)

**Interfaces:**
- Consumes: the full Task 4 HunyuanVideo ring path.

- [ ] **Step 1: Write the gated e2e test**

Add a device-gated test that runs a short HunyuanVideo denoise (few steps) with `cp_degree>1, cp_mode=ring` and compares the latent trajectory against the validated `cp_mode=gather_kv` run (cosine ≥ 0.999), gated by `DIFFLET_RUN_HUNYUAN_RING_E2E=1`, mirroring the Wan e2e pattern.

- [ ] **Step 2: Run to verify it collects/skips**

Run: `pytest tests/numerical/test_hunyuan_ring_attention_neff.py -k e2e -v`
Expected: SKIPPED without the env gate.

- [ ] **Step 3: Run the e2e gate on device**

Run: `DIFFLET_RUN_HUNYUAN_RING_E2E=1 pytest tests/numerical/test_hunyuan_ring_attention_neff.py -k e2e -v`
Expected: PASS, cosine ≥ 0.999 vs gather-KV trajectory.

- [ ] **Step 4: Update README**

In `README.md`, extend the context-parallelism status to note ring is now available for Flux, HunyuanVideo (+1.5), and Qwen-Image via `--cp-mode ring` (gather-KV remains the default), noting Flux reuses the uniform ring and Hunyuan/Qwen use the joint-ring text merge.

- [ ] **Step 5: Commit**

```bash
git add tests/numerical/test_hunyuan_ring_attention_neff.py README.md
git commit -m "test(hunyuan): e2e ring vs gather-KV trajectory parity; document --cp-mode ring for joint-MMDiT"
```

---

## Task 7: Finalize — squash all commits into one and push to `origin/cp-ring`

Run ONLY after Tasks 1-6 are complete and all gates (host unit suite green; device parity + spike + e2e green on hardware) have passed.

**Files:** none (git history only).

- [ ] **Step 1: Verify branch and clean tree**

```bash
git rev-parse --abbrev-ref HEAD   # expect: cp-ring
git status --porcelain            # expect: empty
```

- [ ] **Step 2: Review the commits that will be squashed**

```bash
git log --oneline main..HEAD
```
Expected: the Task 1-6 commits plus the spec/plan doc commits.

- [ ] **Step 3: Squash all commits since `main` into one (soft reset)**

```bash
git reset --soft "$(git merge-base main HEAD)"
git commit -m "feat(cp): opt-in ring-attention context parallelism for joint-MMDiT models

Extend cp_mode=ring to Flux, HunyuanVideo (+1.5), and Qwen-Image. Flux reuses
the uniform ring_attention op (both streams sharded). HunyuanVideo/Qwen-Image use
a new joint_ring_attention op: ring the sharded image K,V and merge a replicated
text partial by online softmax. Gather-KV unchanged as the default; parity vs
gather-KV cosine >= 0.999 per model.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 4: Verify the squash preserved the tree**

```bash
git diff --stat main..HEAD
```
Expected: the full set of changed files from Tasks 1-6.

- [ ] **Step 5: Push to the remote dev branch**

```bash
git push -u origin cp-ring
```
(If history diverged, re-push with `--force-with-lease` only after confirming `origin/cp-ring` holds no other work.)

---

## Self-Review

**Spec coverage:**
- Two-case split (Flux uniform-ring vs Hunyuan/Qwen joint-ring) → Task 1 (Flux), Tasks 2-5 (joint). ✓
- Shared `joint_ring_attention` op (B impl + CPU ref + expose) → Task 2. ✓
- Risk retired first (joint-Q seqlen + inference LSE) → Task 3 spike, gating Tasks 4-5. ✓
- Flux reuses existing `ring_attention`, both block types → Task 1 Step 5. ✓
- `cp_mode` threading per model (config + orchestrator + attention) → Tasks 1/4/5 Steps 3/6. ✓
- Parity-only validation (host CPU equivalence, per-model device parity, one e2e) → Tasks 2/1/4/5/6. ✓
- Out of scope (padding/perf/LTX-2/cross-attention) → enforced by Global Constraints. ✓
- Finalize squash + push → Task 7. ✓

**Placeholder scan:** The Trainium merge body (Task 2 Step 5) and the device parity/e2e harness construction (Tasks 1/3/4/5/6) carry deliberate device-harness deferrals — the LSE-stat reshape is environment-specific and resolved by the Task 3 spike; the CP-launch parity scripts mirror the existing `scripts/wan_ring_parity_smoke.sh` and cannot be hard-coded without the target topology. These are clearly marked, mirror the Wan increment's precedent, and every host-runnable step (threading tests, CPU-equivalence) contains complete code. No `TODO`/`TBD` placeholders remain.

**Type consistency:** `joint_ring_attention(q, image_k, image_v, text_k, text_v, *, scale, causal=False) -> [B,H,Sq,d]` identical across `difflet.ops`, trainium impl, cpu ref, and all three call sites. `cp_mode: str` ∈ {`"gather_kv"`,`"ring"`} used identically in config, threading, and every `forward` branch. `ring_attention(q, k, v, *, scale, causal=False)` reused unchanged for Flux.
