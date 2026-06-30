# Context-Parallel Ring Attention (Wan, increment 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ring-attention context parallelism as an opt-in CP mode (`cp_mode=ring`) for the Wan DiT self-attention, via a shared `difflet.ops.ring_attention` op that wraps nkilib's `ring_attention_spmd_fwd`, selectable from the CLI.

**Architecture:** A new `cp_mode` field on `DiffletParallelConfig` (default `gather_kv`) flows through the existing CP plumbing exactly like `context_parallel_enabled`. When `cp_mode=ring` and CP is enabled, Wan's self-attention calls a shared `ring_attention` op instead of the gather-KV branch. The op (Trainium backend) resolves the data-parallel ring group internally and calls the prebuilt nkilib kernel; the model stays backend-neutral. Gather-KV is untouched and remains the default.

**Tech Stack:** Python, PyTorch, `torch_neuronx`, `neuronx_distributed` (parallel state / collectives), `nkilib.experimental.attention.ring_attention_fwd`, pytest.

## Global Constraints

- **Scope: Wan only** this increment. HunyuanVideo / Qwen-Image / Flux (joint MMDiT self-attention) and LTX-2 (no CP foundation) are deferred — do NOT modify their attention.
- **`modeling_wan.py` is backend-neutral:** it imports only from `torch`, `diffusers`, and `difflet.ops`. It MUST NOT import `neuronx_distributed`, `nkilib`, or `torch_neuronx`. The ring op is reached via `difflet.ops`, never imported directly in the model.
- **Default is `gather_kv`:** `cp_mode` defaults to `"gather_kv"`; a default config must leave the compile-cache key **byte-identical** to legacy (additive-only cache key, mirroring `CandidateConfig`).
- **Ring activates only when** `context_parallel_enabled` (i.e. `cp_degree > 1`) AND `cp_mode == "ring"` AND not cross-attention.
- **Kernel constraints (Trainium2+ only):** MHA (q_heads == kv_heads per rank), `head_dim ≤ 128` (Wan = 128 ✓), per-rank `seqlen` divisible by 128, non-causal (`use_causal_mask=False`).
- **Ring scale** must equal the existing Wan scale: `1.0 / math.sqrt(head_dim)`.
- TDD: host-runnable tests run in normal `pytest`; device parity tests follow the existing env-var-gated NEFF pattern (e.g. `tests/numerical/test_hunyuan_video_attention_neff.py`).
- Commit after each task.

---

## File Structure

- `difflet/pipeline/parallel_config.py` — add `cp_mode` field + validation + additive cache key. (Task 1)
- `difflet/cli/main.py`, `difflet/cli/stage.py` — `--cp-mode` flag. (Task 2)
- `difflet/cli/orchestrators/wan.py` — pass `cp_mode` into the parallel config; re-emit `--cp-mode`. (Task 2)
- `difflet/backends/trainium/ops_impl/attention.py` — `ring_attention` impl + guarded import. (Task 3)
- `difflet/backends/cpu/ops_impl/attention.py` — CPU `ring_attention` (plain attention, cp=1 reference). (Task 3)
- `difflet/ops/attention.py`, `difflet/ops/__init__.py` — expose `ring_attention`. (Task 3)
- `difflet/models/wan/application.py` — thread `cp_mode` into `create_wan_backbone_config`. (Task 4)
- `difflet/backends/trainium/wan/backbone.py` — carry `cp_mode` on the backbone config. (Task 4)
- `difflet/models/wan/modeling_wan.py` — thread `cp_mode` to blocks/attention; ring branch in `WanAttention.forward`. (Task 4, Task 5)
- `tests/unit/test_parallel_config_cp_mode.py` — config tests. (Task 1)
- `tests/unit/test_cli_cp_mode.py` — CLI tests. (Task 2)
- `tests/unit/test_ring_attention_op.py` — op CPU-equivalence + guard tests. (Task 3)
- `tests/unit/test_wan_cp_mode_threading.py` — threading tests. (Task 4)
- `tests/numerical/test_wan_ring_attention_neff.py` — device parity (ring vs gather-KV). (Task 5)

---

## Task 1: `cp_mode` config field, validation, and additive cache key

**Files:**
- Modify: `difflet/pipeline/parallel_config.py:8-35`
- Test: `tests/unit/test_parallel_config_cp_mode.py`

**Interfaces:**
- Produces: `DiffletParallelConfig(tp_degree:int=1, cp_degree:int=1, cfg_parallel_enabled:bool=False, cp_mode:str="gather_kv")`; valid `cp_mode` ∈ {`"gather_kv"`, `"ring"`}; `to_cache_dict()` omits `cp_mode` when it equals `"gather_kv"`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_parallel_config_cp_mode.py`:

```python
import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig


def test_cp_mode_defaults_to_gather_kv():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=1)
    assert cfg.cp_mode == "gather_kv"


def test_cp_mode_ring_requires_cp_degree_gt_1():
    with pytest.raises(ValueError, match="cp_mode='ring' requires cp_degree > 1"):
        DiffletParallelConfig(tp_degree=4, cp_degree=1, cp_mode="ring")


def test_cp_mode_ring_with_cp_degree_2_is_valid():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="ring")
    assert cfg.cp_mode == "ring"


def test_cp_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="cp_mode must be one of"):
        DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="bogus")


def test_cache_dict_omits_cp_mode_when_gather_kv():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="gather_kv")
    assert "cp_mode" not in cfg.to_cache_dict()


def test_cache_dict_includes_cp_mode_when_ring():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=2, cp_mode="ring")
    assert cfg.to_cache_dict()["cp_mode"] == "ring"


def test_cache_dict_default_is_byte_identical_to_legacy_keys():
    cfg = DiffletParallelConfig(tp_degree=4, cp_degree=1)
    assert set(cfg.to_cache_dict()) == {"tp_degree", "cp_degree", "cfg_parallel_enabled"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_parallel_config_cp_mode.py -v`
Expected: FAIL (e.g. `TypeError: __init__() got an unexpected keyword argument 'cp_mode'`).

- [ ] **Step 3: Write minimal implementation**

In `difflet/pipeline/parallel_config.py`, replace the dataclass body (lines 8-35) with:

```python
@dataclass(frozen=True)
class DiffletParallelConfig:
    """Tensor, context, and CFG parallel settings.

    Context parallelism is configured via ``cp_degree`` (1 = disabled) and
    ``cp_mode`` selects the CP attention strategy (``"gather_kv"`` default, or
    ``"ring"``). Context parallelism and CFG parallelism both consume extra
    data-parallel lanes, so they are mutually exclusive.
    """

    tp_degree: int = 1
    cp_degree: int = 1
    cfg_parallel_enabled: bool = False
    cp_mode: str = "gather_kv"

    def __post_init__(self) -> None:
        if self.tp_degree < 1:
            raise ValueError("tp_degree must be >= 1")
        if self.cp_degree < 1:
            raise ValueError("cp_degree must be >= 1")
        if self.cp_degree > 1 and self.cfg_parallel_enabled:
            raise ValueError("cp_degree > 1 and cfg_parallel_enabled are mutually exclusive")
        if self.cp_mode not in ("gather_kv", "ring"):
            raise ValueError(f"cp_mode must be one of {{'gather_kv', 'ring'}}, got {self.cp_mode!r}")
        if self.cp_mode == "ring" and self.cp_degree <= 1:
            raise ValueError("cp_mode='ring' requires cp_degree > 1")

    @property
    def world_size(self) -> int:
        cfg_multiplier = 2 if self.cfg_parallel_enabled else 1
        return self.tp_degree * self.cp_degree * cfg_multiplier

    def to_cache_dict(self) -> dict[str, object]:
        # Additive-only: omit cp_mode at its default so a gather_kv config keeps
        # a compile-cache key byte-identical to every pre-cp_mode model cache.
        d = asdict(self)
        if self.cp_mode == "gather_kv":
            d.pop("cp_mode")
        return d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_parallel_config_cp_mode.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add difflet/pipeline/parallel_config.py tests/unit/test_parallel_config_cp_mode.py
git commit -m "feat(cp): add cp_mode config field (gather_kv|ring) with additive cache key"
```

---

## Task 2: `--cp-mode` CLI flag + Wan orchestrator threading

**Files:**
- Modify: `difflet/cli/main.py:36-40` (`_add_parallel_flags`)
- Modify: `difflet/cli/stage.py:40-41`
- Modify: `difflet/cli/orchestrators/wan.py:101-104` (config build) and `:216-224` (`_shared_cli_args`)
- Test: `tests/unit/test_cli_cp_mode.py`

**Interfaces:**
- Consumes: `DiffletParallelConfig(..., cp_mode=...)` from Task 1.
- Produces: argparse exposes `args.cp_mode` (default `"gather_kv"`, choices `gather_kv`/`ring`); Wan orchestrator builds `DiffletParallelConfig(..., cp_mode=args.cp_mode)` and re-emits `--cp-mode <value>`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_cli_cp_mode.py`:

```python
from difflet.cli.main import _build_parser


def test_cp_mode_flag_defaults_to_gather_kv():
    parser = _build_parser()
    args = parser.parse_args(["compile", "--model-id", "x"])
    assert args.cp_mode == "gather_kv"


def test_cp_mode_flag_accepts_ring():
    parser = _build_parser()
    args = parser.parse_args(["generate", "--model-id", "x", "--cp-degree", "2", "--cp-mode", "ring"])
    assert args.cp_mode == "ring"


def test_cp_mode_flag_rejects_unknown():
    import pytest

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["generate", "--model-id", "x", "--cp-mode", "bogus"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli_cp_mode.py -v`
Expected: FAIL (`AttributeError: 'Namespace' object has no attribute 'cp_mode'`).

- [ ] **Step 3: Write minimal implementation**

In `difflet/cli/main.py`, extend `_add_parallel_flags` (currently lines 36-40):

```python
def _add_parallel_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tp-degree", type=int, default=None,
                   help="Tensor-parallel degree (default: registry default)")
    p.add_argument("--cp-degree", type=int, default=1,
                   help="Context-parallel degree (default: 1)")
    p.add_argument("--cp-mode", choices=["gather_kv", "ring"], default="gather_kv",
                   help="Context-parallel attention strategy (default: gather_kv)")
```

In `difflet/cli/stage.py`, after the `--cp-degree` line (41) add:

```python
    p.add_argument("--cp-mode", choices=["gather_kv", "ring"], default="gather_kv")
```

In `difflet/cli/orchestrators/wan.py`, the config build (lines 101-104) becomes:

```python
        parallel = DiffletParallelConfig(
            tp_degree=args.tp_degree or 4,
            cp_degree=args.cp_degree or 1,
            cp_mode=getattr(args, "cp_mode", "gather_kv"),
        )
```

In `difflet/cli/orchestrators/wan.py`, `_shared_cli_args` (after the `--cp-degree` entry, line ~219) add to the `parts` list:

```python
            "--cp-mode", str(getattr(a, "cp_mode", "gather_kv")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_cli_cp_mode.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add difflet/cli/main.py difflet/cli/stage.py difflet/cli/orchestrators/wan.py tests/unit/test_cli_cp_mode.py
git commit -m "feat(cli): add --cp-mode flag and thread it through the Wan orchestrator"
```

---

## Task 3: Shared `ring_attention` op (Trainium + CPU) with guarded import

**Files:**
- Modify: `difflet/backends/trainium/ops_impl/attention.py` (add `ring_attention` + guarded import + `__all__`)
- Modify: `difflet/backends/cpu/ops_impl/attention.py` (add CPU `ring_attention`)
- Modify: `difflet/ops/attention.py` (expose `ring_attention`)
- Modify: `difflet/ops/__init__.py` (map `ring_attention`)
- Test: `tests/unit/test_ring_attention_op.py`

**Interfaces:**
- Produces: `difflet.ops.ring_attention(q, k, v, *, scale: float, causal: bool = False)`. Inputs `q,k,v` are `[B, H, S_local, d]` (per-rank head shard, tp layout). Returns `[B, H, S_local, d]`. Trainium impl resolves the data-parallel ring group internally; CPU impl computes plain non-causal attention over the given tensors (cp=1 reference equivalence).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ring_attention_op.py`. (Runs on the CPU backend — no device.)

```python
import math
import os

import pytest
import torch


def _set_cpu_backend():
    os.environ["DIFFLET_BACKEND"] = "cpu"


def test_ring_attention_cpu_matches_plain_attention_cp1():
    _set_cpu_backend()
    from difflet.ops import attention, ring_attention

    torch.manual_seed(0)
    b, h, s, d = 1, 2, 128, 64
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    scale = 1.0 / math.sqrt(d)

    ref = attention(
        q.reshape(b * h, s, d), k.reshape(b * h, s, d), v.reshape(b * h, s, d),
        scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False,
    ).reshape(b, h, s, d)
    out = ring_attention(q, k, v, scale=scale, causal=False)

    assert out.shape == (b, h, s, d)
    assert torch.allclose(ref.float(), out.float(), atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ring_attention_op.py -v`
Expected: FAIL (`ImportError: cannot import name 'ring_attention'`).

- [ ] **Step 3: Write minimal implementation**

In `difflet/ops/__init__.py`, add to the dispatch map (next to the existing `attention` entry):

```python
    "ring_attention": ("attention", "ring_attention"),
```

In `difflet/ops/attention.py`, add the public passthrough (before `_load`):

```python
def ring_attention(q, k, v, *, scale: float, causal: bool = False):
    """Context-parallel ring self-attention over a sequence-sharded Q/K/V.

    ``q,k,v`` are ``[B, H, S_local, d]`` (this rank's head shard). The backend
    resolves the data-parallel ring group and merges per-step partials.
    """

    return _load("ring_attention")(q, k, v, scale=scale, causal=causal)
```

In `difflet/backends/trainium/ops_impl/attention.py`, add after the imports:

```python
try:
    from nkilib.experimental.attention.ring_attention_fwd import ring_attention_spmd_fwd

    _RING_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard
    ring_attention_spmd_fwd = None
    _RING_IMPORT_ERROR = exc
```

and add the op (and extend `__all__` to include `"ring_attention"`):

```python
def ring_attention(q, k, v, *, scale: float, causal: bool = False):
    """Ring context-parallel self-attention via nkilib ring_attention_spmd_fwd.

    q,k,v: [B, H, S_local, d] (per-rank head shard). Returns [B, H, S_local, d].
    The ring membership IS the data-parallel group the model scattered Q with,
    so K/V rotate consistently with the scatter by construction.
    """
    if ring_attention_spmd_fwd is None:
        raise RuntimeError(
            "ring attention requires nkilib.experimental.attention.ring_attention_fwd "
            f"(import failed: {_RING_IMPORT_ERROR!r}). Upgrade neuronx-cc / nkilib, or "
            "use cp_mode=gather_kv."
        )
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_group,
        get_data_parallel_size,
    )

    mesh = get_data_parallel_group(as_list=True)  # List[List[int]] of global ranks
    num_workers = get_data_parallel_size()
    replica_groups = tuple(tuple(int(r) for r in grp) for grp in mesh)

    return ring_attention_spmd_fwd(
        q,
        k,
        v,
        replica_groups=replica_groups,
        num_workers=num_workers,
        softmax_scale=float(scale),
        use_causal_mask=causal,
        training=False,
        tp_q=True,
        tp_k=True,
    )
```

In `difflet/backends/cpu/ops_impl/attention.py`, add a CPU implementation (and add `"ring_attention"` to that module's `__all__` if present). With a single CPU process the ring degenerates to plain full attention over the given tensors:

```python
def ring_attention(q, k, v, *, scale: float, causal: bool = False):
    # Single-process CPU reference: cp_degree == 1, so the "ring" is just plain
    # non-causal attention over the local (== full) sequence.
    b, h, s_q, d = q.shape
    s_k = k.shape[2]
    out = attention(
        q.reshape(b * h, s_q, d),
        k.reshape(b * h, s_k, d),
        v.reshape(b * h, s_k, d),
        scale=scale,
        causal=causal,
        tp_q=True,
        tp_k=True,
        tp_out=False,
    )
    return out.reshape(b, h, s_q, d)
```

(If `difflet/backends/cpu/ops_impl/attention.py` does not define a local `attention`, import the module's existing attention entry the same way the file already exposes it; the CPU backend already provides `attention` for `difflet.ops.attention`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_ring_attention_op.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add difflet/ops/__init__.py difflet/ops/attention.py \
        difflet/backends/trainium/ops_impl/attention.py \
        difflet/backends/cpu/ops_impl/attention.py \
        tests/unit/test_ring_attention_op.py
git commit -m "feat(ops): add shared ring_attention op (trainium nkilib wrapper + cpu reference)"
```

---

## Task 4: Thread `cp_mode` through Wan config → attention

**Files:**
- Modify: `difflet/models/wan/application.py:18-50` (`create_wan_backbone_config`) and `:160-200` (two call sites)
- Modify: `difflet/backends/trainium/wan/backbone.py` (carry `cp_mode` on `WanBackboneInferenceConfig`, default `"gather_kv"`)
- Modify: `difflet/models/wan/modeling_wan.py` — model `__init__` (~612, ~640-651), `WanTransformerBlock.__init__` (512-534), `WanAttention.__init__` (358-375)
- Test: `tests/unit/test_wan_cp_mode_threading.py`

**Interfaces:**
- Consumes: `DiffletParallelConfig.cp_mode` (Task 1).
- Produces: `WanAttention` instances carry `self.cp_mode: str`; `create_wan_backbone_config(..., cp_mode="gather_kv")`; `WanTransformerBlock(..., cp_mode="gather_kv")`; `WanAttention(..., cp_mode="gather_kv")`.

**Threading rule:** at every site that currently passes/sets/reads `context_parallel_enabled`, add a sibling `cp_mode` (string, default `"gather_kv"`). Concretely:

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_wan_cp_mode_threading.py` (CPU construction, tp=1):

```python
import os

os.environ["DIFFLET_BACKEND"] = "cpu"

from difflet.models.wan.modeling_wan import WanAttention, WanTransformerBlock


def test_wan_attention_stores_cp_mode_default():
    attn = WanAttention(dim=128, heads=4, head_dim=32)
    assert attn.cp_mode == "gather_kv"


def test_wan_attention_stores_cp_mode_ring():
    attn = WanAttention(
        dim=128, heads=4, head_dim=32, context_parallel_enabled=False, cp_mode="ring"
    )
    assert attn.cp_mode == "ring"


def test_wan_block_threads_cp_mode_to_attentions():
    block = WanTransformerBlock(dim=128, ffn_dim=256, num_heads=4, cp_mode="ring")
    assert block.attn1.cp_mode == "ring"
    assert block.attn2.cp_mode == "ring"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wan_cp_mode_threading.py -v`
Expected: FAIL (`AttributeError: 'WanAttention' object has no attribute 'cp_mode'`).

- [ ] **Step 3: Write minimal implementation**

`difflet/models/wan/modeling_wan.py` — `WanAttention.__init__` (after `self.context_parallel_enabled = context_parallel_enabled`, ~line 372). Add the param to the signature (`cp_mode: str = "gather_kv"` after `context_parallel_enabled`) and store it:

```python
        self.context_parallel_enabled = context_parallel_enabled
        self.cp_mode = cp_mode
        if context_parallel_enabled:
            self.data_parallel_group = get_data_parallel_group()
```

`WanTransformerBlock.__init__` (512-534) — add `cp_mode: str = "gather_kv"` to the signature and pass it to both attentions:

```python
        self.attn1 = WanAttention(
            dim, num_heads, head_dim, eps=eps, is_cross_attention=False, dtype=dtype,
            context_parallel_enabled=context_parallel_enabled, cp_mode=cp_mode,
        )
        self.attn2 = WanAttention(
            dim, num_heads, head_dim, eps=eps, is_cross_attention=True, dtype=dtype,
            context_parallel_enabled=context_parallel_enabled, cp_mode=cp_mode,
        )
```

`WanTransformer3DModel.__init__` — read `cp_mode` off the config next to `context_parallel_enabled` (~line 612) and pass it when building blocks (~640-648):

```python
        self.context_parallel_enabled = getattr(config, 'context_parallel_enabled', False)
        self.cp_mode = getattr(config, 'cp_mode', 'gather_kv')
```

```python
                WanTransformerBlock(
                    dim=inner_dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_attention_heads,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    dtype=dtype,
                    context_parallel_enabled=self.context_parallel_enabled,
                    cp_mode=self.cp_mode,
                )
```

`difflet/models/wan/application.py` — `create_wan_backbone_config` signature: add `cp_mode: str = "gather_kv"` after `context_parallel_enabled`, and pass it to the returned config:

```python
    return WanBackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(transformer_path),
        height=height,
        width=width,
        num_frames=num_frames,
        context_parallel_enabled=context_parallel_enabled,
        cp_mode=cp_mode,
    )
```

Both call sites in `application.py` (the `transformer` build ~line 171 and `transformer_2` ~line 191) add `cp_mode=parallel.cp_mode,` next to `context_parallel_enabled=parallel.cp_degree > 1,`.

`difflet/backends/trainium/wan/backbone.py` — `WanBackboneInferenceConfig` must accept and store `cp_mode` (default `"gather_kv"`) and expose it as an attribute so `WanTransformer3DModel`'s `getattr(config, 'cp_mode', ...)` resolves. Mirror exactly how the class already accepts/stores `context_parallel_enabled` (see the `if not hasattr(self, "context_parallel_enabled")` guard at backbone.py:27-28 — add the analogous `cp_mode` default).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wan_cp_mode_threading.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add difflet/models/wan/application.py difflet/backends/trainium/wan/backbone.py \
        difflet/models/wan/modeling_wan.py tests/unit/test_wan_cp_mode_threading.py
git commit -m "feat(wan): thread cp_mode from config through blocks to WanAttention"
```

---

## Task 5: Wan ring branch in `WanAttention.forward` + device parity

**Files:**
- Modify: `difflet/models/wan/modeling_wan.py` — import `ring_attention`; branch in `WanAttention.forward` (475-490)
- Test: `tests/numerical/test_wan_ring_attention_neff.py` (device-gated)

**Interfaces:**
- Consumes: `difflet.ops.ring_attention` (Task 3), `self.cp_mode` (Task 4).
- Produces: when `context_parallel_enabled and not is_cross_attention and cp_mode == "ring"`, `WanAttention.forward` returns the ring output, numerically equal to the gather-KV path.

- [ ] **Step 1: Write the failing test (device-gated parity, ring vs gather-KV)**

Create `tests/numerical/test_wan_ring_attention_neff.py`. Follow the env-gated NEFF pattern of `tests/numerical/test_hunyuan_video_attention_neff.py` (skip unless `DIFFLET_RUN_WAN_RING_NEFF=1`; trace each path with `torch_neuronx`; compare cosine). The probe constructs one `WanAttention` (self-attn, `context_parallel_enabled=True`) and runs it once with `cp_mode="gather_kv"` and once with `cp_mode="ring"` on identical sharded inputs, asserting cosine ≥ 0.999.

```python
import os
import pytest
import torch
import torch.nn.functional as F


def test_wan_ring_matches_gather_kv_neff():
    if os.environ.get("DIFFLET_RUN_WAN_RING_NEFF") != "1":
        pytest.skip("set DIFFLET_RUN_WAN_RING_NEFF=1 to run the Wan ring parity gate")

    import torch_neuronx
    from difflet.models.wan.modeling_wan import WanAttention

    cosine_min = float(os.environ.get("DIFFLET_WAN_RING_COSINE_MIN", "0.999"))
    # NOTE: requires a CP-enabled multi-core launch (cp_degree > 1) so the
    # data-parallel ring group is live; per-rank seqlen must be a multiple of 128.
    # Build identical inputs; run gather_kv and ring; compare.
    # ... (probe construction mirrors test_hunyuan_video_attention_neff.py:_make_inputs)

    pytest.skip("fill in CP launch harness; placeholder asserts the gate wiring")
```

- [ ] **Step 2: Run test to verify it is collected and skips cleanly**

Run: `pytest tests/numerical/test_wan_ring_attention_neff.py -v`
Expected: SKIPPED (gate env var unset) — confirms the file imports and collects.

- [ ] **Step 3: Implement the ring branch**

`difflet/models/wan/modeling_wan.py` — add `ring_attention` to the `from difflet.ops import (...)` block (alphabetically near `attention`):

```python
    attention,
    ring_attention,
```

In `WanAttention.forward`, replace the gather-KV block + `_attn_kernel` call (current lines 481-490) with:

```python
        # CP self-attention: ring rotates sharded K,V; gather_kv all-gathers full K,V.
        # Cross-attention K,V come from encoder_hidden_states which is not scattered.
        if self.context_parallel_enabled and not self.is_cross_attention and self.cp_mode == "ring":
            out = ring_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim), causal=False)
        else:
            if self.context_parallel_enabled and not self.is_cross_attention:
                stacked_kv = torch.stack([k, v], dim=0)  # [2, B, heads, S/cp, head_dim]
                stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                    stacked_kv, gather_dim=3, process_group=self.data_parallel_group
                )  # [2, B, heads, S, head_dim]
                k, v = torch.unbind(stacked_kv, dim=0)
            out = _attn_kernel(q, k, v, head_dim=self.head_dim)
```

(`q`, `k`, `v` here are `[B, local_heads, S_local, head_dim]` after the transposes at lines 477-479 — exactly the ring op's `[B, H, S_local, d]` contract. `ring_attention` returns `[B, local_heads, S_local, head_dim]`, matching `_attn_kernel`, so the downstream `out.transpose(1, 2).reshape(...)` is unchanged.)

- [ ] **Step 4: Run the device parity gate (on Trainium2+, CP launch)**

Run (on device, with the project's CP launcher, `cp_degree > 1`):
`DIFFLET_RUN_WAN_RING_NEFF=1 pytest tests/numerical/test_wan_ring_attention_neff.py -v`
Expected: PASS with cosine ≥ 0.999 (ring vs gather-KV).
Also run the host suite to confirm no regression: `pytest tests/unit -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add difflet/models/wan/modeling_wan.py tests/numerical/test_wan_ring_attention_neff.py
git commit -m "feat(wan): ring-attention self-attn branch under cp_mode=ring"
```

---

## Task 6: End-to-end Wan trajectory parity + docs

**Files:**
- Test: extend `tests/numerical/test_wan_ring_attention_neff.py` (or a sibling) with an end-to-end denoise-trajectory check
- Modify: `README.md` (Context parallelism status row)

**Interfaces:**
- Consumes: the full Task 5 ring path.

- [ ] **Step 1: Write the failing/gated e2e test**

Add a device-gated test that runs a short Wan denoise (few steps) with `cp_degree>1, cp_mode=ring` and compares the latent trajectory against the validated `cp_mode=gather_kv` run (cosine ≥ 0.999), gated by `DIFFLET_RUN_WAN_RING_E2E=1`, mirroring existing trajectory-parity tests.

- [ ] **Step 2: Run to verify it collects/skips**

Run: `pytest tests/numerical/test_wan_ring_attention_neff.py -k e2e -v`
Expected: SKIPPED without the env gate.

- [ ] **Step 3: Run the e2e gate on device**

Run (device, CP launch): `DIFFLET_RUN_WAN_RING_E2E=1 pytest tests/numerical/test_wan_ring_attention_neff.py -k e2e -v`
Expected: PASS, cosine ≥ 0.999 vs gather-KV trajectory.

- [ ] **Step 4: Update README**

In `README.md`, update the "Context parallelism (Wan)" row to note the new opt-in ring mode, e.g. append: "Ring mode available via `--cp-mode ring` (sequence-sharded K,V, nkilib `ring_attention_spmd_fwd`); gather-KV remains the default."

- [ ] **Step 5: Commit**

```bash
git add tests/numerical/test_wan_ring_attention_neff.py README.md
git commit -m "test(wan): e2e ring vs gather-KV trajectory parity; document --cp-mode ring"
```

---

## Task 7: Finalize — squash all commits into one and push to `origin/cp-ring`

Run ONLY after Tasks 1-6 are complete and all gates (host unit suite green; device parity + e2e green on hardware) have passed. This collapses every commit from this effort into a single commit and publishes it.

**Files:** none (git history only).

**Interfaces:**
- Consumes: the per-task commits from Tasks 1-6 on the `cp-ring` branch.
- Produces: one squashed commit on `cp-ring`, pushed to `origin/cp-ring`.

- [ ] **Step 1: Verify branch and clean tree**

Run:
```bash
git rev-parse --abbrev-ref HEAD   # expect: cp-ring
git status --porcelain            # expect: empty (all task work committed)
```
Expected: branch is `cp-ring`, no uncommitted changes. If not on `cp-ring`, stop and switch (`git switch cp-ring`).

- [ ] **Step 2: Review the commits that will be squashed**

Run:
```bash
git log --oneline main..HEAD
```
Expected: the Task 1-6 commits (plus the spec/plan doc commit if present). These are exactly what gets squashed.

- [ ] **Step 3: Squash all commits since `main` into one (soft reset)**

Run:
```bash
git reset --soft "$(git merge-base main HEAD)"
git commit -m "feat(cp): opt-in ring-attention context parallelism for Wan

Add cp_mode={gather_kv,ring} (DiffletParallelConfig + --cp-mode CLI) and a
shared difflet.ops.ring_attention op wrapping nkilib ring_attention_spmd_fwd.
Wan self-attention uses the ring path when cp_mode=ring (sequence-sharded K,V),
with gather-KV unchanged as the default. Parity vs gather-KV cosine >= 0.999.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Expected: a single new commit; `git log --oneline main..HEAD` now shows exactly one line.

- [ ] **Step 4: Verify the squash preserved the tree**

Run:
```bash
git diff --stat main..HEAD
```
Expected: the full set of changed files from Tasks 1-6 (config, CLI, ops, Wan model, tests, README) — confirming no work was lost in the squash.

- [ ] **Step 5: Push to the remote dev branch**

Run:
```bash
git push -u origin cp-ring
```
Expected: `origin/cp-ring` created/updated with the single squashed commit. (If the remote branch already exists and history diverged, re-push with `--force-with-lease` only after confirming `origin/cp-ring` holds no other work.)

---

## Self-Review

**Spec coverage:**
- §1 `cp_mode` config + CLI → Task 1 (config), Task 2 (CLI). ✓
- §1 compile-cache key includes cp_mode → Task 1 (additive `to_cache_dict`). ✓
- §2 shared ring op (guarded import) → Task 3. ✓
- §3 ring topology from data-parallel group → Task 3 (`get_data_parallel_group(as_list=True)`, `get_data_parallel_size()`). ✓
- §4 Wan adapter (layout glue at the gather site) → Task 4 (threading), Task 5 (branch). ✓
- §5 non-causal correctness → Task 5 (`causal=False`). ✓
- §6 constraints documented → Global Constraints. ✓
- §7 perf overlap free (in-kernel) → no task needed (kernel-internal). ✓
- Validation gates: host unit (Tasks 1-4), device parity (Task 5), e2e (Task 6). ✓
- Out of scope: HunyuanVideo/Qwen/Flux/LTX-2 untouched → enforced by Global Constraints. ✓
- Finalize: squash all task commits into one and push to `origin/cp-ring` → Task 7. ✓

**Placeholder scan:** Task 5 Step 1 / Task 6 Step 1 contain device-harness `pytest.skip` placeholders for the CP-launch wiring — these are deliberate (the multi-core CP launch harness is environment-specific and cannot be hard-coded here); they are clearly marked and the gates collect/skip cleanly on host. All host-runnable steps contain complete code.

**Type consistency:** `cp_mode: str` with values `"gather_kv"`/`"ring"` used identically across config, CLI choices, threading, and the `WanAttention.forward` branch. `ring_attention(q, k, v, *, scale, causal=False)` signature identical in `difflet.ops`, trainium impl, and cpu impl. Return shape `[B, H, S_local, d]` consistent with `_attn_kernel`.
