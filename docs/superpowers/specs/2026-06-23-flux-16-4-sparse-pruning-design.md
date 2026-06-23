# FLUX 16:4 Structured Sparse Pruning Design

**Date:** 2026-06-23
**Status:** Approved — awaiting implementation plan

## Overview

Apply 16:4 structured sparsity pruning to the FLUX diffusion model on AWS
Trainium, using the NKI2 private `nc_matmul_sparse` ISA instruction wrapped in a
`TorchXlaKernel` for XLA compilation. Supports both BF16 and FP8 sparse modes.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Precision | BF16 + FP8 (both) | BF16 for safety; FP8 for max speed/savings |
| Pruning scope | All Linear layers | Maximize speedup per user priority |
| Pruning strategy | Magnitude-based (per 16-group) | Simplicity, no calibration data needed |
| Integration method | TorchXlaKernel (matching existing MX path) | Cleanest fit with XLA compilation pipeline |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│ Offline (scripts/prune_flux.py)                       │
│                                                       │
│  HF weights → magnitude prune → compress → save       │
│  (per layer: keep top-4 of every 16 along K dim)      │
│  Output: compressed_weight + tags per linear layer     │
└──────────────────────┬───────────────────────────────┘
                       │ load compressed weights
┌──────────────────────▼───────────────────────────────┐
│ Runtime (Trainium)                                    │
│                                                       │
│  FluxBackboneInferenceConfig.sparse_mode = "bf16"/"fp8"│
│         │                                             │
│  modeling_flux.py                                     │
│    ColumnParallelLinear → SparseColumnParallelLinear   │
│    RowParallelLinear    → SparseRowParallelLinear      │
│         │                                             │
│  sparse_linear.py                                     │
│    TorchXlaKernel(nc_matmul_sparse_kernel)             │
│         │                                             │
│  NKI kernel (nki_kernels/sparse_matmul.py)             │
│    nc_matmul_sparse(moving, stationary, tags, ...)     │
│         │                                             │
│  XLA compiler → NEFF → Trainium hardware              │
└──────────────────────────────────────────────────────┘
```

## Component Details

### 1. NKI Sparse Matmul Kernel

**File:** `difflet/backends/trainium/nki_kernels/sparse_matmul.py`

Two kernels:

- `sparse_matmul_bf16_kernel` — BF16 compressed stationary × BF16 moving → BF16 out
- `sparse_matmul_fp8_kernel` — FP8 x4 packed stationary × FP8 x4 packed moving → BF16 out

Both use `neuronxcc.nki._private.private_api.nc_matmul_sparse` with
`compress_ratio=4` (16:4 pattern). Partition dimension P is the compressed
contraction dimension: for weight [M, K], after 16:4 compression the
contraction dimension becomes K/16 groups, each contributing 4 nonzero values.
The exact P/F (partition/free) layout follows the ISA specification from the
probe scripts (`scripts/sparse_fp8_focused.py`).

### 2. Sparse Linear Ops

**File:** `difflet/backends/trainium/ops_impl/sparse_linear.py`

Wraps the NKI kernels as `TorchXlaKernel` (singleton pattern, matching
`difflet/backends/trainium/ops_impl/mx.py`):

```python
_SPARSE_MATMUL_BF16_TORCHXLA_KERNEL = None

def _sparse_matmul_bf16_torchxla_kernel():
    global _SPARSE_MATMUL_BF16_TORCHXLA_KERNEL
    if _SPARSE_MATMUL_BF16_TORCHXLA_KERNEL is None:
        from nki.framework.torch_xla import TorchXlaKernel
        _SPARSE_MATMUL_BF16_TORCHXLA_KERNEL = \
            sparse_matmul_bf16_kernel[1]._to_subclass(TorchXlaKernel)
    return _SPARSE_MATMUL_BF16_TORCHXLA_KERNEL
```

### 3. Sparse Linear Layers

**File:** `difflet/layers/sparse_linear.py`

- `SparseColumnParallelLinear` — drop-in replacement for `ColumnParallelLinear`
  - Stores pre-compressed `compressed_weight` + `tags` (non-trainable parameters)
  - Forward: call `TorchXlaKernel` sparse matmul → gather output (if needed)
  - TP-compatible: weights compressed per-shard

- `SparseRowParallelLinear` — drop-in replacement for `RowParallelLinear`
  - Same pattern; input_is_parallel mode preserved

### 4. Pruning Tool

**File:** `scripts/prune_flux.py`

Algorithm per weight tensor `W[M, K]`:

1. Reshape to `[M, K/16, 16]`
2. Per group of 16: select 4 positions with largest |value|
3. Extract kept values → `compressed_weight [M, K/16, 4]`
4. Pack 4-bit indices into `tags` (uint16 per compressed element)
5. For FP8 mode: additionally quantize kept values to FP8 E4M3FN

Output: checkpoint with `{layer}.sparse_weight` and `{layer}.sparse_tags` entries.

### 5. Model Integration

**File:** `difflet/models/flux/modeling_flux.py` (modified)

- `FluxBackboneInferenceConfig` gains `sparse_mode: str | None` field
- In `NeuronFluxAttention`, `NeuronFluxSingleTransformerBlock`, `NeuronFluxTransformerBlock`, `NeuronFeedForward`, `NeuronFluxTransformer2DModel`: when `sparse_mode` is set, replace `ColumnParallelLinear`/`RowParallelLinear` instances with their sparse counterparts
- `convert_hf_to_neuron_state_dict` loads compressed weights when `sparse_mode` is active

**Layers replaced (all Linear layers in the model):**

| Layer | Class | Sparse Replacement |
|---|---|---|
| x_embedder | ColumnParallelLinear | SparseColumnParallelLinear |
| context_embedder | ColumnParallelLinear | SparseColumnParallelLinear |
| proj_out | ColumnParallelLinear | SparseColumnParallelLinear |
| to_q / to_k / to_v | ColumnParallelLinear | SparseColumnParallelLinear |
| add_q_proj / add_k_proj / add_v_proj | ColumnParallelLinear | SparseColumnParallelLinear |
| to_out[0] / to_add_out | RowParallelLinear | SparseRowParallelLinear |
| proj_mlp | ColumnParallelLinear | SparseColumnParallelLinear |
| proj_out_attn / proj_out_mlp | RowParallelLinear | SparseRowParallelLinear |
| ff.net[0] (GELU act) | ColumnParallelLinear (via NeuronGELU) | SparseColumnParallelLinear |
| ff.net[2] (proj out) | RowParallelLinear | SparseRowParallelLinear |
| ff_context (same structure) | ColumnParallelLinear / RowParallelLinear | Sparse* variants |

### 6. Configuration

```python
class FluxBackboneInferenceConfig(InferenceConfig):
    def __init__(self, *args,
                 cfg_parallel_enabled=False,
                 context_parallel_enabled=False,
                 sparse_mode: str | None = None,  # NEW: None | "bf16" | "fp8"
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_mode = sparse_mode
        # Validate
        if self.sparse_mode not in (None, "bf16", "fp8"):
            raise ValueError(f"sparse_mode must be None, 'bf16', or 'fp8', got {sparse_mode!r}")
```

## Data Flow (Inference)

```
Input activations [B, S, D]
    │
    ▼
SparseColumnParallelLinear.forward()
    │
    ├─ reshape activation to match kernel P/F layout
    ├─ TorchXlaKernel(compressed_weight, tags, activation) → sparse matmul
    ├─ (XLA traces this as a custom call into the NEFF)
    └─ gather output along TP dimension (if gather_output=True)
    │
    ▼
Output [B, S, D'] (numerically ≈ dense matmul)
```

## Tile Layout Constraints

Based on probe scripts (`scripts/sparse_fp8_focused.py`, `scripts/sparse_matmul_nki2_probe.py`):

- P dimension (contraction after compression) must match across stationary, moving, and tags
- For BF16: P = K_compressed = K/16 groups × R=4 nonzero = K/4 elements
- For FP8: P packed as x4, so P = K/(16×4) = K/64 x4 elements
- F dimensions are free (output rows and columns)
- The exact layout will be validated during kernel bring-up against ISA error messages

## Testing Plan

### Numerical Correctness
1. **Unit:** Single tile `nc_matmul_sparse` vs dense matmul with 16:4 mask — cosine similarity > 0.99 (BF16), > 0.98 (FP8)
2. **Layer:** `SparseColumnParallelLinear` output vs `ColumnParallelLinear` with masked weight
3. **End-to-end:** FLUX latent output max relative error < 1e-3 vs dense baseline

### XLA Compilation
1. Confirm `TorchXlaKernel` wraps successfully (no import/type errors)
2. Confirm NEFF generation succeeds on Trainium
3. Confirm cached NEFF can be reloaded on subsequent runs

### Performance
1. Single-layer sparse matmul latency vs dense — target approx 4x (16:4 theoretical)
2. Full model step time reduction — depends on fraction of FLOPs in Linear layers
3. Memory bandwidth reduction — compressed weights use 4x (BF16) or 16x (FP8) less HBM

## Files Summary

| Action | File | Purpose |
|---|---|---|
| NEW | `difflet/backends/trainium/nki_kernels/sparse_matmul.py` | NKI kernels |
| NEW | `difflet/backends/trainium/ops_impl/sparse_linear.py` | TorchXlaKernel wrappers |
| NEW | `difflet/layers/sparse_linear.py` | nn.Module drop-ins |
| NEW | `scripts/prune_flux.py` | Offline pruning tool |
| MODIFY | `difflet/models/flux/modeling_flux.py` | Sparse mode + layer switching |
| MODIFY | `difflet/backends/trainium/flux/__init__.py` | Exports (if needed) |

## Out of Scope

- Dynamic/input-dependent sparsity patterns
- Sparsity-aware training or fine-tuning
- Automatic sparsity detection (compiler auto-routing)
- Non-16:4 sparsity patterns (e.g., 4:8, unstructured)
- VAE or Text Encoder pruning (Transformer backbone only)
