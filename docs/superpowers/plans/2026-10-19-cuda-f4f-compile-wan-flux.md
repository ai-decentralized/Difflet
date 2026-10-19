# CUDA F4f — `--torch-compile` for Wan and Flux; the Flux single block's fused projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline TDD, one fresh whole-branch reviewer at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eager Difflet sits 2–8 % behind diffusers per run (Wan 44.5 vs 42.8 s, Flux 8.8 vs 8.3 s, Qwen 16.5 vs 15.2 s). F4 made Qwen faster than diffusers with `--torch-compile` (12.4 s vs 15.2): per-block inductor, opt-in, D4 unchanged. (1) Extend that option to Wan's experts and Flux's DiT — the CLI flag is already parsed and both orchestrators refuse it by name. (2) Remove the Flux single block's remaining eager gap: its `proj_out` runs as two gemms plus an add (`proj_out_attn`, `proj_out_mlp`, the Megatron split) where diffusers runs one gemm over `cat([attn, mlp])` — the F4e bench's +10 ms of gemm per forward.

**Architecture:** *(1)* `CudaWanTransformerApplication(torch_compile=False)` and `CudaFluxTransformerApplication(torch_compile=False)` compile each transformer block in `_prepare_module` the way Qwen's does (`torch.compile(block, mode="max-autotune-no-cudagraphs", fullgraph=False)`; `dynamic=False` — Wan and Flux run one shape per process, no prompt trim). `CudaWanApplication` / `CudaFluxApplication` read `torch_compile` from `application_kwargs` (or `DIFFLET_CUDA_TORCH_COMPILE=1`, as Qwen) and hand it to their transformer app(s); the kwarg leaves `_F1B_KWARGS` / `_F5B_KWARGS`. `WanCudaOrchestrator` / `FluxCudaOrchestrator` forward `--torch-compile` through `application_kwargs()` instead of refusing it. The compile cache key already carries application kwargs. *(2)* At tp=1 a single block may hold a fused `proj_out` (`nn.Linear(dim + mlp_hidden, dim)` whose weight is `[attn | mlp]` concatenated, as diffusers' is): `NeuronFluxSingleTransformerBlock.fuse_proj_out()` builds it from the two split weights and the bias and drops them; `forward` takes the fused path when it is present (`cat([attn_output, mlp_hidden_states], dim=2)` → one gemm → gate, residual), otherwise today's two-gemm path. The CUDA transformer application calls it after loading at tp=1; Trainium/TPU never do, so their graphs are unchanged. Numerics: one fp32-accumulated gemm rounded once, where the split path rounded twice and added — closer to diffusers (which computes exactly this).

**Spec:** `docs/plans/2026-10-16-cuda-backend-support.md` — D4 (eager by default; `torch.compile` a measured, opt-in option), the F4 entry (compile recipe, the gates compile must pass: parity ≤ 1.15× the control, e2e floors), the F4e entry (the gemm half of the Flux gap), the F1b / F5b follow-up lists (`torch_compile` named there).

## Global Constraints

- `DIFFLET_BACKEND=cpu` on every Mac command; CPU gate green after each task; black 100; one task, one commit. Branch `feat/cuda-f4f-compile-wan-flux` on `feat/cuda-f4e-fused-rmsnorm` (2df7993, PR #99).
- Eager stays the default and byte-identical: `torch_compile=False` (and no fused `proj_out` off CUDA) is today's path on every backend; Trainium/TPU graphs unchanged.
- Numeric claims come from the pod: compiled Wan and Flux must pass their parity and e2e gates; the fused projection too.

## Review Focus

1. Compile wraps every block and only when asked (Task 1 tests on the tiny CPU snapshots: `OptimizedModule` per block with the flag, none without; both Wan experts; Flux's double and single blocks).
2. The flag reaches the apps from the CLI: `--torch-compile` → `application_kwargs["torch_compile"]` for Wan and Flux (orchestrator tests), and from `DIFFLET_CUDA_TORCH_COMPILE=1`.
3. The fused projection is exact in structure and loads from the same checkpoint: fused weight == `cat([attn, mlp], dim=1)`, bias carried; forward with the fused path equals the two-gemm path within fp32 rounding on CPU (Task 2 test) and the diffusers reference as before; `fuse_proj_out()` is a no-op off tp=1 / twice.

---

### Task 1: `--torch-compile` for Wan and Flux

**Files:**
- Modify: `difflet/backends/cuda/wan/transformer.py`, `difflet/backends/cuda/flux/transformer.py` (the `torch_compile` kwarg, per-block compile in `_prepare_module`), `difflet/models/wan/cuda_application.py`, `difflet/models/flux/cuda_application.py` (read the kwarg / env, pass it on; drop it from the refused kwargs), `difflet/cli/orchestrators/wan_cuda.py`, `difflet/cli/orchestrators/flux_cuda.py` (forward instead of refuse)
- Test: `tests/unit/backends/test_cuda_wan_transformer.py`, `tests/unit/backends/test_cuda_flux_transformer.py`, `tests/unit/models/wan/test_wan_cuda_application.py`, `tests/unit/models/flux/test_cuda_flux_application.py` (or the existing file), `tests/unit/cli/test_orchestrator_wan_cuda.py`, `tests/unit/cli/test_orchestrator_flux_cuda.py`

- [x] **Step 1:** failing tests → **Step 2:** run → **Step 3:** implement → **Step 4:** CPU gate → **Step 5: Commit** `cuda: --torch-compile for Wan's experts and Flux's DiT, per-block inductor as Qwen's (F4f task 1)`.

### Task 2: The Flux single block's fused `proj_out` at tp=1

**Files:**
- Modify: `difflet/models/flux/modeling_flux.py` (`NeuronFluxSingleTransformerBlock.fuse_proj_out`, the fused forward path), `difflet/backends/cuda/flux/transformer.py` (call it after loading at tp=1)
- Test: `tests/unit/models/flux/test_flux_vs_diffusers_cpu.py` or a new `test_flux_fused_proj_out.py`, `tests/unit/backends/test_cuda_flux_transformer.py`

- [x] **Step 1:** failing tests → **Step 2:** run → **Step 3:** implement → **Step 4:** CPU gate → **Step 5: Commit** `flux: the single block's proj_out as one gemm at tp=1 on CUDA (F4f task 2)`.

### Task 3: Status log, docs, PR; the pod run (pending the owner's yes)

- [ ] Status-log entry; push; PR stacked on #99. Pod: Wan and Flux `--torch-compile` smoke + parity + e2e; the Flux bench (eager with the fused projection, and compiled); the Qwen gates (nothing changed for Qwen, a regression net).

## Rulings

- R1: `dynamic=False` for Wan and Flux (one static shape per process); Qwen keeps `dynamic=True` for its trimmed prompt lengths.
- R2: the fused projection is a load-time transformation on CUDA at tp=1, never a checkpoint format change: the converter keeps splitting, Trainium/TPU keep the split weights.
- R3: inline execution, one fresh whole-branch reviewer.
