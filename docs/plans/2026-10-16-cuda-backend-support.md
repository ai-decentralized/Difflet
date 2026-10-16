# CUDA Backend Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Every phase ends by appending its verdict and measurements to the **Status log** at the bottom of this file — the TPU plan that this one is modelled on was never committed, and its Phase 0 findings survive only as docstrings. Do not repeat that.

**Goal:** Make `DIFFLET_BACKEND=cuda` a real backend. **Release 1** is one model on one GPU: Qwen-Image at `tp=1` on a single NVIDIA RTX Pro 6000, generating a diffusers-faithful image through `difflet run`, proven by the same relative parity gate the TPU backend uses. Everything else — Wan, Flux, a second card, serving, performance — is a scheduled follow-up with its own status-log entry, not part of Release 1.

**Architecture:** Mirror the TPU backend's shape, not Trainium's. `difflet/backends/cuda/` gets a `runtime.py` (already a stub), an `ops_impl/` package implementing the frozen `difflet.ops` v1 surface on plain `torch`, and a `core/` package with the component lifecycle `DiffletPipeline` calls reflectively (`compile() / has_compiled_artifacts() / load() / __call__()`). Model code keeps importing only `difflet.ops`; per-model `entry.py` factories gain a `backend == "cuda"` branch exactly as `difflet/models/wan/entry.py` has for TPU. CUDA has **no ahead-of-time compile step**: `requires_aot=False`, so `compile()` validates and writes a manifest, and `load()` builds the model eagerly on the device. The backend is written as a **generic torch-accelerator backend**: nothing CUDA-specific outside `ops_impl/platform.py`, so ROCm can later register the same code under its own name.

**Tech Stack:** Python ≥ 3.10, `torch` built for CUDA 12.8 or newer (the Pro 6000 is Blackwell; older wheels carry no kernels for it), `diffusers==0.38.0`, `safetensors`, `pytest`. `torch.distributed` + NCCL + `torchrun` only when a second card arrives (follow-up F2). Deliberately **not** used in Release 1: `flash-attn`, `torch.compile`, Triton, CUTLASS, MX/FP8.

---

## Decisions (settled with the owner, 2026-10-16)

Reached in a structured review of the first draft; recorded here so nobody re-litigates them by accident. Each one names its alternative and why it lost.

| # | Decision | Alternative rejected | Why |
|---|---|---|---|
| D1 | **Hardware: one RTX Pro 6000 (Blackwell, 96 GB GDDR7, PCIe, no NVLink)** from the Yottalabs console. A second card is a follow-up. | 2 × H100 80 GB as first drafted | It is what is available. 96 GB fits every current model's bf16 weights at `tp=1`, including both Wan experts (~56 GB). |
| D2 | **Release 1 = Qwen-Image, `tp=1`, one GPU, `difflet run`, parity gate passed.** | Qwen + Wan at tp=1 and tp=2; or the full matrix | One engineer, roughly three GPU days. Anything wider is a follow-up with its own entry. |
| D3 | **Model order: Qwen-Image → Wan → Flux.** | Flux first (README headline) | Qwen-Image's and Wan's modeling already import only `difflet.ops` (the TPU port needed that). `modeling_flux.py` imports five Trainium core classes and must be split first, with Trainium proven byte-identical — which needs trn2 access we do not have yet (D13). |
| D4 | **Eager PyTorch.** `compile()` writes a manifest only. | `torch.compile` / AOTInductor as the architectural premise, like TPU's `torch.export` | Difflet's multi-shape sets would mean recompilation pain for no proven win. `torch.compile` is a measured option in follow-up F4. |
| D5 | **One process per GPU under `torchrun` when multi-GPU arrives. Release 1 initialises no `torch.distributed` at all** (world size 1). | One process driving all GPUs | Does not move bytes faster — PCIe is the limit either way — and loses every PyTorch distributed tool. |
| D6 | **Second card, when used: data parallel and CFG-parallel first. Tensor parallel across PCIe only when a named workload fails to fit one card, and only after measuring it against context parallel.** | TP by default, as on Trainium | No NVLink: TP all-reduces every layer (~60×/step) over PCIe; CFG-parallel syncs once per step. Weights fit at tp=1; it is long-video *activations* that may overflow, and CP moves less data per layer. |
| D7 | **Attention through `torch.nn.functional.scaled_dot_product_attention`. No `flash-attn` package.** | `flash-attn` 2/3 | Prebuilt wheels for this architecture are patchy and a build costs most of a day. Phase 0 records which SDPA kernel is selected; revisit only if a profile names attention. |
| D8 | **All components on the GPU in one process: text encoder, DiT, VAE.** | TPU's host-side encoders/VAE and staged subprocesses | TPU did that because of 16 GB chips. 96 GB removes the reason. |
| D9 | **bf16 only. MX ops raise `NotImplementedError`.** | FP8 / NVFP4 on Blackwell tensor cores; dequantize-at-load | No Release 1 model needs MX. FP8 is a measured follow-up (F4). |
| D10 | **Lift TPU's Megatron layers, checkpoint loader and shard-at-load into shared code**, parameterised by a collectives module. | Copy into `backends/cuda/` | A sharding bug is then fixed once. Guarded by the TPU unit tests before and after. |
| D11 | **Design as a generic torch-accelerator backend; ROCm registers later under its own name.** Not implemented or tested here. | CUDA-specific code | Torch on ROCm exposes the same API. CUDA-only facts (compute capability, NCCL name) sit in `ops_impl/platform.py`. |
| D12 | **CPU-only CI job** (`DIFFLET_BACKEND=cpu`, unit suite) added at the end of Phase 1. GPU tests stay manual, results in the status log. | No CI (today: no `.github/` at all) | Phase 1 makes a laptop-green suite possible for the first time. |
| D13 | **No Trainium host requested until the Flux split is scheduled.** Flux split deferred until a Trainium verification path exists. | Request trn2 now | Nothing in Release 1 needs it; the shared lift (D10) is guarded by TPU unit tests. Known-good type when needed: `trn2.3xlarge`. |
| D14 | **Parity oracle runs on the GPU**: diffusers fp32 reference, diffusers bf16 control and the Difflet run, three forwards in one script. | TPU's two-stage CPU harness (57–80 GB host RAM) | 96 GB on the card makes the host-RAM requirement unnecessary. Same relative gates. |
| D15 | **Weights come straight from Hugging Face on the box** (`difflet download`, public model, no token). | S3 artifact store / internal mirror | Simplest; the compile-cache S3 store is not involved in Release 1. |
| D16 | **Three PRs**: (1) this plan + `.gitignore` fix + CI job; (2) Phase 1 hygiene; (3) CUDA backend + Qwen-Image tp=1 + parity script. Each gated on the CPU unit suite. | One PR | Hygiene touches Trainium-adjacent files and deserves its own review. |
| D17 | **`cuda` extra in `pyproject.toml` + `requirements-cuda.lock` generated on the box in Phase 0.** | Ad-hoc pip installs | The compile-cache key hashes toolchain versions; future hosts must reproduce them. |
| D18 | **`.gitignore` gains `!docs/plans/`** so plans are tracked by default. | Force-add each plan | The TPU plan was lost this way. |
| D19 | **Timebox: three GPU working days from the moment the box is available.** Phase 1 hygiene is done on the Mac before that clock starts. | Open-ended | Makes slippage visible in the status log. |

Still factual, not decided (first Phase 0 tasks once SSH details exist): the box's hostname, OS, Python version, NVIDIA driver, and the exact torch wheel.

---

## Why this plan looks the way it does

Three facts about the repo, measured 2026-10-16 against commit `117ca45` on a non-Neuron host (macOS, Python 3.14.7, torch 2.14.1).

1. **The TPU backend is the precedent, and it worked because Qwen-Image and Wan have backend-neutral modeling.** `difflet/models/wan/modeling_wan.py` and `difflet/models/qwen_image/modeling_qwen.py` import only `difflet.ops`; the Trainium specifics live in `application.py` and under `difflet/backends/trainium/`. Flux does not have that split yet (D3).

2. **The Neuron SDK leaks well outside `difflet/backends/trainium/`.** 24 files outside that package import `neuronx_distributed`, `torch_neuronx`, `torch_xla` or `nkilib` directly (9 in `difflet/utils/`, 4 in `difflet/cli/`, 3 in `difflet/models/`, 3 in `difflet/serving/`, 5 in `difflet/backends/tpu/` — the TPU ones are legitimate `torch_xla` use). Another ~35 files import `difflet.backends.trainium` directly, including `difflet/pipeline/difflet_pipeline.py` (`canonicalize_shapes`, and `_resolve_load_rank_range` hard-codes `get_backend("trainium")`) and `difflet/pipeline/compile_cache.py`. On a non-Neuron host with `DIFFLET_BACKEND=cpu`: `scripts/test_imports.sh` fails at `difflet.models.flux.application`, and `tests/unit` reports **2088 passed, 131 failed, 36 errors, 43 skipped** — every failure a missing Neuron module or a model registered as `backends=("trainium",)`. The GPU box will not have the Neuron SDK either, so Phase 1 is on the critical path, not optional hygiene.

3. **Auto-detection already returns `"cuda"` when `torch.cuda.is_available()`** (`difflet/backends/registry.py:_auto_detect_backend`), and the compile cache already keys on a non-default backend plus per-backend extra toolchain packages (`compile_cache.py:_BACKEND_EXTRA_TOOLCHAIN_PACKAGES`). Both only need entries.

Two TPU lessons carried over verbatim:

- **A wrong-but-silent implementation is worse than an explicit failure.** Context-parallel attention modes and MX ops raise `NotImplementedError` until their follow-up lands (`difflet/backends/tpu/ops_impl/attention.py`, `mx.py`).
- **The oracle is upstream diffusers, never another difflet backend.** Two independently wrong implementations can agree (`tests/numerical/test_tpu_vs_diffusers.py`, and the DEVELOPER.md "validation lesson").

---

## Global Constraints

- Nothing under `difflet/backends/cuda/` or the shared package imports `neuronx_distributed`, `torch_neuronx`, `torch_xla`, `nkilib`, or anything from `difflet.backends.trainium`. Enforce with a unit test, not a convention.
- Model code (`difflet/models/**`, `difflet/layers/**`) imports only `difflet.ops`, `torch`, stdlib, `diffusers`, `transformers` — the existing `scripts/test_imports.sh` rule. Phase 1 makes that script pass on a host without the Neuron SDK.
- Every CUDA numerical gate compares against upstream `diffusers` (fp32 reference + same-dtype control, relative gates as in `tests/numerical/test_tpu_vs_diffusers.py`). CPU-vs-CUDA is allowed only as a fast unit-level smoke, never as acceptance.
- Unimplemented ops raise `NotImplementedError` naming the follow-up that implements them. No silent fallbacks, no `pass`.
- `DiffletPipeline` calls `compile(path[, debug])`, `has_compiled_artifacts(path)`, `load(path[, start_rank_id, local_ranks_size, skip_warmup])` reflectively via `inspect.signature`. Parameter names are part of the contract; do not rename.
- Release 1 runs in one process on one device. No process group is created when `WORLD_SIZE` is 1 or unset. Every collective is the identity at world size 1, exactly as the CPU backend.
- CUDA-only facts live in `ops_impl/platform.py`: compute capability, device name, the distributed backend name (`"nccl"`), SDPA kernel selection. Nothing else may branch on "is this CUDA".
- Tests mirror the source tree under `tests/unit/`; GPU-only tests carry a marker and skip without CUDA; on-device evidence is logged to files under `/tmp/logs/` on the box and summarised in the Status log. Black, line length 100.
- Each task is one commit (repo convention). After every task: `DIFFLET_BACKEND=cpu ./scripts/test_unit.sh`, and from Phase 1 on, `DIFFLET_BACKEND=cpu ./scripts/test_imports.sh`.
- The timebox is three GPU days (D19). A phase that will not fit is cut, not stretched; the cut is written in the Status log.

---

## File Map

**Created:**
- `.github/workflows/unit-cpu.yml` — `pip install -e ".[test]"`, `DIFFLET_BACKEND=cpu pytest tests/unit -q` on every PR (D12)
- `difflet/backends/_torch_megatron/{__init__,linear,norm,embeddings,weights,checkpoint}.py` — lifted from `backends/tpu/` (D10); takes a collectives module, no device assumptions
- `difflet/backends/cuda/ops_impl/__init__.py`
- `difflet/backends/cuda/ops_impl/collectives.py` — identity at world size 1; `torch.distributed` paths added in F2
- `difflet/backends/cuda/ops_impl/parallel_mesh.py` — world-size-1 mesh now; NCCL groups from `MeshSpec` in F2
- `difflet/backends/cuda/ops_impl/linear.py`, `norm.py`, `embeddings.py` — thin re-exports of `_torch_megatron` bound to CUDA collectives
- `difflet/backends/cuda/ops_impl/attention.py` — `attention`, `cross_attention` on SDPA; CP modes raise (F2/F6)
- `difflet/backends/cuda/ops_impl/mx.py` — raises (D9)
- `difflet/backends/cuda/ops_impl/platform.py` — `hardware` enum, `get_platform_target()`, compute capability, SDPA kernel report
- `difflet/backends/cuda/core/__init__.py`, `application_base.py` — `CudaComponentApplication`: manifest-only `compile()`, eager `load()`
- `difflet/backends/cuda/qwen_image/{__init__,config,transformer}.py`
- `difflet/models/qwen_image/cuda_application.py` — text encoder + DiT + VAE all on the GPU (D8)
- `difflet/pipeline/bucketing.py` — backend-neutral `canonicalize_shapes` (moved out of `backends/trainium/core/bucketing.py`, which re-exports it)
- `tests/unit/backends/test_cuda_backend.py`, `test_cuda_import_hygiene.py`, `test_cuda_layers.py`, `test_cuda_attention.py`, `test_cuda_application.py`, `test_torch_megatron.py`
- `tests/numerical/cuda_parity_gpu.py` — fp32 oracle + bf16 control + Difflet, one GPU, one script (D14); `tests/numerical/test_cuda_vs_diffusers.py` reads its JSON
- `scripts/cuda_smoke.sh` — `DIFFLET_BACKEND=cuda difflet run --model-id Qwen/Qwen-Image --tp-degree 1 …`
- `requirements-cuda.lock` (D17)

**Modified:**
- `.gitignore` — `!docs/plans/` (D18)
- `pyproject.toml` — add the missing `transformers` dependency (9 unit errors today); `[project.optional-dependencies] cuda = [...]` (D17)
- `difflet/backends/cuda/runtime.py` — real `prepare_runtime`: pick device, no process group at world size 1
- `difflet/backends/tpu/ops_impl/{linear,norm,embeddings}.py`, `backends/tpu/core/{weights,checkpoint}.py` — become thin wrappers over `_torch_megatron` (D10); TPU unit tests unchanged
- `difflet/pipeline/compile_cache.py` — `_BACKEND_EXTRA_TOOLCHAIN_PACKAGES["cuda"]`; `torch.version.cuda` + cuDNN in the key
- `difflet/pipeline/difflet_pipeline.py` — import `canonicalize_shapes` from `difflet.pipeline.bucketing`; `_resolve_load_rank_range` uses the selected backend
- `difflet/registry.py` — Qwen-Image `backends=("trainium", "tpu", "cuda")`
- `difflet/models/qwen_image/entry.py` — `backend == "cuda"` branch
- `difflet/utils/{random,accuracy,benchmark,constants,debug_utils,hf_adapter,tensor_capture_utils,testing,version_utils}.py` — Neuron imports deferred into functions or relocated (Phase 1)
- `difflet/models/flux/entry.py` — Trainium import moved inside the Trainium branch so the import guard passes (the modeling split itself is F5)
- `scripts/test_imports.sh`, `scripts/test_unit.sh` — honour `DIFFLET_BACKEND` and a non-Neuron `PYTHON_BIN`
- `DEVELOPER.md` — backend section, CUDA quick start, link to this plan

---

## Release 1 — Qwen-Image at tp=1 on one Pro 6000

### Phase 0 — Box check and verdicts (first hours on the GPU box)

Goal: confirm the environment and write the facts down before any code depends on them.

- [x] **0.1** Record: hostname, OS, kernel, `nvidia-smi` driver and CUDA versions, GPU name and memory, Python version, disk free.
- [x] **0.2** Install torch from the CUDA 12.8+ wheel index into a fresh venv; confirm `torch.cuda.is_available()`, `torch.cuda.get_device_capability()`, and that a bf16 matmul runs. If no wheel supports the box's Python, pick the newest Python the wheels do support and record it.
- [x] **0.3** Load diffusers' `QwenImageTransformer2DModel` from the real checkpoint in bf16 on the GPU, one seeded forward: wall time, peak memory. This is the oracle baseline and proves the card holds the model.
- [x] **0.4** Record which SDPA kernel torch selects for Qwen-Image's head dim and sequence length (`torch.nn.attention.sdpa_kernel` probing), per D7.
- [x] **0.5** Generate `requirements-cuda.lock` from the venv; write the `cuda` extra (D17).
- [x] **0.6** Append the Phase 0 verdict to the Status log and start the three-day clock (D19).

Exit: Status log has a dated "Phase 0 verdict"; the capability triple in `runtime.py` is confirmed (`requires_aot=False`, `single_process_multi_core=False`, `supports_torchrun_mpmd=True`).

### Phase 1 — Import hygiene and CI (Mac, before the box arrives) — PR 1 and PR 2

Goal: `DIFFLET_BACKEND=cpu ./scripts/check_quick.sh` is green on a laptop, so the GPU box (which also has no Neuron SDK) can import Difflet at all.

**PR 1 — plan, gitignore, CI**
- [ ] **1.1** `.gitignore`: `!docs/plans/`; commit this plan (D18).
- [ ] **1.2** `.github/workflows/unit-cpu.yml` running `DIFFLET_BACKEND=cpu pytest tests/unit -q --continue-on-collection-errors` on Ubuntu, Python 3.12. It is red on day one and turns green with PR 2; that is the point (D12).

**PR 2 — hygiene**
- [x] **1.3** `tests/unit/backends/test_cuda_backend.py` (copy `test_tpu_backend.py`): `"cuda"` resolves; capabilities per Phase 0; `prepare_runtime` on a CUDA-less host raises a clear `RuntimeError`; auto-detect order Trainium → TPU → CUDA → ROCm pinned.
- [x] **1.4** `tests/unit/backends/test_cuda_import_hygiene.py`: static scan of `difflet/backends/cuda/**` and `difflet/backends/_torch_megatron/**` for forbidden imports; `importlib.import_module` of every `difflet.ops` module on the CUDA backend (xfail per module until Phase 2 lands it).
- [x] **1.5** Move `canonicalize_shapes` to `difflet/pipeline/bucketing.py`; `backends/trainium/core/bucketing.py` re-exports it; update `difflet_pipeline.py`, `compile_cache.py`, `models/*/entry.py`, `models/*/application.py`. Fix `_resolve_load_rank_range`.
- [x] **1.6** *(narrowed to `random.py` — ruling 1 in the Phase 1 status entry)* `difflet/utils/`: for each of the nine files, move the Neuron import inside the function that needs it or split the Neuron half into `difflet/backends/trainium/utils/`. Gate: `DIFFLET_BACKEND=cpu python -c "import difflet.utils.random, difflet.utils.accuracy, ..."`.
- [x] **1.7** `difflet/models/flux/entry.py` and any other `entry.py`: Trainium imports inside the `backend == "trainium"` branch. `scripts/test_imports.sh` imports entries, not Trainium applications.
- [x] **1.8** `pyproject.toml`: add `transformers`; `scripts/test_unit.sh` documents `PYTHON_BIN` for non-Neuron hosts.
- [x] **1.9** `compile_cache.py`: `"cuda"` in `_BACKEND_EXTRA_TOOLCHAIN_PACKAGES`; `torch.version.cuda` and cuDNN version in the key. Unit test: cuda and trainium specs differ; a trainium key is byte-identical to before.
- [ ] **1.10** *(entry written; CI has not run — branch is local)* Status log: new CPU-host baseline (target: 0 errors; remaining failures only trainium-only models, listed by name). CI green.

Exit: `DIFFLET_BACKEND=cpu ./scripts/check_quick.sh` green on the Mac and in CI.

### Phase 2 — Op surface at world size 1 (Mac for tests, box for the GPU smoke) — PR 3 begins

- [ ] **2.1** Lift (D10): create `difflet/backends/_torch_megatron/` from `backends/tpu/ops_impl/{linear,norm,embeddings}.py` and `backends/tpu/core/{weights,checkpoint}.py`, parameterised by a collectives module. Rewire TPU to import from it. **Run the full TPU unit set before and after; it must be identical.**
- [ ] **2.2** `cuda/ops_impl/collectives.py` + `parallel_mesh.py`: world-size-1 implementations (identity; `get_tp_size() == 1`). Multi-process paths raise `NotImplementedError("follow-up F2")`.
- [ ] **2.3** `cuda/ops_impl/{linear,norm,embeddings}.py`: bind `_torch_megatron` to the CUDA collectives.
- [ ] **2.4** `cuda/ops_impl/attention.py`: `attention`, `cross_attention` on SDPA honouring `scale`, `attention_mask`, `causal`; accept and ignore `tp_q/tp_k/tp_out`. `ring_*`, `ulysses_*`, `joint_*` raise naming F2/F6. Dedicated test for the mask (the DEVELOPER.md silently-dropped-mask lesson).
- [ ] **2.5** `cuda/ops_impl/platform.py`: `hardware` enum, `get_platform_target()`, compute capability, `distributed_backend_name() == "nccl"`, `sdpa_kernel_report()`. `mx.py`: all names raise (D9).
- [ ] **2.6** `tests/unit/backends/test_cuda_layers.py`, `test_cuda_attention.py`, `test_torch_megatron.py` — all run on CPU tensors. Flip the 1.4 xfails.

Exit: every `difflet.ops` export resolves on the CUDA backend; CPU-host tests pass; TPU tests unchanged.

### Phase 3 — Lifecycle without AOT (Mac for contract tests)

- [ ] **3.1** `cuda/core/application_base.py`: `CudaComponentApplication`. `compile(path)` validates config + shape, writes `manifest.json` (model id, dtype, torch/CUDA versions). `has_compiled_artifacts(path)` = manifest present and matching. `load(path, start_rank_id=None, local_ranks_size=None, skip_warmup=False)` builds on `meta`, loads safetensors shards onto the device. `__call__` runs the forward.
- [ ] **3.2** `tests/unit/backends/test_cuda_application.py`: contract names pinned with `inspect.signature`; manifest round-trip; tiny synthetic component `compile → has_compiled_artifacts → load → __call__` on CPU.
- [ ] **3.3** Docstring + Status log: what `has_compiled_artifacts` *means* for CUDA (manifest only), as TPU's docstring did for "graph exported, not compiled".

### Phase 4 — Qwen-Image on the box (GPU days 1–3) — PR 3 completes

- [ ] **4.1** `difflet/backends/cuda/qwen_image/config.py`, `transformer.py`, `difflet/models/qwen_image/cuda_application.py`: text encoder, DiT and VAE all on the GPU in one process (D8). `entry.py` branch; registry tuple; registration unit test.
- [ ] **4.2** `difflet download` Qwen-Image from Hugging Face on the box (D15). Record size and time.
- [ ] **4.3** `scripts/cuda_smoke.sh`: `DIFFLET_BACKEND=cuda difflet run --model-id Qwen/Qwen-Image --tp-degree 1 --prompt … --output /tmp/qwen.png`. Record wall time, peak memory, and the image path in the Status log. The image must be prompt-faithful by eye before any numeric gate is trusted.
- [ ] **4.4** `tests/numerical/cuda_parity_gpu.py` (D14): on the GPU, diffusers fp32 forward → diffusers bf16 forward → Difflet bf16 forward on the same seeded inputs; writes `cuda_parity_qwen_image.json`. `test_cuda_vs_diffusers.py` applies the TPU harness's three relative gates (≤ 1.15× the bf16-alone deviation; inside the control's distance from fp32; loose absolute floor). Record the numbers, including the residual, without loosening the gate.
- [ ] **4.5** Open PR 3 with the Status log entries for Phases 0, 2, 3, 4. Update `DEVELOPER.md` (CUDA quick start, backend section).

Exit (Release 1 done): Qwen-Image generates a prompt-faithful image on one Pro 6000 at `tp=1`, the diffusers-relative gates pass, three PRs merged or in review, numbers in the Status log.

---

## Follow-ups (scheduled, each gets its own Status-log entry when started)

- **F1 — Wan 2.2 on CUDA.** Same recipe as Phase 4. Both experts resident by default on 96 GB (`enable_transformer_2=True` on CUDA; the Trainium/TPU default stays False). Needs the CFG path.
- **F2 — Second card: DP and CFG-parallel (D5, D6).** `torchrun --nproc_per_node=2`; NCCL process groups from `MeshSpec.axis_groups()`; collectives on `torch.distributed`; CFG-parallel for Wan; `ulysses_attention` if a workload needs CP. TP across PCIe is **not** part of F2.
- **F3 — CLI and serving on CUDA.** `difflet/cli/runner.py` / `stage.py` map parallel config to `torchrun` + `CUDA_VISIBLE_DEVICES`; orchestrators and `difflet/serving/**` lose module-level Neuron imports; `difflet serve` for Qwen-Image; `scripts/verify_cli_cuda.py` column of the matrix.
- **F4 — Performance.** Three runs per change (warm-up untimed; clean timed with profilers off; separate profiled run with `nsys` and `ncu`). Then, each gated on that table and on the parity gate still passing: `torch.compile` (D4), TeaCache port, FP8 on Blackwell tensor cores (D9), SDPA-vs-`flash-attn` only if the profile names attention (D7). Custom Triton/CUTLASS kernels only for a bottleneck the profile names.
- **F5 — Flux.** Requires a Trainium verification path (D13; `trn2.3xlarge`). Split `modeling_flux.py` into neutral modeling + Trainium wrapper; `scripts/flux_smoke.sh` before and after, same cosine; then port with the Phase 4 recipe. Flux is guidance-distilled: no CFG-parallel.
- **F6 — Tensor parallel across PCIe (need-triggered, D6).** Implemented when a named shape fails to fit one card and after measuring against CP. The Status log records the triggering workload.
- **F7 — ROCm registration (D11).** `backends/rocm/` registers the shared torch-accelerator code with its own platform module. No new op code expected.
- **F8 — `difflet-cuda-verify` skill and docs.** CUDA sibling of `.claude/skills/difflet-device-verify/`, written from what F1–F4 actually ran; `docs/verification/` evidence report.

---

## Status log

Append-only. Each entry: date, phase, host (GPU, driver, CUDA, torch), what ran, numbers, verdict.

- **2026-10-16 — Plan drafted.** Baseline on a non-Neuron host (macOS, Python 3.14.7, torch 2.14.1, diffusers 0.38.0, transformers 5.20.1): `DIFFLET_BACKEND=cpu ./scripts/test_unit.sh` → 2088 passed, 131 failed, 36 errors, 43 skipped in ~20 s; `scripts/test_imports.sh` fails at `difflet.models.flux.application` (`neuronx_distributed` missing). Coupling: 24 files outside `backends/trainium` import the Neuron stack; ~35 import `difflet.backends.trainium`.
- **2026-10-16 — Plan reviewed with the owner; decisions D1–D19 recorded.** First draft assumed 2 × H100 80 GB, Qwen + Wan at tp=1/tp=2 as Release 1, and a host-side encoder/VAE split. Replaced by: one RTX Pro 6000 (96 GB, PCIe, no NVLink) from the Yottalabs console; Release 1 = Qwen-Image tp=1 only; everything on the GPU; TP need-triggered; three PRs; three-GPU-day timebox. GPU box not yet requested. Phase 1 (hygiene, Mac) can start now.
- **2026-10-16 — Phase 1 done (PR 2, local).** host: Mac (macOS, Python 3.14.7, torch 2.14.1, diffusers 0.38.0, transformers 5.20.1 — no Neuron SDK, no GPU). ran: `.claude/skills/difflet-cuda-backend/scripts/cpu-gate.sh` (import guard + `tests/unit`, `DIFFLET_BACKEND=cpu`). numbers: before 2088 passed / 131 failed / 36 errors / 43 skipped; after **2140 passed / 0 failed / 0 errors / 172 skipped / 9 xfailed**, import guard green, `known-red.txt` empty. Execution plan: `docs/superpowers/plans/2026-10-16-cuda-pr2-phase1-hygiene.md` (10 tasks, one commit each). rulings: (1) task 1.6 narrowed to `utils/random.py` — the NxDI-era utils (`accuracy`, `benchmark`, `constants`, `debug_utils`, `testing`, `tensor_capture_utils`, `version_utils`) have no importers and the Trainium adapters (`hf_adapter`, `diffusers_adapter`, `kv_cache_reconstruct_utils`, `tensor_replacement/registry`) are Trainium by nature; their tests skip under the Neuron gate instead. (2) `scripts/test_imports.sh` is two-tiered; `ltx_2.entry`/`.pipeline` and `qwen_image.pipeline` sit in the Trainium tier (the ltx_2 package `__init__` imports the Trainium application; `qwen_image.pipeline` imports `QwenImageDiTInputBundle` from the Trainium application — **Phase 4 must move that type to a neutral module**). (3) Neuron-only tests skip on hosts without the SDK by exact signature (`tests/_neuron_gate.py`, 26 ignored modules + 3 exception signatures); inactive on Neuron hosts. (4) `test_cpu_linear_mx_outer_n_reference_matches_logical_linear[512|2048]` compared cosine in fp32; the 0.999 gate sits ~4e-5 above the inherent MX-quantization cosine (0.99904) and an fp32 reduction over 262k elements eats that margin (0.99898 on torch 2.14 CPU). The test now computes the cosine in float64 (0.99903/0.99904); the CPU mx reference itself is exact (0.9998 vs an fp64 matmul of the dequantized operands). Review finding, verified. (5) `.gitignore` also tracks `docs/superpowers/plans/*.md`. (6) `_resolve_load_rank_range` in `difflet_pipeline.py` has no production caller (`_load_app` already used the backend object); task 1.5's fix there changed test-only code — the real load path was never Trainium-hard-coded. (7) The Neuron gate records every converted test id; the CPU gate diffs them against `neuron-skips.txt` so a *new* Neuron-only skip fails the gate (review finding: signature drift). deferred minor: `tests/unit/serving/test_video_service.py::test_retention_sweep_clears_active_item_job_slot_ownership` flaked once in a full run (asyncio shielded-future timing), passes alone. next: Phase 2 — task 2.1 (lift TPU Megatron layers into `difflet/backends/_torch_megatron/`); GPU box still not requested.
- **2026-10-16 — Phase 0 verdict (PR 3 Task 8).** host: Yottalabs pod `difflet-cuda-qwen` (ID 502004828165112280, us-central-7, $1.35/h), Ubuntu 22.04.5, kernel 6.8.0-90, 16 vCPU, 141 GB RAM, 713 GB disk; **NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 580.126.09, CUDA 13.0 driver, compute capability (12, 0)**. The image's own torch is a nightly (2.12.0.dev20260311+cu128) and was not used; fresh uv venv, Python 3.11.14, **torch 2.11.0+cu128**, cuDNN 9.19, triton 3.6.0, diffusers 0.38.0, transformers 5.20.1 (`requirements-cuda.lock`, `cuda` extra). ran: `scripts/cuda_env_check.sh`; 53 CUDA unit tests on the real GPU (application tests at device=cuda, Review Focus 4 proven); the CPU gate on the box (2246 passed; 2 Linux-only failures in `tests/unit/serving/test_video_storage.py`, overlayfs inode semantics, pre-existing, not CUDA); `hf download Qwen/Qwen-Image` 54 GB in ~93 s. numbers: bf16 matmul ok; SDPA kernel for Qwen-Image's shape (24 heads, 5120 tokens, d=128, bf16): **flash for 4-D (B,H,S,D) but math for the 3-D (B*H,S,D) layout the model code uses** → the CUDA attention now lifts 3-D inputs to 4-D before SDPA (1.2 ms per call, 0.06 GB peak after the fix). Capability triple confirmed as the stub declared. rulings: released wheel over the image's nightly; gate flags fixed (`-rs` → `-rfEs`, failures were invisible to the regression diff). next: Task 9 smoke.
