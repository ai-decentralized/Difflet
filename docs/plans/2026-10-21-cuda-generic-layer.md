# CUDA Generic Layer Design and Migration Plan

> **For agentic workers:** this is a design doc first and a plan second. Nothing here is code yet. The migration phases at the end use checkbox (`- [ ]`) syntax and the one-task-one-commit convention, but no phase starts until the owner has settled the decisions table and the open questions, and the reviewer has signed off on the layering. Every phase ends with the CPU unit suite green and a status-log entry.

**Goal:** Replace "one CUDA stack per model" with one generic CUDA transformer lifecycle, one generic CUDA model application, one generic serving adapter and one generic CLI orchestrator, each driven by a small per-model spec, so that a fourth CUDA model costs a spec plus its real differences and a ROCm backend costs a platform file, not three more stacks.

**Architecture:** Five levels, from the op surface up. Levels 1 and 2 live in `difflet.ops` and are backend-neutral. Level 3 (model-and-hardware fusions) stays under `difflet/backends/cuda/<model>/` as load-time module swaps, the pattern `difflet/backends/cuda/flux/proj_out.py` already sets. Level 4 (one `torch.compile` helper) and Level 5 (generic application, serving and CLI) move the shared flow into `difflet/backends/cuda/core/`, `difflet/serving/orchestrators/cuda.py` and `difflet/cli/orchestrators/cuda.py`. Each model keeps a torch-free spec next to its registry metadata, a transformer subclass of about fifty lines, and a driver that owns the parts no generic code should guess at.

**Tech Stack:** unchanged from `docs/plans/2026-10-16-cuda-backend-support.md`: Python ≥ 3.10, torch (CUDA 12.8+ wheel), `diffusers==0.38.0`, `safetensors`, `pytest`; `torch.distributed` + NCCL + `torchrun` for CFG-parallel and data-parallel replicas. No new dependency.

**Spec:** the reviewer's architecture comment on the Qwen-Image CUDA PR (ai-decentralized/Difflet#84, 2026-10-21, "Architecture: one stack per model × backend"), reproduced in the next section. The merge condition it set (the Flux fusion out of `modeling_flux.py`) is met on the current stack; this document is the agreed follow-up: design before code, and no further CUDA model until the generic layer exists. The comment points to vLLM-Omni's RFC on kernel reuse and hardware integration (vllm-project/vllm-omni#7382) as the model for levels 1 to 3; the next-but-one section says what this design takes from it.

---

## The brief, as given

| Level | Example | Today | Proposal |
|---|---|---|---|
| 1. Base ops | attention, rms_norm | `difflet.ops` + `ops_impl` | Keep. Turn the reference implementation plus one tolerance table into a test every backend must pass. |
| 2. Cross-model fusion | qk_norm + rope | none | Add to `difflet.ops` once two or more models use it. |
| 3. Model/hardware-specific fusion | Flux fused `proj_out` | a load-time swap under `backends/cuda/flux/` | Keep that pattern; shared model files stay untouched. |
| 4. Compile | per-block `torch.compile` | copied 3× | One helper, each model declaring its repeated blocks. |
| 5. Load / serve | — | one stack per model | One generic CUDA application and orchestrator; each model supplies a small spec (pipeline class, output kind, smoke request, validation) plus its real differences. |

The real differences the brief names: rope from ids and the split `proj_out` for Flux; the prompt trim for Qwen-Image; two experts, CFG-parallel, the fp32 timestep and video output for Wan.

---

## What this design takes from vLLM-Omni RFC #7382

The RFC (open, first phase under way) is about where kernels and hardware-specific code live and how each reuse is proven. It covers this design's levels 1 to 3. It says nothing about levels 4 and 5: vLLM-Omni already has one model runner, so a generic per-backend application, serving adapter and CLI orchestrator are Difflet's own problem.

Already consistent with it: shared operations on one public surface, model-specific computation with its model, existing dispatch reused and no new runtime registry (here `difflet.ops`); load-time module replacement as a valid integration pattern (Level 3); moving code by re-exporting first, then moving, keeping the old import as a forwarding shim (G6). Its rejected alternative "copy a vendor directory tree into every model" is the reviewer's critique of the current per-model CUDA stacks.

Adopted from it, each folded into the section named:

| From the RFC | In this design |
|---|---|
| Layout and arithmetic variants of a "same" fusion are explicit, and a consumer keeps its own reference chain unless the shared op is proven to reproduce it | Level 2 (the op's signature and its preconditions) |
| The model decides whether to use a fusion; the operation decides which implementation runs for valid inputs | G12; Levels 2 and 3 |
| A caller that always takes the reference path proves nothing about the fast path: tests assert the selected path, not only the output | G11; Levels 1 to 3; shared tests |
| A hardware-specific block is accepted on checkpoint keys, non-trivial weights, the selected class, block parity and then a real model run | Level 3, "Acceptance for a swap" |
| Contracts are per operation, not one universal tolerance, and each change carries a pinned evidence record with unrun cells left "pending" | G7; Level 1, "The ops index" |
| Moves preserve behaviour; a change to defaults, errors or compile behaviour is reviewed on its own, and a move is checked for performance non-regression | G10; Phase 2; Risks |

---

## What the three stacks look like today (measured 2026-10-21 at `c45d2059`)

The twelve production files, raw lines:

| File | Qwen-Image | Flux | Wan | Shared today |
|---|---|---|---|---|
| `difflet/backends/cuda/<model>/transformer.py` | 148 | 240 | 171 | `backends/cuda/core/application_base.py` (210) |
| `difflet/models/<model>/cuda_application.py` | 308 | 256 | 442 | — |
| `difflet/serving/orchestrators/<model>_cuda.py` | 325 | 304 | 387 | `serving/orchestrators/cuda_resident.py` (107) |
| `difflet/cli/orchestrators/<model>_cuda.py` | 369 | 57 | 109 | `CudaOneProcessOrchestrator`, which lives *inside* `qwen_image_cuda.py` |
| **Per model** | **1,150** | **857** | **1,109** | **3,116 total** |

Alongside them, the GPU scripts the brief also counts: six end-to-end and bench scripts under `scripts/cuda_*` (1,801 lines) and three parity harnesses under `tests/numerical/cuda_parity*_gpu.py`, one of each per model.

**How much is the same.** Measured with `difflib.SequenceMatcher` over each file's non-blank, non-comment lines after normalising the model names (`Qwen-Image`/`Flux`/`Wan` and their Hub ids to one token); "matched" is the number of lines in a longest-common-subsequence match, as a share of each file.

| Pair | transformer.py | cuda_application.py | serving `<m>_cuda.py` | cli `<m>_cuda.py` |
|---|---|---|---|---|
| Qwen-Image vs Flux | 55% / 30% | 46% / 56% | **82% / 85%** | 5% / 37% |
| Qwen-Image vs Wan | 41% / 33% | 37% / 27% | **65% / 52%** | 7% / 23% |
| Flux vs Wan | 20% / 29% | 47% / 28% | **72% / 55%** | 65% / 32% |

Read: "82% / 85%" means 82% of the Qwen serving adapter's code lines reappear in the Flux one and 85% of the Flux adapter's reappear in the Qwen one. The reviewer's 67–77% for the serving adapters is the same picture under a different normalisation. The CLI column looks low only because the generic class already exists: `CudaOneProcessOrchestrator` (about 250 of the 298 code lines in `difflet/cli/orchestrators/qwen_image_cuda.py`) is the shared flow, and the Flux and Wan orchestrators are already 43- and 87-line specialisations of it. The CLI level is therefore mostly a *move*, not a rewrite.

**What is literally copied.** Three things recur verbatim or near-verbatim:

1. The `_prepare_module` body in each `transformer.py`: build on meta, `materialize_meta_`, load weights, `.to(device)`, `requires_grad_(False)`, `.eval()`, `_assert_on_device`, then the per-block `torch.compile` loop. What differs: the weight-loading call (three loaders, below), the compile `dynamic` flag, Flux's swap loop and second block list, and Wan passing `tp_size=1, tp_rank=0` where Qwen reads the mesh.
2. In each `cuda_application.py`: the `DiffletPipeline` contract trio `compile`/`load`/`has_compiled_artifacts`, `_require_transformer` with its "transformer/config.json" message, `dit_input_contract`, and the `__call__` dispatcher. Wan's trio additionally loops over its named experts; Flux and Qwen also share a `forward_dit` over an input-bundle type (`difflet/models/<model>/contract.py`), which Wan does not have.
3. The four serving classes per model: `*ServingArtifactPreparer.prepare_runtime`, `*ServingRequestValidator` (tokenizer preload, shape-equals-profile, prompt-token budget), `*PipelineRunner.execute`, and `*ServingStageAdapter` (`create_loaded_runners`, `generate_*`, `smoke_request`, `validate_smoke_output`, `shutdown`). Everything except the request contract and the output encoding is the same text with the model name swapped.

**What is genuinely different, and must stay per model.**

| | Qwen-Image | Flux | Wan 2.2 |
|---|---|---|---|
| Hub ids the CLI routes | `Qwen/Qwen-Image` | FLUX.1-dev, FLUX.1-schnell | Wan2.2-T2V-A14B, Wan2.1-T2V-14B |
| Checkpoints `serve` accepts | the same | FLUX.1-dev only | Wan2.2 only (Wan2.1 never ran on a GPU) |
| Denoise driver | Difflet's `QwenImageOrchestrator` (`difflet/models/qwen_image/pipeline.py`), text encoder and VAE loaded by Difflet | diffusers' own `FluxPipeline`, with `FluxDiTAdapter` standing in for its transformer | Difflet's `WanOrchestrator` (`difflet/models/wan/pipeline.py`) with host-side fp32 latents and three device callables |
| Weight loading | `load_checkpoint_into(prefix="transformer.")` reading the mesh | `build_weight_map` + `convert_flux_hf_state_dict` (splits `proj_out`) | `load_checkpoint_into(rename=checkpoint_key, tp_size=1)`, once per expert |
| Module build | `_QwenImageTransformerTraceModule(apply_text_mask=True, trim_text_padding=True)` | `NeuronFluxTransformer2DModel` | `WanTransformer3DModel(dtype=...)`, buffers computed in `__init__` |
| Experts | one | one | two, selected by `boundary_ratio`; artifact subdirectory per expert; partial-download refusal at construction |
| Parallel mesh | `tp=1` | `tp=1` | `tp=1`, `cfg=2` under CFG-parallel; refuses to load without a world-2 process group |
| Compile | per block, `dynamic=True` (prompt length varies), warm-up twice | per block, `dynamic=False`, two block lists | per block, `dynamic=False`, per expert |
| Rope | real-valued tables inside the modeling (`_apply_qwen_rope_real`) | computed from `txt_ids`/`img_ids` on the host by the adapter, cached per ids | 3-D tables as buffers, `difflet.ops.apply_rotary_emb` |
| q/k norm | per-head `CustomRMSNorm` | per-head `CustomRMSNorm` | `RMSNorm` across heads over the full `inner_dim` |
| Input bundle | `QwenImageDiTInputBundle`, `forward_dit` | `FluxDiTInputBundle`, `forward_dit` | none; `__call__` is `prompt=` or raw tensors |
| Timestep dtype | model dtype | model dtype | fp32 by default (`timestep_dtype` kwarg) |
| Guidance | refused (`guidance_embeds=False`); the serving validator pins a no-op 4.0; the CLI refuses `--guidance-scale` with a "has no effect" message | distilled `guidance_scale`, one pass | true CFG, `guidance_scale` and `guidance_scale_2`, `negative_prompt` |
| Output | PNG | PNG | `(B, 3, F, H, W)` tensor → MP4; `generate` defaults to `output_type="pt"`, refuses a batch of prompts and a `num_frames` other than the loaded one |
| Application kwargs refused | the four probe-mode TeaCache keys, by `is not None`; `shapes` is *not* refused today | `taef1`, `taef1_path`, `shapes`, `sp_enabled`, `teacache_*`, when not None/False | `shapes`, `teacache_*`, when not None/False |
| CLI flags refused | `--shapes`, `--teacache-speedup`, `--teacache-online-delta`, `--guidance-scale` | `--shapes`, the three `--teacache-*` modes, `--teacache-calibration`, `--taef1/--taef1-path`, `--host-vae` | same as Flux |
| CLI flags forwarded as application kwargs | `torch_compile`, `teacache_cadence` | `torch_compile` | `torch_compile` |
| Programmatic-only application kwargs | `text_seq_len`, `batch_size`, `load_text_encoder`, `load_vae`, `device` | `load_text_encoders` (plural), `load_vae`, `local_files_only`, `fuse_proj_out`, `device` | `text_seq_len`, `batch_size`, `load_text_encoder`, `load_vae`, `timestep_dtype`, `enable_transformer_2`, injected `text_encoder`/`vae_decoder`, `device` |
| Prompt budget | 1024 DiT tokens after a 34-token template prefix (`tokenizer`) | 512 T5 tokens (`tokenizer_2`) | 512 UMT5 tokens for prompt and negative prompt (`tokenizer`) |
| Serving-only | — | — | three refused video fields (`boundary_ratio`, `flow_shift`, `true_cfg_scale`), fps equals profile |

This table is the content of the per-model specs and drivers. Everything not in it is a candidate for the generic layer.

---

## Decisions (proposed; to be settled with the owner and the reviewer before Phase 1)

| # | Decision | Alternative rejected | Why |
|---|---|---|---|
| G1 | **Five levels as in the brief, implemented bottom-up except Level 2, which comes last and on its own evidence.** | Build Level 5 first because it removes the most lines | Levels 1, 3 and 4 are the safety net Level 5's refactor runs under. Level 2 changes modeling files the Trainium graphs trace, so it needs the Trainium verification path and a measured win, and must not block the rest. |
| G2 | **The generic layer stays under the `cuda` name.** `difflet/backends/cuda/core/` gains `compile.py` and `application.py`; the orchestrators become `difflet/serving/orchestrators/cuda.py` and `difflet/cli/orchestrators/cuda.py`. ROCm registers its own package that re-exports them (D11 of the 2026-10-16 plan). | A new `difflet/backends/torch_accel/` package now | A rename is churn with no user; the ROCm section lists the fourteen code literals that would have to go through `platform.py` first. Do that when ROCm is scheduled. |
| G3 | **The spec is torch-free data; behaviour lives in a driver.** `CudaModelSpec` and `PromptBudget` are frozen dataclasses in `difflet/common/cuda_spec.py`; the three instances sit in `difflet/common/registry/<model>.py` next to `cuda_serving_metadata()`, which is generated from them. The driver is named by import string and resolved only in the worker. | Spec under `difflet/backends/cuda/<model>/` | `difflet/backends/cuda/__init__.py` imports `runtime`, which imports torch. The serving parent resolves the preparer and validator factories and calls `cuda_serving_metadata()` itself, and every adapter today defers even `cuda_pipeline_kwargs` to the worker ("the parent process never loads torch for this adapter"). A spec under the backend package would repeal that rule silently. A field goes in the spec when at least two generic call sites read it; otherwise it is the driver's. |
| G4 | **The transformer subclasses stay; the base class owns the lifecycle.** `CudaApplicationBase._prepare_module` becomes the one implementation (materialise → `load_weights` → device → `requires_grad_(False)` → eval → assert → swaps → compile), with `load_weights(module)`, `repeated_blocks(module)` and `apply_load_time_swaps(module)` as the hooks a model implements. | A single `CudaTransformerApplication(spec)` with loader and block-list callables in the spec | A fifty-line subclass is easier to read than a spec with six callables; it is also what the TPU backend does, so the Trainium → TPU → CUDA reading order stays the same. |
| G5 | **Acceptance for every Level 5 phase is byte-identical output per model at a fixed seed against the pre-refactor build**, on `run` and `serve`, eager. Compiled mode joins the gate only once Phase 2's first task has shown the compiled md5 is stable run to run; otherwise compiled mode is gated on parity against eager within a recorded tolerance. | Parity against diffusers only | The stack already proved eager determinism across cards and across `run`/`serve` (`docs/verification/cuda-release1-evidence.md`); a refactor that is byte-identical to itself has nothing to re-prove upstream. The compiled path's run-to-run identity was never measured: `max-autotune` picks kernels by benchmarking, so it is measured first, not assumed. |
| G6 | **Per-model unit tests are kept during migration through re-export shims and deleted with the shims in the last phase.** New shared tests are parametrised over the three specs. | Rewrite the tests as each file moves | 252 CUDA tests name today's classes and module paths (census below); moving them in the same commit as the code doubles every review. Shims keep each phase's diff to its subject. |
| G7 | **Level 1's tolerance table is one test file with one table; each row names its reference and says whether it must match exactly or within a tolerance; bf16 entries are measured on the GPU in Phase 1, not invented here.** fp32 entries are the ones today's tests already use (`atol=1e-5` attention, `1e-6` linear/norm). `difflet/ops/README.md` lists every op, and gives every fusion and every op with more than one implementation on a backend a written contract. | Guess bf16 tolerances in the doc; one tolerance for every op | A tolerance nobody measured is either so loose it catches nothing or so tight it fails on the first card with a different reduction order. An op that must reproduce a model's rounding chain exactly and an op that may add bf16 noise are different contracts; one number for both hides one of them. |
| G8 | **The GPU scripts (`scripts/cuda_e2e_*`, `scripts/cuda_bench_*`, `tests/numerical/cuda_parity*_gpu.py`) are out of scope here** beyond a note on what they would become. | Fold them into the same series | They are evidence tooling, not production code; the brief lists them but asks for the production layering. They collapse once the generic application exists (one script taking `--model`), and that is a later, smaller PR. |
| G9 | **Defaults have one owner.** Shape comes from `ModelEntry.default_shape` in `difflet/registry.py` (already the registry's); steps, guidance and fps live in the spec only, and the CLI, serving metadata and application read them from there. | A spec copy of everything | Today the shape is in the registry and in each CLI/serving/application file, and steps/guidance in three places per model; a spec that restates the shape would be a fourth copy. |
| G10 | **Refusals are two lists, not one, and this series reproduces today's lists exactly.** `refused_cli_flags` and `refused_application_kwargs`, each `(name, reason)`, because the two namespaces differ (`--teacache-online-delta` vs `teacache_online_delta_alpha`; `--host-vae` is never an application kwarg). Qwen-Image's application does not refuse `shapes=` today and its spec keeps it that way; making it refuse `shapes=` like Flux and Wan is a behaviour change with its own small PR after Phase 5. | One list applied to both; fixing the Qwen-Image gap inside the refactor | A flag spelled differently from its kwarg would be silently un-refused on one side. A behaviour change inside a move cannot be reviewed as one, and the move's acceptance (G5) is that nothing changes. |
| G11 | **A test that claims a fast path asserts that the fast path ran.** Attention asserts the SDPA kernel chosen for the shapes the models use (flash, cuDNN or efficient, never `math`); a Level 3 swap reports what it applied and its test asserts the swapped forward executed; a Level 2 consumer test asserts the fused path for the inputs it means to fuse and the model's own chain for the rest. | Compare outputs only | A caller that always falls back passes every output test and proves nothing about the fused code. The 3-D-to-4-D lift in `ops_impl/attention.py` exists because SDPA was silently choosing the math kernel. |
| G12 | **For a fusion, the model decides whether; the op or swap decides which.** The model owns enablement and policy (a constructor option, a size threshold); the op or swap owns implementation choice for valid inputs and its own reference. | The op decides everything from its inputs | The same fused kernel can win on large inputs and lose on small ones. In vLLM-Omni's measurements one fused QK/RoPE made the smallest request about 4.7% slower on an A100, so that model enables it only above a token threshold: model policy, not op dispatch. |

---

## Level 1 — Base ops: keep, and add the tolerance table

**Today.** `difflet/ops/__init__.py` lists 41 exports; each backend implements them under `difflet/backends/<backend>/ops_impl/`. The CPU implementation is the oracle. CUDA's tests (`tests/unit/backends/test_cuda_attention.py`, `test_cuda_layers.py`, `test_cuda_rms_norm_module.py`) compare CUDA to CPU op by op with hand-picked `atol` values; TPU has its own file; nothing says which tolerance an op is held to, and a new backend would start from zero tests.

**What moves.** Nothing moves. One test file is added: `tests/unit/ops/test_backend_tolerance.py`. It holds one table and one parametrised test:

```python
# (op, dtype, atol, rtol, note)
TOLERANCE = [
    ("attention",            torch.float32,  1e-5, 0,    "SDPA vs the CPU softmax"),
    ("attention",            torch.bfloat16, None, None, "measured in Phase 1"),
    ("cross_attention",      torch.float32,  1e-5, 0,    ""),
    ("RMSNorm",              torch.float32,  1e-6, 0,    "fused F.rms_norm vs the seven-pass reference"),
    ("RMSNorm",              torch.bfloat16, None, None, "measured in Phase 1"),
    ("rms_norm_fp32",        torch.float32,  1e-6, 0,    ""),
    ("LayerNorm",            torch.float32,  1e-6, 0,    ""),
    ("apply_rotary_emb",     torch.float32,  1e-6, 0,    "pure torch, shared with TPU"),
    ("ColumnParallelLinear", torch.float32,  1e-6, 0,    "world size 1"),
    ("RowParallelLinear",    torch.float32,  1e-6, 0,    "world size 1"),
    ("ParallelEmbedding",    torch.float32,  1e-6, 0,    ""),
    ("quantize_mx",          torch.float32,  None, None, "raises NotImplementedError on cuda: xfail(strict)"),
]
```

The test is parametrised over the backends importable on the host (`cpu` always; `cuda` when `torch.cuda.is_available()`; `tpu` when `torch_xla` and `libtpu` import) × the table rows. Each row builds the op's inputs once from a fixed seed with a small shape-set per op (the shapes today's per-backend tests use, so nothing gets looser), runs the CPU reference in fp32, runs the backend in the row's dtype, and asserts `allclose` against the reference cast back to fp32. A `None` tolerance is `xfail(strict=True)`: the row exists so the gap is visible, and filling it flips the test.

**Where the bf16 numbers come from.** Phase 1 runs the table on the GPU once with tolerances at `inf`, records the max abs and rel error per row in the status log, and sets each tolerance to the smallest power of ten above the measured error. The ring and Ulysses attention modes are not in the table: CUDA raises for them, and the table should say what a backend must pass, not what it may skip.

**Exact or tolerance, and which path ran (G7, G11).** Each row names its reference and its kind: `exact` rows (a pure move, or an op that must reproduce a fixed rounding chain) compare with `torch.equal` and also assert dtype, shape and strides, since `torch.equal` checks neither dtype nor NaN placement; `tolerance` rows use the row's `atol`/`rtol`. Rows for ops with more than one implementation on a backend also assert which one ran: the CUDA attention rows record the SDPA kernel through `platform.sdpa_kernel_report` and fail if a model shape lands on `math`.

**The ops index.** `difflet/ops/README.md` lists every op on the surface in one table (op, backends that implement it, reference, exact or tolerance, consumers). Every fusion, and every op with more than one implementation on a backend, also gets a five-part contract:

1. tensor shapes, layouts and dtypes, and the numerics, including intermediate casts and where rounding happens;
2. the implementations per backend and when each is selected;
3. the reference and the error behaviour for unsupported and invalid inputs;
4. the public signature;
5. the consumers, each with its own policy for using it (G12).

Each qualification run adds a record line under its op: commit, device and software versions, shapes and dtype, the selected path, the baseline, the criterion and the result. A cell nobody has run says "pending"; a run on one device does not qualify another. A unit test asserts that every name `difflet.ops` exports appears in the index.

**What each model still supplies.** Nothing. Models do not appear at this level.

**What a new backend writes.** Its `ops_impl/` package. The table then tells it exactly which op it got wrong and by how much.

---

## Level 2 — Cross-model fusions in `difflet.ops`

**Today.** None. The brief's example, q/k norm followed by rope, is done four ways across Difflet's models, and they differ in exactly the dimensions a shared op has to state:

| Model | q/k norm | Rope layout | Rope arithmetic |
|---|---|---|---|
| Flux (`difflet/models/flux/modeling_flux.py:1240-1255`) | per head, `CustomRMSNorm` | interleaved pairs; `(S, D, 2)` table; `difflet.layers.embeddings.apply_rotary_emb` | fp32, cast back once |
| Qwen-Image (`difflet/models/qwen_image/modeling_qwen_image.py:57-66, 327-343`) | per head, `CustomRMSNorm` | interleaved pairs; repeat-interleaved `(cos, sin)`; `_apply_qwen_rope_real` | fp32, cast back once |
| Wan (`difflet/models/wan/modeling_wan.py:532-542`) | across all heads, over the full `inner_dim` | interleaved pairs; `difflet.ops.apply_rotary_emb(x, cos, sin)` | the tables' dtype |
| MiniMax-H3, Trainium only (`difflet/models/minimax_h3/modeling_minimax_h3.py:55-71`) | per head, `nn.RMSNorm` | first and second half, over the leading `rotary_dim` channels only | the input dtype |

"Two models use it" holds for Flux and Qwen-Image only, and even those two reach their rotation through different code. vLLM-Omni's RFC found the same in its own models: its fused QK/RoPE first served a half-split model, its second consumer needed an interleaved variant, and that consumer keeps its own reference chain because the op's generic fallback rounds differently from it.

**Proposal.** One op, additive to the frozen surface, with the variants in its signature:

```python
# difflet/ops/attention.py
def qk_norm_rope(q, k, *, norm_q_weight, norm_k_weight, eps: float, cos, sin,
                 interleaved: bool, rotary_dim: int | None = None):
    """Per-head RMSNorm of q and k with fp32 statistics, then rotation of the leading
    rotary_dim channels (all when None) as interleaved pairs or as first/second half.
    q, k are (B, H, S, D); cos, sin are (S, rotary_dim). The contract in difflet/ops/README.md
    states where each implementation rounds."""
```

The op owns its reference implementation and its dispatch for valid inputs. It does not own any model's numerics: each consumer keeps its current chain as its reference and switches to the op only after a test on each backend shows the op reproduces that chain bit for bit at the model's shapes and dtype. Where it does not, the model keeps its own code; Wan's across-heads norm and a rope computed in a different dtype are two such cases. Whether a model uses the op at all is the model's decision (G12). A CUDA fused kernel then sits behind the op for the inputs it supports, with its own tolerance row and path assertions (G11).

**Preconditions, none met yet.**

1. Flux's and Qwen-Image's modeling would call the op, and both files are traced by Neuron. A Trainium verification run must show the traced graph unchanged; the 2026-10-16 plan deferred this kind of Flux modeling change until such a path exists (its D13).
2. A transformer bench on Flux or Qwen-Image shows the fused kernel winning at the shapes those models run, including the smallest request shape.
3. Each consumer has its bit-exactness test against its own chain and its path assertions.

So Level 2 is scheduled after Level 5, in that order of preconditions. If the win is not there, the op is not added; the layering stands without it.

**What each model still supplies.** Its decision to use the op, its tables in the op's layout, and its own reference chain. Wan and MiniMax-H3 do not use the op in this series.

---

## Level 3 — Model-and-hardware fusions as load-time module swaps

**Today.** `difflet/backends/cuda/flux/proj_out.py` is the example: `fuse_proj_out(block) -> bool` rebuilds diffusers' single `proj_out` from the split weights at tp=1 and binds a fused forward onto the block with `types.MethodType`. It is off by default, opt-in with `fuse_proj_out=True`, and `CudaFluxTransformerApplication._prepare_module` calls it over `single_transformer_blocks` *before* compile so the graph sees the fused layer. The shared `modeling_flux.py` is untouched; Trainium never sees the swap.

**What stays.** The pattern, exactly: a function per swap under `difflet/backends/cuda/<model>/`, returning `True` when it swapped and `False` when it must not (tp > 1, already fused), never editing a shared model file, off by default until measured. The `proj_out` swap keeps its "until the 10 dB PSNR move on one seed is explained, the split path ships" note; nothing here changes that verdict.

**What moves.** The *call site*. Today the Flux transformer application loops over its blocks and calls the swap inline. In the generic lifecycle (Level 4/5) the base class calls one hook after the weights are on the device and before compile:

```python
# difflet/backends/cuda/core/application_base.py
def apply_load_time_swaps(self, module) -> list[tuple[str, int]]:
    """Apply this model's enabled swaps in place; return (swap name, blocks swapped)
    pairs. The default applies nothing and returns []."""
    return []
```

Flux's subclass applies `fuse_proj_out` over `single_transformer_blocks` when `fuse_proj_out=True` and returns `[("fuse_proj_out", n)]`; the base logs the pairs so a parity run's record says which swaps were live. The ordering guarantee (swaps before compile) is then the base class's, tested once.

**Who decides (G12).** The swap's option is the model's enablement, off by default; the swap function decides only whether it can apply to a given block (tp > 1, already fused) and reports what it did.

**Acceptance for a swap.** The same four obligations vLLM-Omni's RFC sets for a hardware-specific block:

1. *Construction and weights.* The swap runs after `load_weights`, so checkpoint keys and strict loading are untouched; a test loads the same non-trivial weights into a swapped and an unswapped block. Today's `test_split_state_dict_still_loads_strict_and_the_fused_block_has_no_split_keys` and the bit-equal fused-weight check already cover this for `proj_out`.
2. *Block parity.* The swapped block is compared with the unswapped block and with diffusers' own block on the same inputs, with weights that make every branch observable, in the dtype and at the shapes the model runs. For `proj_out` this is the missing piece: today's comparison is a tiny fp32 model on the CPU within `1e-4`, while the end-to-end shift was seen in bf16 on the GPU. The reviewer asked for exactly this block-level check against diffusers before any noise-floor explanation of the shift is accepted.
3. *Selection.* `apply_load_time_swaps` returns what it applied, and the test asserts that the swapped forward ran, not only that the module was replaced (G11).
4. *Model.* The parity harness against diffusers and the byte-identity check of G5, recorded separately with the swap off (the default) and on.

**What each model still supplies.** Its swap functions and the option that enables each. A model with no swap implements nothing.

---

## Level 4 — One `torch.compile` helper

**Today.** The same six-line loop in three files, with two differences that matter: Qwen compiles with `dynamic=True` because the prompt trim changes the text length per request (`dynamic=False` hit the recompile limit, `dynamic=None` compiled a third graph; both recorded in the Qwen file's comment and pinned by `tests/unit/models/qwen_image/test_qwen_text_mask.py::test_compiled_blocks_do_not_recompile_per_prompt_length`), while Flux and Wan compile with `dynamic=False` (one static shape per process). Qwen also warms up twice under compile because dynamo's guard on a `None` block argument errors once. Flux has two block lists; Wan compiles each expert separately, which falls out of "one application per expert".

**Proposal.** `difflet/backends/cuda/core/compile.py`:

```python
COMPILE_MODE = "max-autotune-no-cudagraphs"

def compile_blocks(block_lists: Iterable[nn.ModuleList], *, dynamic: bool) -> int:
    """Replace every block in every list *in place* with torch.compile(block,
    mode=COMPILE_MODE, dynamic=dynamic, fullgraph=False); return the number compiled.
    In place is an invariant: the existing tests index the module's own lists and
    assert OptimizedModule."""
```

and three hooks on the base class, which a model overrides:

```python
def repeated_blocks(self, module) -> list[nn.ModuleList]: ...   # required when torch_compile=True
compile_dynamic: bool = False                                    # Qwen sets True
warmup_passes_when_compiled: int = 1                             # Qwen sets 2
```

The base's `_prepare_module` calls `compile_blocks(self.repeated_blocks(module), dynamic=self.compile_dynamic)` when `torch_compile` is set, after the Level 3 swaps. `self.module` stays the eager module around the compiled blocks, as today, so the parity scripts and the prompt-trim host sync are unaffected.

**The base lifecycle in full**, so that the ten existing `test_cuda_application.py` tests (which subclass with `build_module` and a `get_state_dict`, never a `load_weights`) keep passing:

```python
def _prepare_module(self):
    module = self.build_module()
    if any meta parameter or buffer: materialize_meta_(module, device=self.device, dtype=self.dtype)
    self.load_weights(module)            # default: today's get_state_dict() → load_sharded_state_dict path
    module.to(self.device); module.requires_grad_(False); module.eval()
    self._assert_on_device(module)
    swaps = self.apply_load_time_swaps(module)
    if self.torch_compile: compile_blocks(self.repeated_blocks(module), dynamic=self.compile_dynamic)
    self.module = module
```

`requires_grad_(False)` moves into the base (all three subclasses call it; `test_cuda_wan_transformer.py` asserts it).

**What each model still supplies.** `load_weights` (the three loaders in the table), `repeated_blocks`: Qwen returns `[module.transformer.transformer_blocks]`, Flux `[module.transformer_blocks, module.single_transformer_blocks]`, Wan `[module.blocks]`; plus the two flags where they differ from the default.

**Tests.** The three existing "compile wraps every block" tests (`test_torch_compile_wraps_every_flux_block`, `test_torch_compile_reaches_both_experts`, Qwen's recompile test) keep passing unchanged through Phase 2 and are then joined by one parametrised test over the three subclasses asserting `compile_blocks` was called with exactly the declared lists and the declared `dynamic` flag.

---

## Level 5 — One generic application, one serving adapter, one CLI orchestrator

### The spec

`difflet/common/cuda_spec.py`, importable without torch (the serving parent imports it):

```python
@dataclass(frozen=True)
class PromptBudget:
    tokenizer_subdir: str                 # "tokenizer" | "tokenizer_2"
    max_tokens: int                       # 1024 | 512
    template: str | None = None           # Qwen's chat template, formatted with the prompt
    drop_prefix_tokens: int = 0           # Qwen's 34
    applies_to_negative_prompt: bool = False   # Wan

@dataclass(frozen=True)
class CudaModelSpec:
    model_type: str                              # "qwen_image" | "flux" | "wan"; tags derive from it:
                                                 # "[flux/cuda]" (CLI), "[flux/cuda serve]" (serving)
    cli_model_ids: tuple[str, ...]               # routed by `difflet run`; the first is the default
    served_checkpoint_ids: tuple[str, ...]       # what `difflet serve` accepts; a subset
    allocation_id: str                           # "qwen-cuda" | "flux-cuda" | "wan-cuda" (asserted by tests)
    output: Literal["image", "video"]
    chat_content_type: str                       # "image_url" | "video_url"
    default_steps: int                           # 50 | 28 | 40
    default_guidance_scale: float | None         # None: the model takes none; `--guidance-scale`
                                                 # and a request guidance are refused (Qwen)
    serving_noop_guidance_scale: float | None    # Qwen's 4.0: the only value a request may carry,
                                                 # and the served default when the above is None
    max_guidance_scale: float = 20.0
    default_fps: int | None = None               # video only; 16
    accepts_negative_prompt: bool = False        # Wan
    accepts_guidance_scale_2: bool = False       # Wan
    accepts_cfg_parallel: bool = False           # Wan (CLI behaviour); serving always refuses it
    cfg_parallel_refusal: str                    # the per-model serving message
    refused_video_fields: Mapping[str, str] = {} # Wan's three, with reasons
    refused_cli_flags: tuple[tuple[str, str], ...]          # (flag, reason)
    refused_application_kwargs: tuple[tuple[str, str], ...] # (kwarg, reason); "teacache_*" allowed
    forwarded_cli_flags: tuple[str, ...] = ("torch_compile",)   # Qwen adds "teacache_cadence"
    prompt_budget: PromptBudget
    smoke_steps: int = 4
    smoke_prompt: str = "a small red square"
    driver: str                                  # "difflet.backends.cuda.flux.driver:FluxDriver"
```

Shape is not a field: the generic code reads `resolve_model(model_type).default_shape` (G9). `cli_model_ids` and `served_checkpoint_ids` are two fields because they differ today on purpose: the CLI routes FLUX.1-schnell and Wan2.1, serving accepts FLUX.1-dev and Wan2.2 only, and `tests/unit/serving/test_model_registry_cuda.py::test_wan21_is_not_served_on_cuda_because_it_never_ran_on_a_gpu` guards that. `accepts_cfg_parallel` has one behavioural reader (the CLI); serving refuses `--cfg-parallel` for every model in `difflet/serving/options.py` before an adapter runs, and the preparer's refusal is a backstop whose only per-model content is the message, hence the separate `cfg_parallel_refusal` text.

Three instances live in `difflet/common/registry/{qwen_image,flux,wan}.py`, each a screen of data, and `cuda_serving_metadata()` in each file becomes one call to a helper in `difflet/common/registry/base.py` that builds `ServingModelMetadata` from the spec (`checkpoint_ids=served_checkpoint_ids`, `default_guidance_scale=default_guidance_scale if not None else serving_noop_guidance_scale`, the one-stage `PipelineDefinition`, the factory strings). The `fuse_proj_out`, `timestep_dtype` and `enable_transformer_2` options stay programmatic kwargs read by the drivers: no CLI flag or serving profile field exposes them today, and this design adds none.

### The driver

What the brief calls "pipeline class" is not one class: Flux is driven by diffusers' `FluxPipeline`, Qwen-Image and Wan by Difflet's own orchestrators with different companion wiring (Qwen: text pipeline + VAE; Wan: three callables and host fp32 latents). The generic application therefore owns the `DiffletPipeline` contract and delegates the model-shaped steps to a driver, imported only in the worker:

```python
# difflet/common/cuda_spec.py (typing only; no torch)
class CudaModelDriver(Protocol):
    input_bundle_type: type | None       # QwenImageDiTInputBundle | FluxDiTInputBundle | None (Wan)
    def experts(self, app) -> list[tuple[str, CudaApplicationBase]]:
        """Named transformer applications, built once from CudaApplication.__init__ and cached:
        [("transformer", t)] or two for Wan. Each owns an artifact subdirectory of that name.
        Returns [] when transformer/config.json is missing, so the generic
        _require_transformer raises NotImplementedError naming "transformer/config.json"
        (six tests match that substring). Construction-time refusals (Wan's partial-download
        FileNotFoundError on boundary_ratio) run here."""
    def load_companions(self, app) -> None:
        """Text encoders, VAE, scheduler, the denoise driver; after every expert is loaded."""
    def generate(self, app, prompt, **kwargs):
        """Text → image/video; the model's own generate signature, unchanged from today
        (Wan's defaults to output_type="pt" and refuses a batch or a foreign num_frames)."""
    def dit_input_contract(self, app) -> dict[str, dict[str, Any]]: ...
```

The three drivers are today's `cuda_application.py` files minus the lifecycle trio, the kwargs refusal, `_require_transformer` and the `__call__` dispatcher, and they move under `difflet/backends/cuda/<model>/driver.py`. Each driver reads its own programmatic kwargs (`load_text_encoder`/`load_text_encoders`, `load_vae`, `local_files_only`, `text_seq_len`, `batch_size`, injected components); Flux's plural `load_text_encoders` is not renamed. Flux's driver keeps `FluxDiTAdapter` (rope from ids, cached per ids) and the `FluxPipeline` rebuild; Qwen's keeps `encode_prompt`, `qwen_image_timesteps` and the TeaCache cadence; Wan's keeps `_CudaTextEncoder`, `_CudaDiT` (fp32 timestep), `_CudaVaeDecoder`, the two-expert residency rule and the `boundary_ratio` partial-download check.

### The generic application

`difflet/backends/cuda/core/application.py`:

```python
class CudaApplication(torch.nn.Module):
    def __init__(self, spec: CudaModelSpec, *, model_path, parallel, dtype, shape, **kwargs): ...
    # DiffletPipeline's reflective contract, parameter names unchanged:
    def compile(self, compiled_model_path, debug: bool = False) -> None
    def load(self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False) -> None
    def has_compiled_artifacts(self, compiled_model_path) -> bool
    def generate(self, prompt, **kwargs)        # -> self.driver.generate(self, prompt, **kwargs)
    def dit_input_contract(self)                # -> self.driver.dit_input_contract(self)
    def forward_dit(self, bundle)               # only when driver.input_bundle_type is set
    def __call__(self, *args, **kwargs)         # bundle → forward_dit (if a bundle type); prompt= → generate;
                                                # else the first expert's forward
```

`compile`/`load`/`has_compiled_artifacts` iterate `driver.experts(self)` and join each name onto the artifact path, which is what Wan does today and what Flux and Qwen do with a single `("transformer", …)` entry. The kwargs refusal is one function reading `spec.refused_application_kwargs` with the existing "only when the value is not None/False" rule (G10). The `entry.py` factories keep their parallel-config refusals (they run before anything is built and the messages are model-specific) and construct `CudaApplication(SPEC, ...)`; `difflet/models/<model>/cuda_application.py` becomes a two-line re-export during migration and is deleted at the end.

### The serving adapter

`difflet/serving/orchestrators/cuda.py` absorbs `cuda_resident.py` and the four classes, importing nothing from `difflet.backends.cuda` at module level:

- `CudaServingArtifactPreparer`: `prepare_runtime` as today, with `refuse_unsupported_profile(profile, spec)` reading the shared refusals plus `spec.cfg_parallel_refusal` and the TeaCache/host-VAE refusals.
- `CudaServingRequestValidator`: shape equals profile (including `num_frames` and `fps` for video), guidance within `[0, max_guidance_scale]` or pinned to `serving_noop_guidance_scale`, `guidance_scale_2` and `negative_prompt` only when the spec accepts them, the prompt-token budget through `PromptBudget`, `refused_video_fields`.
- `CudaPipelineRunner`: one class; `execute` calls `adapter.generate_output(request)`.
- `CudaServingStageAdapter`: `create_loaded_runners` builds the pipeline through the one `cuda_pipeline_kwargs` (moved out of the CLI module, imported inside the method as today), adds `num_frames` for video; `generate_output` builds the kwargs from the request and the `accepts_*` flags and encodes the result as PNG for images or through `encode_video_tensor` for video; `smoke_request` and `validate_smoke_output` branch once on `spec.output`.

The registry factory strings point at three two-line subclasses per model that bind the spec (`class CudaFluxServingStageAdapter(CudaServingStageAdapter): spec = FLUX_SPEC`), because the registry constructs factories by import path with no arguments. A subprocess test imports the three serving modules and `model_registry` with `torch` blocked in `sys.meta_path`, pinning the parent-process rule that today is only a comment.

### The CLI orchestrator

`difflet/cli/orchestrators/cuda.py` is `CudaOneProcessOrchestrator` moved out of `qwen_image_cuda.py` and made spec-driven: `HF_MODEL_ID`, `MODEL_IDS`, `DEFAULT_STEPS`, `ACCEPTS_*`, `OUTPUT_TYPE`, `TAG` read off `self.spec`; `refuse_model_flags` reads `spec.refused_cli_flags`, with the `default_guidance_scale is None` branch producing Qwen's "has no effect" refusal of `--guidance-scale`; `application_kwargs` forwards `spec.forwarded_cli_flags` that are set on `args`; `save_result` branches on `spec.output` (Wan's MP4 writer and `.pt` fallback become the video branch). `cuda_pipeline_kwargs` moves to `difflet/backends/cuda/core/pipeline_kwargs.py`, taking the backend name as a parameter (the backend the spec was selected for, `"cuda"` today), not from `current_backend()`, so the five CPU-suite tests asserting `kw["backend"] == "cuda"` stay valid. The torchrun re-exec, the data-parallel worker loop and the rank-0 broadcast of claimed requests do not change.

Through Phase 5, `main._get_orchestrator` keeps returning the three spec-bound subclasses (`CudaQwenImageOrchestrator`, `FluxCudaOrchestrator`, `WanCudaOrchestrator`), because `tests/unit/cli/test_cli_main_cuda.py` asserts those class names; Phase 6 collapses the selection to `CudaOneProcessOrchestrator(args, spec=spec_for(args.model_id))` and updates that test in the same commit.

### What each model still supplies at Level 5

| Model | Spec (torch-free data) | Transformer subclass | Driver | Lines (estimate) |
|---|---|---|---|---|
| Qwen-Image | prompt template and 34-token drop; `teacache_cadence` forwarded; no guidance; the four probe-mode kwargs refused | `_QwenImageTransformerTraceModule(apply_text_mask=True, trim_text_padding=True)`, `load_checkpoint_into(prefix=…)`, `compile_dynamic=True`, two warm-ups | text pipeline, VAE, `QwenImageOrchestrator`, `qwen_image_timesteps`, the bundle type | ~300 |
| Flux | two CLI ids, one served; T5 budget on `tokenizer_2`; the distilled-guidance CFG message | `NeuronFluxTransformer2DModel`, `build_weight_map` + `convert_flux_hf_state_dict`, two block lists, the `proj_out` swap | `FluxDiTAdapter` (rope from ids), `FluxPipeline` rebuild with the adapter, `fuse_proj_out`, the bundle type | ~280 |
| Wan 2.2 | two CLI ids, one served; fps; `guidance_scale_2`; `negative_prompt`; CFG-parallel; three refused video fields | `WanTransformer3DModel`, `load_checkpoint_into(rename=…)`, the cfg=2 mesh and the world-size check | two experts and the boundary check, UMT5, VAE decoder, `WanOrchestrator`, `timestep_dtype`, `enable_transformer_2`, no bundle type | ~420 |

Generic code, estimated: `common/cuda_spec.py` ~110, `core/compile.py` ~40, `core/application.py` ~220, `core/pipeline_kwargs.py` ~40, `serving/orchestrators/cuda.py` ~380, `cli/orchestrators/cuda.py` ~320, plus the existing `core/application_base.py` ~240 after it absorbs the lifecycle. About 1,350 generic lines and about 1,000 per-model lines replace today's 3,223 (3,116 + `cuda_resident.py`), and the per-model share falls from ~1,000 lines to 280–420.

---

## The shared tests

**Census at `c45d2059`** (`grep -c "def test_"` over the files below): 211 tests in the eleven per-model files (serving 19/17/19 for Qwen/Flux/Wan, CLI 22/14/41, application 17/18/25, transformer 9/10 for Wan/Flux) plus 41 in six files the migration also touches: `tests/unit/backends/test_cuda_application.py` (10, the base lifecycle), `tests/unit/serving/test_cuda_resident.py` (4), `tests/unit/serving/test_model_registry_cuda.py` (14), `tests/unit/cli/test_cli_main_cuda.py` (9), `tests/unit/backends/test_cuda_flux_proj_out.py` (4), and the Qwen recompile test in `tests/unit/models/qwen_image/test_qwen_text_mask.py`, which is not a `*cuda*` file and is exempt from Phase 6's deletion. 252 in all; Phase 6 records the count before and after.

After the migration:

- `tests/unit/ops/test_backend_tolerance.py` (Level 1): the table, parametrised over backends present; `exact` rows also assert dtype, shape and strides; rows with more than one implementation on a backend assert which one ran (G11).
- `tests/unit/ops/test_ops_index.py` (Level 1): every name `difflet.ops` exports appears in `difflet/ops/README.md`.
- `tests/unit/backends/test_cuda_compile.py` (Level 4): `compile_blocks` replaces every block in every list in place and returns the count; the three subclasses declare the lists and flags the table above says; swaps run before compile (Level 3).
- `tests/unit/backends/test_cuda_flux_proj_out.py` (Level 3): the four obligations of "Acceptance for a swap" that run on the CPU, including that `apply_load_time_swaps` reports the swap and the swapped forward ran. The bf16 block-parity check against diffusers runs on the GPU and is recorded in the ops index.
- `tests/unit/backends/test_cuda_application.py` (Level 5): the existing base tests, plus tests parametrised over `ALL_SPECS = (QWEN_IMAGE_SPEC, FLUX_SPEC, WAN_SPEC)`: the contract signatures match `DiffletPipeline`'s reflective expectations (the existing `test_cuda_application_signatures_match_the_pipeline_contract` generalised); `compile`/`load`/`has_compiled_artifacts` visit every expert's subdirectory; off-valued refused kwargs are accepted and on-valued ones raise naming the key; `__call__` dispatches a prompt and raw tensors for every spec and a bundle for the two specs with a bundle type; `generate` before `load` is an explicit error.
- `tests/unit/serving/test_cuda_serving.py`: parametrised over `ALL_SPECS`: `prepare_runtime` yields an eager bundle with no compile specs; unsupported profiles are refused before the download; the runtime plan is accepted by the resident worker environment; the smoke request is at the profile shape with `smoke_steps`; the validator enforces shape, guidance range or pinned value, prompt budget, and the `accepts_*` flags; `generate_output` passes exactly the CLI's kwargs and returns the spec's MIME type; cancellation never reaches `generate`; shutdown releases the pipeline; the parent-process import of the module never loads torch (subprocess).
- `tests/unit/cli/test_cuda_orchestrator.py`: parametrised over `ALL_SPECS`: `run` is download → compile → generate; refusals name the flag and happen before any download; defaults and overrides reach `from_pretrained` and `generate`; `world_size` is `cfg × cp × tp`; the torchrun re-exec and the data-parallel loop (today's Wan CLI tests, which are the only ones exercising them) run for whichever spec accepts CFG-parallel.
- `tests/unit/backends/cuda/test_<model>_driver.py`: the model-specific behaviours from the "genuinely different" table, moved from today's application tests: Flux's rope cache per ids and `guidance=None`; Qwen's prompt trim shape and the no-op guidance refusal; Wan's expert routing by boundary, the fp32 timestep default, the batch-of-two under CFG-parallel, the partial-download refusal, the three callables.
- `tests/unit/serving/test_model_registry_cuda.py`: the generated metadata equals the spec, field by field, including the served-checkpoint subset.
- `tests/unit/backends/test_cuda_import_hygiene.py`: `SCANNED` already recurses `difflet/backends/cuda/` and `_torch_megatron/`; it gains `difflet/serving/orchestrators/cuda.py`, `difflet/cli/orchestrators/cuda.py` and `difflet/common/cuda_spec.py` (and loses the two `cuda_application.py` entries in the commit that deletes those files, since the test fails on a missing entry). The forbidden-prefix list is unchanged.

The GPU-only gate stays what it is: the three parity harnesses against diffusers plus, for this migration, the byte-identity check of G5 (one `run` and one `serve` request per model at seed 0 before and after each Level 5 phase, md5 of the PNG or of the MP4's frame tensor, since the MP4 container is not byte-deterministic).

---

## What a fourth CUDA model writes

Assume a diffusers image model with backend-neutral modeling already on `difflet.ops` (the precondition the 2026-10-16 plan's D3 set for CUDA at all). The new model adds:

1. `CudaModelSpec` instance in `difflet/common/registry/<model>.py` (~25 lines) and `cuda_serving_metadata()` as one helper call (~3 lines).
2. `difflet/backends/cuda/<model>/transformer.py` (~50 lines): `build_module`, `load_weights`, `get_example_inputs`, `repeated_blocks`, `forward`.
3. `difflet/backends/cuda/<model>/driver.py` (~100–200 lines): `experts`, `load_companions`, `generate`, `dit_input_contract`, `input_bundle_type`.
4. The `backend == "cuda"` branch in `difflet/models/<model>/entry.py` (~15 lines, mostly its parallel refusals), the three two-line serving subclasses, the entry in `model_registry`'s CUDA table and in `main._get_orchestrator`'s spec lookup. If it uses a shared fusion or adds a swap: its consumer line in `difflet/ops/README.md`, its bit-exactness test against its own chain, and the swap's acceptance record.
5. Tests: one line appending the spec to `ALL_SPECS`, and a `test_<model>_driver.py` for its own differences.
6. GPU evidence: a parity run against diffusers and the smoke on `run` and `serve`, recorded in a status log.

About 250–400 lines of code where today's precedent is 850–1,150, and no new serving adapter, CLI orchestrator or compile loop. A model with no real difference beyond its pipeline writes only items 1, 3 and 4.

## What a ROCm backend writes

`difflet/backends/rocm/runtime.py` already exists with `requires_aot=False`, and `difflet/backends/registry.py` already registers and auto-detects `"rocm"`. Torch on ROCm exposes the CUDA API under the `cuda` device type, so the op implementations and the generic layer run unchanged once two things are done:

1. `difflet/backends/rocm/` gets `ops_impl/` re-exporting `difflet.backends.cuda.ops_impl.*` except `platform.py`, which it provides (device name, `distributed_backend_name()` still `"nccl"` through RCCL, an `sdpa_kernel_report` of its own, and `device_type()`), and `runtime.py` stops raising `NotImplementedError`.
2. The fourteen code `"cuda"` literals on the generic path go through one accessor. Today they are the `torch.device("cuda")` defaults in `core/application_base.py` and the three applications, `backend="cuda"` in `cuda_pipeline_kwargs`, `placement="cuda"` and `"backend": "cuda"` in `cuda_resident.resident_runtime_plan`, the `current_backend() == "cuda"` branches in `cli/main.py` (`_on_cuda`, `_get_orchestrator`) and `serving/model_registry.py`, and the manifest's `"cuda": torch.version.cuda`. `platform.device_type()` is new (today's `platform.py` exports `hardware`, `get_platform_target`, `distributed_backend_name`, `sdpa_kernel_report`); the backend *name* is a parameter of `cuda_pipeline_kwargs` and the resident plan, set by whoever selected the spec, never read from `current_backend()` inside the generic layer (the CPU suite runs these paths with `DIFFLET_BACKEND=cpu`).

Then ROCm registers the same three specs, the same drivers and the same orchestrators under its own backend name, and the tolerance table runs against it on the first ROCm host with its own measured bf16 column. None of this is scheduled; item 2 is listed so that no new `"cuda"` literal is added to the generic layer during this migration (a unit test greps for it, the way the import-hygiene test scans for vendor imports).

---

## Migration order

Every phase: one branch, one PR, one-task-one-commit, `DIFFLET_BACKEND=cpu ./scripts/test_unit.sh` green at every commit (baseline at `c45d2059`: 2819 passed, 176 skipped, torch 2.14.1, diffusers 0.38.0, on a host without a GPU), `scripts/test_imports.sh` green, Black at line length 100, and a status-log entry here. Phases 2–5 additionally end with the G5 byte-identity check per model on a GPU.

The order is chosen so that each phase's diff is one subject and so that the safety nets (Level 1 table, Level 4 helper tests, shims) exist before the file moves they protect.

### Phase 1 — Level 1 tolerance table and the scan tests (no production change)

- [ ] Add `tests/unit/ops/test_backend_tolerance.py` with the fp32 rows at today's tolerances, each marked `exact` or `tolerance` with its reference, and `xfail(strict=True)` rows for bf16; CPU-only run green.
- [ ] Add `difflet/ops/README.md` (the index of every op, plus the five-part contract for each op with more than one implementation on a backend; no fusion exists yet) and `tests/unit/ops/test_ops_index.py`.
- [ ] On a GPU: run the table with the CUDA column, record max abs/rel error per row in the status log, set the bf16 tolerances, remove the `xfail` markers; the attention rows assert the SDPA kernel for the shapes Qwen-Image, Flux and Wan run (G11), and the selected kernels go into the index as the first qualification records.
- [ ] Add the `"cuda"`-literal scan test for the generic path (ROCm item 2) with today's fourteen sites as the allowed list, so the list can only shrink.
- [ ] Add the subprocess test that imports the three CUDA serving modules and `model_registry` with torch blocked, pinning the parent-process rule before Phase 4 moves that code.

### Phase 2 — Levels 3 and 4: lifecycle into the base class

- [ ] GPU, before any change: run `--torch-compile` twice per model on this build (cold and warm inductor cache), record whether the md5 is stable; set G5's compiled-mode rule accordingly in the status log.
- [ ] Add `difflet/backends/cuda/core/compile.py::compile_blocks` with its test.
- [ ] Move the shared `_prepare_module` body into `CudaApplicationBase` with the hooks `load_weights(module)` (default: the `get_state_dict` path), `repeated_blocks(module)`, `apply_load_time_swaps(module)`, `compile_dynamic`, `warmup_passes_when_compiled`, and `requires_grad_(False)` in the base; the three transformer subclasses shrink to their hooks. Flux's swap call moves into `apply_load_time_swaps`.
- [ ] Existing transformer, base and application tests pass unchanged; add the parametrised compile-declaration test, and the CPU part of the `proj_out` swap's acceptance (it reports the swap; the swapped forward ran).
- [ ] GPU: byte-identity per model (`run`, seed 0) against the Phase 1 build; eager, and compiled under the rule set above.
- [ ] GPU: no slowdown. The per-model transformer benches (`scripts/cuda_bench_dit.py`, `scripts/cuda_bench_flux_dit.py`, `scripts/cuda_bench_wan_dit.py`), eager and compiled, old and new builds alternated A-B-B-A after warm-up, at least five samples each; the acceptable band is written in the status log before the run, and the result with its spread after it. Compile time is recorded apart from steady-state time.
- [ ] GPU: the `proj_out` swap's block-parity check against diffusers' block in bf16 at Flux's shapes, recorded in the ops index whatever it shows. It changes nothing by default (the swap stays off) and answers the reviewer's question about the 10 dB shift.

### Phase 3 — The spec and the CLI move

- [ ] Add `difflet/common/cuda_spec.py` (`CudaModelSpec`, `PromptBudget`, `CudaModelDriver`) and the three spec instances in `difflet/common/registry/<model>.py`; `difflet/backends/cuda/core/pipeline_kwargs.py` takes `cuda_pipeline_kwargs` from the CLI module with the backend name as a parameter; `qwen_image_cuda.py` re-exports it.
- [ ] Move `CudaOneProcessOrchestrator` to `difflet/cli/orchestrators/cuda.py`, spec-driven; the three `<model>_cuda.py` CLI files become the spec-bound subclasses plus re-exports of their constants; `main._get_orchestrator` still returns those three classes. Each spec's refusal lists reproduce today's per-model refusals exactly, including that Qwen-Image's application does not refuse `shapes=` (G10).
- [ ] Existing CLI tests pass through the re-exports; add `tests/unit/cli/test_cuda_orchestrator.py` parametrised over `ALL_SPECS`; add `difflet/cli/orchestrators/cuda.py` and `difflet/common/cuda_spec.py` to `SCANNED`.
- [ ] GPU: byte-identity per model on `run`, including Wan `--cfg-parallel` and the data-parallel router path.

### Phase 4 — The generic serving adapter

- [ ] Add `difflet/serving/orchestrators/cuda.py` (absorbing `cuda_resident.py`, which re-exports); the three `<model>_cuda.py` serving files become the spec-bound subclasses the registry needs; `cuda_serving_metadata()` becomes the helper call in `difflet/common/registry/base.py`.
- [ ] Existing serving tests pass; add `tests/unit/serving/test_cuda_serving.py` over `ALL_SPECS`; `test_model_registry_cuda.py` asserts the metadata equals the spec; add the module to `SCANNED`; the torch-blocked subprocess test still passes.
- [ ] GPU: byte-identity per model on `serve` (one request each, and `--dp-degree 2` for one model) against Phase 3.

### Phase 5 — The generic application and the drivers

- [ ] Add `core/application.py::CudaApplication`; move the three drivers to `difflet/backends/cuda/<model>/driver.py`; `difflet/models/<model>/cuda_application.py` becomes a re-export; `entry.py` constructs `CudaApplication(SPEC, …)`.
- [ ] Existing application tests pass through the re-exports; add the parametrised application test and the three driver test files (moved tests, not new behaviour).
- [ ] GPU: byte-identity per model on `run` and `serve`; the three parity harnesses against diffusers, unchanged numbers in the status log; the A-B-B-A timing check of Phase 2 repeated on whole `run` requests.

### Phase 6 — Delete the shims

- [ ] One task per deleted file, each with its `SCANNED` edit and the per-model tests it retires in the same commit: `difflet/models/<model>/cuda_application.py`, `difflet/cli/orchestrators/<model>_cuda.py`, `difflet/serving/orchestrators/<model>_cuda.py` (the spec-bound serving subclasses move next to the generic adapter), `cuda_resident.py`.
- [ ] `main._get_orchestrator` collapses to the generic class with `test_cli_main_cuda.py` updated in the same commit; DEVELOPER.md's backend section updated.
- [ ] CPU suite green; the CUDA test count is recorded before and after against the 252 baseline so nothing was dropped silently.

### Phase 7 — Level 2, conditional

- [ ] Precondition check, recorded in the status log: a Trainium verification host exists, and a transformer bench on Flux or Qwen-Image with a prototype fused q/k-norm+rope kernel shows a win at those models' shapes, including the smallest request shape.
- [ ] Add `difflet.ops.qk_norm_rope` with explicit `interleaved` and `rotary_dim`, its own reference in every backend, its contract in the ops index and its tolerance rows.
- [ ] Per consumer, Flux then Qwen-Image: a bit-exactness test of the op against the model's own chain on each backend at the model's shapes and dtype; switch the model to the op only where it passes, keeping its own chain otherwise; the model's enable decision (G12); path assertions (G11); Trainium graph proven unchanged.
- [ ] The CUDA fused kernel behind the same name, with its tolerance row, its path assertions and its qualification records.

### Afterwards, separate and small

- Qwen-Image's application refuses `shapes=` like Flux and Wan (G10): a behaviour change in its own PR, with its test, after Phase 5.
- The six GPU scripts become `scripts/cuda_e2e_vs_diffusers.py --model` and `scripts/cuda_bench_dit.py --model`, reading the spec (G8).
- The ROCm package (item 1 above) when a ROCm host is scheduled.

---

## Risks

| Risk | Where it bites | Mitigation |
|---|---|---|
| **A refactor changes numerics without failing a test.** The parity gate against diffusers allows bf16 noise, so a one-ulp regression hides inside it. | Phases 2–5 | G5: byte-identical output per model at seed 0 before and after each phase, on `run` and `serve`. The recorded md5s from the current stack are the first reference. |
| **Compiled mode may not be byte-stable at all**, so a compiled gate would fail or flake with zero regression. | Phase 2 | Measured first (Phase 2's first task); the gate rule is written down before the code moves. |
| **The serving parent starts importing torch** through a spec placed under the backend package. | Phase 3 | G3 puts the spec in the torch-free registry layer; the subprocess test from Phase 1 fails the build if any serving-side import pulls torch. |
| **The spec grows into a hook bag.** Each new model adds "just one more optional callable". | Phase 3 onward | G3's rule (two generic readers or it is the driver's), enforced in review; the spec stays a frozen dataclass of data plus one driver import string. |
| **Wan is the only video, CFG-parallel, two-expert model**, so "generic" is shaped by one example. | Phases 3–5 | The generic code branches on `spec.output` and `accepts_cfg_parallel` only; the expert loop is the *same* loop for one expert. If a second video model arrives and the branch is wrong, the branch is the whole surface to change. |
| **Level 2 touches Trainium-traced modeling, and a "same" fusion is not the same arithmetic.** Flux and Qwen-Image rotate in fp32, Wan in the tables' dtype, MiniMax-H3 half-split in the input dtype. | Phase 7 | Scheduled last, conditional on a Trainium verification host and a measured win; each consumer keeps its own chain unless a per-backend bit-exactness test shows the op reproduces it; the variants are explicit in the op's signature. |
| **A fast path silently never runs.** A caller that always falls back passes every output test. | every level | G11: tests assert the selected path, not only the output. |
| **A behaviour change rides inside a move**, where it cannot be reviewed as one and breaks the byte-identity argument. | Phases 2–6 | G10: the series reproduces today's behaviour; the one known change (Qwen-Image refusing `shapes=`) is its own PR afterwards. |
| **A move slows the model without changing a byte**, for example a compile graph break introduced by moving the compile loop. | Phases 2 and 5 | The A-B-B-A timing check with a band set before the run; compile time recorded apart from steady state. |
| **Test churn hides regressions.** 252 tests name today's paths; a rewrite in the same commit as a move is unreviewable. | Phases 3–6 | G6: shims first, parametrised tests added beside the old ones, deletion in its own phase with a before/after count. |
| **The open 20-PR stack moves under this series.** | Phase 1 start | This series bases on the stack's top (or on `main` once it merges) and touches no file the stack's remaining review changes; Phase 1 has no production diff and can land first either way. |
| **Compile quirks are per model and easy to flatten by accident** (Qwen's `dynamic=True` and double warm-up; Flux's two lists). | Phase 2 | The three flags are declared per subclass and asserted by the parametrised compile test; the Qwen recompile test stays. |
| **`"cuda"` literals creep back into the generic path**, making ROCm a hunt again. | every phase | The literal-scan test from Phase 1, with an allow-list that may only shrink. |
| **The hygiene scan's `SCANNED` list names files**, so a deleted file breaks it and a new orchestrator module outside `difflet/backends/cuda/` is unscanned. | Phases 3–6 | Each move adds its module in the same task; each deletion removes its entry in the same task. |

---

## Open questions for the owner and the reviewer

Each is worded so that "yes" takes the recommended answer.

1. **Spec placement in the torch-free registry layer** (`difflet/common/cuda_spec.py` + instances in `difflet/common/registry/<model>.py`), as G3 says, rather than under `difflet/backends/cuda/<model>/`?
2. **Base on `main` after the stack merges**, with Phase 1 (no production diff) openable against the stack's top in the meantime?
3. **Level 2 stays out of this series** (Phase 7 is written down with its preconditions and is cut if the measured win is absent)?
4. **One PR per phase** (six likely, seven at most), each landable alone, because Phases 2–5 each need their own GPU byte-identity run?
5. **Make Qwen-Image's application refuse `shapes=` like Flux and Wan in its own PR after Phase 5**, not inside the migration (G10)?
6. **Keep `fuse_proj_out`, `timestep_dtype` and `enable_transformer_2` programmatic-only** (no new CLI flags or serving profile fields in this series)?
7. **Keep Difflet's layout of model under backend** (`difflet/backends/cuda/<model>/`, as `trainium/` and `tpu/` already do) rather than vLLM's newer vendor-under-model layout (`models/<model>/{nvidia,amd,common}/`, cited from vllm-project/vllm#42770 through the RFC)? Switching would move the Trainium and TPU per-model code too, which is outside this series.

---

## Status log

| Date | Phase | Verdict | Evidence |
|---|---|---|---|
| 2026-10-21 | Design | Draft on a branch cut from the top of the CUDA stack; duplication measured at `c45d2059`; CPU baseline 2819 passed / 176 skipped; CUDA test census 252. Stress-tested against the code before commit: sixteen findings folded in (two Hub-id sets, the torch-free serving parent, the test census, Phase 3 vs the CLI routing test, the base lifecycle's `get_state_dict` path, compiled-mode stability unmeasured). | This document. |
| 2026-10-21 | Design | Revised against vLLM-Omni RFC #7382, which the reviewer cited: rope variants explicit and per-consumer exactness at Level 2 (the earlier "bit-identical by construction" claim did not hold); the model decides whether to use a fusion (G12); fast-path assertions (G11); acceptance for a swap at Level 3; per-op contracts and the ops index (G7); behaviour changes and timing checks kept apart from moves (G10, Phases 2 and 5); open question 7 on directory layout. | This document; vllm-project/vllm-omni#7382. |
