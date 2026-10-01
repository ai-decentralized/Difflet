# Difflet on native PyTorch for Trainium (TorchNeuron): research, impact, and migration plan

Status: research report and plan, 2026-10-01. No code changed. Written on a `trn2.3xlarge`
(one Trainium2 chip, 8 physical cores, LNC=2 -> 4 logical cores, Ubuntu 24.04, Python 3.12).

Sources examined (all read-only):

- AWS Neuron docs 2.32.0 (released 2026-08-17): the PyTorch-native overview, the PyTorch
  about-page, the PyTorch 2.10 transition announcement, What's New 2.27-2.32, compiler/NKI/runtime
  component release notes, the compiler CLI reference, the OSS repositories page.
- `/home/ubuntu/torch-neuron/torch-neuronx-develop`: AWS's TorchNeuron ("TorchNeuronEager")
  source drop. It contains **only the test suite and build glue** (369 Python test files, 86 K LOC;
  82 C++ test files, 30 K LOC; `pyproject.toml`, `CMakeLists.txt`, `WORKSPACE`), not the
  `torch_neuronx/` package. Everything stated below about the native API is reconstructed from
  what those tests import, call, and assert.
- The public AWS pip index (`pip.repos.neuron.amazonaws.com`): wheel inventories for
  `torch-neuronx`, `neuronx-cc`, `nki`, `torch-xla`, `libtorch-neuronx-lite`, `vllm-neuron`,
  `vllm-omni-neuron`; the `libtorch-neuronx-lite` 2.13 wheel was downloaded and unpacked, and the
  older 2.11 build installed in this host's AMI vLLM venv was read for comparison.
- AWS's public reference implementation of a video DiT on the new stack:
  `github.com/aws-neuron/vllm-omni-neuron` (Wan 2.2 T2V/I2V, release 0.24.0.0.1.0), plus
  `nki-library` and `nki-samples`.
- The installed environments on this host and the Difflet tree (branch `quantization`, commit
  `5190dd6`).

---

## 0. Answers in one page

**What "PyTorch native" is.** TorchNeuron is a PrivateUse1 backend that registers Trainium as the
`neuron` device. Models run in eager mode (`.to("neuron")`), under `torch.compile(backend="neuron")`
(TorchDynamo FX graph -> torch-mlir -> StableHLO -> Neuron compiler -> NEFF), and with standard
`torch.distributed` (`init_process_group(backend="neuron")`, DTensor, FSDP, DDP). NKI kernels are
plain Python callables on `neuron` tensors or `torch.library` custom ops via
`@torch_neuronx.nki_op`. There is no lazy tensor, no `mark_step`, no `torch_neuronx.trace`, no
`neuronx_distributed`, no `libneuronxla`, no PJRT.

**Status and timeline.** Closed beta ("private beta, requires account representative approval").
PyTorch 2.9 is the last torch-xla-based `torch-neuronx`; the native implementation ships under the
**same distribution and import name** (`torch-neuronx` / `torch_neuronx`) starting with PyTorch 2.10
support "in a future Neuron release". Neuron 2.32.0 (Aug 2026) still ships the XLA-based
`torch-neuronx 2.9.0.2.15` as the public PyTorch package. The only *public* artifact of the native
stack is `libtorch-neuronx-lite` ("Lite", torch 2.10 through 2.13 builds, Alpha, proprietary
licence), a vLLM-oriented runtime that vendors TorchNeuron's compiler and reserves the `neuron`
backend name for the real package; AWS's own Wan 2.2 serving plugin runs on Lite's native lane, not
on `backend="neuron"`.

**What Difflet has to replace.** The whole Trainium execution model, not just API spellings:
the NxD `ModelBuilder.trace()` AOT lifecycle (`model.pt` + presharded weights + `nxd_model.initialize`),
the single-process/N-core SPMD runtime, NxD `parallel_layers` and `parallel_state`, the XLA
collectives emitted inside traced graphs, the HLO custom calls (`AwsNeuronRmsNorm`, module markers),
NKI kernels invoked through `torch_xla`, the NxDI text-encoder models, the FP8 path built on NxD
quantized layers, the TeaCache probe NEFF, and the serving worker's artifact model. Section 3 has
the item-by-item table.

**What changes.** Add a new Difflet backend (`DIFFLET_BACKEND=neuron`) beside `trainium`: eager
model construction on `neuron`, `torch.compile(backend=<neuron backend>, fullgraph=True)` per component
per shape with everything that varies per step passed as a tensor, `torch.distributed` process groups
(one process per logical core), `difflet.ops` reimplemented on `torch.distributed` + built-in/NKI
kernels, NKI kernels re-registered as custom ops, a persisted NEFF cache + manifest instead of
`model.pt`, and a multi-process resident serving worker. A thin runtime-adapter module (the shape of
vllm-omni-neuron's `lite_compat.py`) isolates Difflet from whether the runtime underneath is Lite or
the TorchNeuron beta. Model code (`difflet/models/*`, `difflet/layers/*`) mostly stays as is because
it already imports only `difflet.ops`. Section 4.

**Verification.** Reuse the house conventions (`difflet-device-verify`, `scripts/verify_cli.py`
matrix, DP bit-identity, serving smoke) and add the three-way accuracy methodology AWS uses for Wan
2.2 (FP32 CPU -> BF16 CPU -> BF16 Neuron, per component, per step, end to end). Freeze today's XLA
outputs as goldens before touching anything. Section 5.

**Impact.** Big structural change with a large upside (eager debugging, no AOT trace, weights resident
across shapes, trivial TeaCache/segmented execution, one toolchain shared with vLLM) and real risks
(closed-beta dependency or an Alpha public runtime with private APIs, static-shape-only compile with
Dynamo recompiles, collective topology limits, no point-to-point send/recv for ring CP, no
`_scaled_mm` for FP8 and OCP FP8/MX kernels being Trainium3-only, untested VAE ops such as
conv3d/GroupNorm/upsample, Python >= 3.11 and torch >= 2.10, 40-50 minute cold compiles reported by
AWS for Wan 2.2). Section 6.

**Plan.** Six phases, gated by a two-to-three-week on-device spike: access + spike -> backend
scaffold -> ops + kernels -> Wan 2.1 end to end incl. serving -> remaining models -> parity/perf
campaign and cut-over, with the XLA backend kept until the campaign passes. The spike can start today
on Lite (public) while beta access to TorchNeuron proper is requested. Section 7.

**Compiler.** Difflet compiles today with **`neuronx-cc 2.26.6360.0+6f180f47`** (Neuron SDK 2.31 line),
deliberately pinned because 2.27 crashes on the FLUX CLIP encoder (commit `23674df`), invoked
through `libneuronxla` from NxD `ModelBuilder` with hardcoded flag strings. TorchNeuron **does not
introduce a new compiler**: its Dynamo backend and eager path shell out to the same `neuronx-cc`
binary on `PATH`. What changes is the *input*: the native lowering emits StableHLO MLIR (the XLA path
emits HLO protobuf), compiler flags are passed via `NEURON_CC_FLAGS` / `compiler_args` instead of
`ModelBuilder.add(compiler_args=...)`, and the required compiler version will be whatever the runtime
pins (AWS's Wan 2.2 plugin pins `neuronx-cc==2.27.5334.0` exactly; the `nki_library` package needs
`neuronx-cc>=2.26`). So: yes, expect to move off the 2.26 pin, and treat "which neuronx-cc does the
native runtime need, and does it still ICE on FLUX CLIP" as the first question of the spike. Section 8.

---

## 1. What TorchNeuron is (from the docs and the test suite)

### 1.1 Programming model

| Aspect | TorchNeuron (native) | torch-neuronx 2.9 (what Difflet uses) |
|---|---|---|
| Device | `torch.device("neuron")` (PrivateUse1 renamed), `torch.accelerator` aware, autoloaded via the `torch.backends` entry point on torch >= 2.9 | `xla` lazy device; Difflet never names it, NxD traces on CPU |
| Eager | Real eager dispatch of ATen ops to per-op NEFFs; "Adaptive Eager Execution" fuses consecutive ops ("op concatenation") and folds view/transpose prologues, async execution engine, caching allocator, CPU fallback for unregistered ops | Not applicable (lazy tensors) |
| Compile | `torch.compile(backend="neuron")`: Dynamo -> AOT autograd graphs -> FX passes (dynamic-shape check, collective legalisation, aliasing, functionalisation) -> torch-mlir -> StableHLO -> `neuronx-cc` -> one NEFF per graph segment; graph breaks supported | `neuronx_distributed.trace.ModelBuilder.trace()` -> HLO -> `neuronx-cc` -> bucketed NEFFs saved as `model.pt` |
| Shapes | **Static only.** `DynamicShapeAnalysis` raises on any symbolic dim; a new shape is a Dynamo guard miss and a new NEFF | Explicit bucket list, router dispatches by input shape signature |
| Distributed | `dist.init_process_group(backend="neuron")`; one device per process, device index = local rank; `new_group`, `init_device_mesh("neuron", ...)`, DTensor TP, DDP, FSDP; collectives compiled into graphs with SPMD replica groups from a mesh registry | Single process owns N cores; NxD `parallel_state`; XLA collectives inside the traced graph |
| Collectives | TorchNeuron proper: eager and compiled all_reduce, all_gather(_into_tensor), reduce_scatter, all_to_all(_single), broadcast, gather, scatter, reduce, barrier; **no send/recv**; replica groups must be 4/8/16/32-aligned; 2-rank all_to_all with LNC=2 unsupported. **Lite: a device collective is serviced only when traced into a compiled graph; eager `dist.*` on `neuron` tensors is silently dropped** | `xm.all_gather/reduce_scatter/all_to_all/collective_permute` plus NxD mappings |
| NKI | `@nki.jit` kernels callable directly on `neuron` tensors; `@torch_neuronx.nki_op("ns::op", mutates_args=...)` registers a `torch.library` custom op usable in eager and inside `torch.compile(fullgraph=True)`; autograd via `register_autograd()`. Lite: `wrap_nki(kernel)[lnc](...)` HigherOrderOperator with fake-tensor shape inference, `nki_op` only around a Python function that calls it | `nki.jit` kernels invoked inside the traced graph; `nki.framework.torch_xla.TorchXlaKernel` |
| Built-in kernels | SDPA -> `_scaled_dot_product_fused_attention_overrideable` (NKI flash, MHA/GQA/MQA, causal, mask, head dim 128/192); fused RMSNorm (`aten::_fused_rms_norm`, torch >= 2.9); `_grouped_mm`; flex attention lowered to generated NKI | nkilib `attention_cte`, `AwsNeuronRmsNorm` custom call |
| Dtypes | f32/f16/bf16/int8-64/bool; float8 e4m3/e5m2 casts and data-path ops under compile; **no `_scaled_mm`**; eager auto-casts f64->f32, i64->i32; RNG ops fall back to CPU; `torch.Generator(device="neuron")` not available | bf16/fp32; FP8 via NxD quantized layers + compiler flag |
| Caching | In-memory LRU + persistent NEFF cache `TORCH_NEURONX_NEFF_CACHE_DIR` (NFS-shareable, XXH3 key over IR bytes + compiler args + compiler version), HLO cache, `ModelHandleCache`; torch.compile artifacts plug into PyTorch MegaCache (`torch.compiler.save_cache_artifacts`); NKI has its own cache (`NKI_COMPILE_CACHE_URL`) | `/var/tmp/neuron-compile-cache` via libneuronxla; Difflet `~/.cache/difflet/<model>/<hash>/` with `model.pt` and presharded weights |
| Flags | `NEURON_CC_FLAGS`, `NEURON_COMPILER_OPT_LEVEL` (default `-O1`), `options={"model_name": ...}`; Lite also accepts `options={"compiler_args": [...], "compiler_workdir": ...}` | Hardcoded flag strings per application (`--model-type=transformer -O1 --auto-cast=none ...`) |
| Observability | `torch.neuron.get_dynamo_metrics()`, `torch_neuronx.memory_stats()`, `torch.profiler` with NRT inspect traces, op tracking (`get_executed_ops/get_fallback_ops`) | `neuron-profile`, NEFF-level timing |

Package facts from the develop drop's `pyproject.toml`: distribution `torch-neuronx`, import `torch_neuronx`,
`requires-python >=3.10,<3.14`, **`torch~=2.10.0`**, Bazel 8.4 + StableHLO 1.13.1 C++ extension linked
against `libnrt`, entry points `torch.backends: torch_neuronx = torch_neuronx:_autoload` and
`torch_dynamo_backends: neuron = torch_neuronx.neuron_dynamo_backend.backend:neuron_backend`.
Because the distribution name is unchanged, **the native package cannot coexist with the XLA-based
`torch-neuronx` in one virtualenv**; the migration needs a second venv until cut-over. AWS's setup
guide for the Wan plugin says the same from the other side: "Do not mix this stack with a standalone
`torch-neuronx` installation."

Maturity signals from the suite: 250 xfails, 158 skips, "[TODO FIX post Beta]" markers, device RNG
not implemented (falls back), `send/recv` unsupported, barrier only on the default group, >1 device
per process unsupported, host-side multi-stream collectives "under development", experimental
"tensorizer" backend gated off, 736 upstream ATen test methods recorded as failing. The C++ runtime
core (caches, async engine, allocator, concatenation) is deeply unit-tested; op coverage is the thinner
part. **Diffusion-relevant ops with zero tests in the suite: conv3d, conv_transpose, GroupNorm,
upsample/interpolate, pixel_shuffle, einsum, baddbmm, roll.** channels_last conv is rejected
("Neuron only supports contiguous/preserve memory format").

### 1.2 The public derivative: `libtorch-neuronx-lite` and vLLM's two routes

`libtorch-neuronx-lite` ("Custom torch-neuronx runtime for vLLM inference", Development Status
Alpha, proprietary licence) is on the public index for torch 2.10, 2.11, 2.12 and 2.13 and is installed
in this host's AMI vLLM venv (`torch 2.11.0 + torch-xla 2.11.0 + libtorch-neuronx-lite 2.11.0.1.0.1284 +
neuronx-cc 2.27.5334.0 + nki 0.6.0`; note that this particular build predates the native lane). It
renames PrivateUse1 to `neuron`, sets `NEURON_LOGICAL_NC_CONFIG=2`, requires torch-xla (NKI's torch
glue still imports `torch_neuronx.pyhlo` / `xla_impl`), and has two compile lanes:

- `neuron_libtorch` (default "XLA route"): Dynamo FX graph -> HLO via `torch_xla.core.xla_builder`
  -> `neuronx-cc --framework XLA` -> NEFF executed by Lite's runtime; distributed backend `gloo` for
  the control plane, collectives compiled into the NEFF from registered replica groups.
- `neuron_native_lite` ("native route", `VLLM_NEURON_BACKEND=neuron_native`,
  `NEURON_EXECUTION_BACKEND=native`): the **vendored TorchNeuron compiler** (`_compiler/neuron_dynamo_backend`,
  torch-mlir, StableHLO) used compile-only, writing `module.mlir` and calling `neuronx-cc compile module.mlir
  --framework XLA --target <trn2> --model-type transformer --lnc 2 -O1 --auto-cast=none`; execution by Lite;
  distributed backend `cpu:gloo,neuron:neuron` (composite, dispatch by tensor device). Its code says the
  `neuron` backend name "stays reserved for the real torch-neuronx package", and with
  `TORCH_NEURONX_DYNAMO_BACKEND_ONLY=1` Lite delegates `torch.compile(backend="neuron")` to
  `torch_neuronx.neuron_dynamo_backend`, i.e. acts purely as the runtime under TorchNeuron proper.

The plugin reaches Lite through one compatibility module and uses **private** Lite APIs for the three
things Difflet would also need: device count (`_compiler.device_count()`), kernel registration
(`_compiler.nki_op`), and replica-group registration (`_compiler.distributed.mesh_registry._MESH_REGISTRY`).

This matters for Difflet in two ways: Lite is the only *public* artifact of the native stack today
(installable without approval), and it shows the exact wiring AWS uses in production serving for a
diffusion model. It is also Alpha, vLLM-shaped, and its private APIs are not a contract.

### 1.3 AWS's own video DiT on this stack (vllm-omni-neuron, Wan 2.2)

From the plugin's code and `docs/model-dev/*`:

- Device `torch.device("neuron", local_rank)`; **one worker process per logical core**, spawned by
  vLLM-Omni's multiprocess executor (not torchrun); each process sees one core via
  `NEURON_RT_VISIBLE_CORES`; rank 0 does I/O. TP x CP x CFG mesh (recommended 64-core split TP=4,
  CP=8, CFG=2); `register_replica_groups(tp_size, cp_size)` must run before the first compiled collective
  or compilation fails with "replica id #N not seen in replica groups"; on `trn2.48xlarge` every group
  must form a ring on the 4x4 chip torus or the fabric fails with "no_hier no_mesh".
- About 15 compiled graph families, each `torch.compile(..., fullgraph=True)` with its own `model_name`
  (cache key) and `compiler_args`: DiT `--model-type=transformer --auto-cast=none -O1
  --hbm-scratchpad-page-size=2048`; VAE `--model-type=unet-inference --internal-max-instruction-limit=15000000`
  and **no** `fullgraph` when spatial tiling is on; TP-sharded UMT5 text encoder with the TP rank passed as
  a tensor so all ranks share one NEFF; a steady-state DiT graph that *takes* the cross-attention K/V
  cache and a first-step graph that *returns* it; separate VAE graphs for first/rest chunks; UniPC
  scheduler math compiled; a compiled CFG all-gather+combine.
- **No bucketing: shapes are static and padded** (text to 512 tokens, H/W to multiples of 16, odd
  token counts by one). Everything that varies per step or request is a device tensor (timestep,
  guidance scale, scheduler coefficients, rank); every Python-bool branch gets its own compiled object.
  Recompiles happen for H/W/frames/parallel layout/quant changes only. Cold compile at 480p/720p on 64
  cores is reported as **40-50 minutes**; compile timeout raised to 1800 s and the orchestrator's
  handshake timeout to 3600 s. There is no warm-up ("TODO: Implement custom warmup logic").
- Caches: `model_name` is the compile cache key; NEFF cache under `~/.cache/vllm/<hash>/` movable
  with `TORCH_NEURONX_NEFF_CACHE_DIR`; NKI cache `NKI_COMPILE_CACHE_URL` (a stale one "can replay a
  path from a previous session and surface as what looks like a kernel bug"; tests isolate it per
  worker); `VLLM_NEURON_CPU_MODE=1` traces and runs the whole model on CPU without a device.
- Kernels: the DiT bypasses the SDPA backend and calls nkilib `attention_cte` (fp32 softmax),
  `output_projection_cte` and `mlp` through `@nki_op` wrappers around `wrap_nki(kernel)[2]`; ring CP
  uses a vendored `ring_attention_const_max_fwd` NKI kernel with `nki.collectives` inside it, falling
  back to all-gather K/V + flash where the kernel cannot run; AdaLN is a vendored NKI kernel; the VAE's
  3x3x3 convs are nkilib `conv3d` kernels. Every kernel call site has a `can_run_kernel` check plus a
  shape check and a torch fallback. The FP8 path is **Trainium3-only** and still "in progress".
- Runtime quirks AWS worked around in code: eager dtype casts on `neuron` tensors fail, so casts live
  in a tiny compiled helper; `.contiguous()` on a permuted device tensor raises, so weight layouts are
  prepared on CPU before `.to(device)`; strided assignment into device tensors is unsupported;
  `F.gelu` is re-pointed at `torch.ops.aten.gelu.default` because the backend wraps it with a C builtin
  Dynamo cannot trace; `nn.Upsample(nearest-exact)` lowered to thousands of indirect DMA gathers and was
  replaced by `repeat_interleave`; a per-step backpressure sync (`torch.empty_like(src[:1]).copy_(src[:1])`)
  keeps the asynchronous runtime from running ahead; a direct `dist.broadcast` of a device tensor is
  dropped, so control data goes over the gloo CPU group.
- Accuracy: three tiers, cheapest first. `assert_close_three_way(fp32_ref, bf16_ref, bf16_neuron)`
  passes when **(Bhattacharyya coefficient >= 0.99 or sigma-ratio <= 1.0) and worst L-inf ratio < 5.0
  and worst L2 ratio < 3.0**, where each ratio is the Neuron error over the BF16-CPU error relative to
  the FP32 reference; then single-step denoised-latent tests; then per-frame SSIM against versioned
  golden video. "A small per-step approximation compounds across denoise steps into collapsed video",
  so tier 3 is not optional, and two implementations must share **identical initial latents** (an FP32
  CPU latent saved to disk), not just a seed.
- Reported effects: ring kernel raises self-attention MFU from about 44 % to about 75 %; CFG
  parallelism gives 1.8-2.0x per DiT step; VBench T2V 67.63 % on Trn2. No end-to-end latency figures
  are published.

### 1.4 The NKI library on the new stack

`nki-library` (`pip install nki-library`, package `nki_library`, Apache-2.0, dev version
`0.0.0.0dev0+3b542be2`, last commit 2026-08-28) requires **Python >= 3.12**, `neuronx-cc>=2.26` and
`nki>=0.5.0`; torch is a test-only extra. `neuronx-cc` bundles a copy as `nkilib` (what Difflet's
`nkilib.core...` imports resolve to today); installing the package replaces the whole `nkilib`
namespace, and "kernels from this package are not guaranteed to be compatible with the latest release
of the Neuron compiler".

Neither `nki-library` nor `nki-samples` contains any native-PyTorch integration: zero hits for
`nki_op`, `torch.library`, `register_fake`, `torch.compile`, `device="neuron"`. The kernel-side contract
is uniform and wraps cleanly: plain `@nki.jit`, HBM tensors in, outputs allocated in the kernel with
`nl.ndarray(..., buffer=nl.shared_hbm)` and returned, LNC chosen at launch with `kernel[2](...)`,
non-tensor arguments as trace-time constants (enums, `nl.NKIObject` dataclasses, tuples,
`ReplicaGroup`). Every kernel has a CPU torch reference with the same signature (`*_torch_ref`,
switchable at runtime with `NKILIB_USE_TORCH_REF=1`), and the tests carry shape/dtype output
descriptors that are ready-made fake/meta implementations.

Kernels relevant to Difflet, with the constraints that matter:

| Kernel | Status | Use | Constraints worth knowing |
|---|---|---|---|
| `attention_cte` | core | self/cross attention (what Difflet uses today) | d <= 128 documented (512 in code), seq up to 36 K, batch folds heads, fp32 softmax; SWA and in-kernel CP causal-only; LNC2 shards on batch |
| `attention_const_max` | experimental | non-causal attention for QK-RMSNorm models (FLUX/Qwen-Image/HunyuanVideo-style) | d == 128, Sk % 128 == 0, requires std(Q), std(K) <= 1; test shapes (N=5, Sq=675, Sk=43264) look like a TP-sharded video DiT |
| `ring_attention_spmd_fwd` | experimental | ring CP | MHA only, d <= 128, `ncc.collective_permute_implicit` inside; known `NCC_ISCH900` on trn2 for large striped+packed configs; not available in the simulator |
| `qkv` | core | fused QKV with optional norm/RoPE/QK-norm | H % 128 == 0, H <= 24576, **I <= 4096** (fused QKV of H=3072 models is 9216, so TP >= 3), FP8 ROW/STATIC |
| `output_projection_cte` | core | o-proj (and a de facto bf16 GEMM) | B*S <= 131072, H <= 20705, tested only to N <= 17 heads, H % LNC == 0 |
| `mlp` | core | FFN incl. non-gated GELU via `skip_gate_proj` | H % 128 == 0; `Swish` means GELU-sigmoid approximation, not SiLU |
| `RoPE` / `rope_hf` | core | rotary | d_head 64/128; `rope_hf` needs seq % (128*lnc) == 0 and caller-provided outputs (mutating op) |
| `conv3d`, `conv3d_transpose`, `conv3d_temporal_unroll` | experimental | VAE convolutions (also Conv2d as D=1) | filters permuted to `[Kd,Kh,Kw,Cin,Cout]`; `lnc_shard` is a no-op being deprecated |
| `rmsnorm_quant_kernel` | core | RMSNorm + FP8 row/static quantisation | NO_NORM/RMS only; output `[B,S,H+4]` carries the fp32 scale |
| `cumsum` | core | what Difflet uses for mask bounds | last dim only, ~1e-2 abs error beyond 5 K |
| collectives (`all_gather_hbm_kernel`, `all_to_all_hbm_kernel`, ...) | experimental | in-kernel collectives | "collective src must be in shared_hbm"; rank counts 1/2/4/8/16/32k |

**Missing from the library:** AdaLN/modulation, GroupNorm, a standalone CTE LayerNorm/RMSNorm, a plain
bf16 GEMM entry point, upsample. **FP8 on Trainium2 means non-OCP `float8_e4m3` (max 240)**; OCP
`float8_e4m3fn` (max 448) and every MX (MXFP8/MXFP4) kernel are Trainium3-only in this library, which
matches Difflet's existing `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` flag and bears on the Wan 2.1
FP8 PTQ work and on the first-party MX kernels (section 3, rows 9-10).

---

## 2. Where Difflet stands today

Full inventory with file:line references was produced during this research; the facts that shape the
migration:

1. **The Trainium path is AOT, not lazy-tensor.** Every component goes through the vendored NxDI
   lifecycle in `difflet/backends/trainium/core/application_base.py`: `ModelBuilder.add(...)` with a
   `ShapeBucketedInputGenerator`, `builder.trace()`, `torch.jit.save(model.pt)`, presharded weights
   (`builder.shard_checkpoint`) hardlinked through the shared weight store, then `torch.jit.load` +
   `nxd_model.initialize(weights, start_rank)` and execution on **CPU tensors**. `xm.mark_step`,
   `xm.wait_device_ops` and `torch_xla.device()` are not used anywhere under `difflet/backends/trainium/`.
2. **Direct torch_xla use is confined to collectives inside traced graphs**: `xm.all_gather`,
   `xm.reduce_scatter`, `xm.collective_permute` (joint-MMDiT ring), `xm.all_to_all` (Ulysses), and
   `torch.distributed.new_group(..., pg_options={"xla_pg_options": {"mesh": ...}})` for the CP/CFG/DP axes
   (`core/parallel_mesh.py`, `core/modules/attention/attention_process_groups.py`). Everything else is
   `neuronx_distributed.parallel_layers` (TP linears, mappings, `parallel_state`).
3. **Compiler flags are hardcoded strings**, one `get_compiler_args()` per application
   (`--model-type=transformer -O1 --tensorizer-options='--enable-ccop-compute-overlap' --auto-cast=none
   --internal-hlo2tensorizer-options='--verify-hlo=true'`, plus `--lnc`, `--target`, the FP8 flag), reaching
   `neuronx-cc` through `ModelBuilder.add(compiler_args=...)`. `NEURON_CC_FLAGS` has zero uses.
4. **NKI enters graphs two ways**: nkilib kernels called inside traced forwards with an LNC grid
   (`attention_cte[2]`, `ring_attention_spmd_fwd[2]`, `attention_block_tkg`, `qkv`, `output_projection_cte`,
   `cumsum`), and eight first-party MX FP8 kernels (`nki_kernels/mx.py`) converted with
   `kernel[1]._to_subclass(nki.framework.torch_xla.TorchXlaKernel)`. Two HLO custom calls are also used:
   `AwsNeuronModuleMarkerStart/EndForward` (`core/layer_boundary_marker.py`) and
   `torch_neuronx.xla_impl.ops.RmsNorm` (`core/modules/custom_calls.py`).
5. **Process model**: one process, N NeuronCores (`NEURON_RT_NUM_CORES`, `NEURON_RT_VISIBLE_CORES`,
   `NEURON_RT_VIRTUAL_CORE_SIZE`), no torchrun (`runtime.py: supports_torchrun_mpmd=False`); stages run as
   subprocesses (`cli/runner.py`); DP is a router that spawns one full CLI per replica with disjoint core
   ranges and a unique `NEURON_RT_ROOT_COMM_ID` (`cli/dp/router.py`); the serving resident worker is a
   single spawned process holding the loaded NxD applications.
6. **Coupling**: `difflet/backends/trainium/` is 71 files / 23.4 K LOC; 43 files / 20.6 K LOC carry the
   NxDI fork banner (vendored `neuronx-distributed-inference 0.9.17334`). The `difflet.ops` surface
   (`difflet/ops/__init__.py`, 45 names; CPU and TPU reference backends exist) is honoured by model and
   layer code, **but** 111 import statements in 41 non-backend files still reach into
   `difflet.backends.trainium.*`: mostly `core.config` (`NeuronConfig`/`InferenceConfig`, used as the
   universal config object), `core.bucketing`, `core.application_base`, `core.multi_component_application`,
   `core.model_wrapper`, the serving adapters, and NxDI text-encoder model classes
   (`neuronx_distributed_inference` Llama / Qwen2-VL / Qwen3-VL) in the HunyuanVideo and Qwen-Image
   orchestrators.
7. **Toolchain on this host** (Difflet venv, Python 3.12.3): `torch 2.9.1+cu128`, `torch-xla 2.9.0`,
   `torch-neuronx 2.9.0.2.15.32035`, `libneuronxla 2.2.17544.0`, `neuronx-cc 2.26.6360.0+6f180f47`,
   `neuronx-distributed 0.19.28492`, `neuronx-distributed-inference 0.10.18399`, `nki 0.5.0`; system
   runtime `aws-neuronx-runtime-lib 2.34.10.0`, driver `2.30.2.0` (the Neuron 2.32 set). The compile cache
   key already hashes these versions (`difflet/pipeline/compile_cache.py`), so every artifact recompiles
   after the move regardless.

---

## 3. What has to be replaced

| # | Today (XLA / NxD path) | Native equivalent | Notes |
|---|---|---|---|
| 1 | `ModelBuilder.add/trace`, `torch.jit.save/load(model.pt)`, `nxd_model.initialize`, `BaseModelInstance`, `ModelWrapper.input_generator` (`core/application_base.py`, `core/model_wrapper.py`) | Build the `nn.Module` on CPU, `.to("neuron")` the shards, wrap `forward` in `torch.compile(backend=<neuron backend>, fullgraph=True, dynamic=False, options={"model_name": ..., "compiler_args": [...]})`, warm once per compile shape; persist nothing but the NEFF cache + a Difflet manifest | Replaces ~3 K LOC of lifecycle code with a thin eager/compile lifecycle; `load()` becomes "load weights to device + warm". `torch.compiler.save_cache_artifacts()` (present in torch 2.9) can make the "compile" step a portable bundle |
| 2 | Bucketed multi-shape NEFF via `ShapeBucketedInputGenerator` + NxD router (`core/bucketing.py`) | One compiled callable per canonical shape (dict keyed by `(H, W, F)`), each warmed at load; raise `torch._dynamo.config.recompile_limit` (default 8) or compile per-shape wrappers so Dynamo never evicts; `torch.compiler.set_stance("fail_on_recompile")` after warm-up, as AWS's LLM path does | Static shapes only; keep `canonicalize_shapes` and the serving shape-set membership check; pad instead of bucketing where AWS does (text length, token count parity) |
| 3 | Single process, `NEURON_RT_NUM_CORES=N` SPMD, `LOCAL_WORLD_SIZE` env written by `get_compiler_args()` | One process per logical core; `MASTER_ADDR/PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK` env; `dist.init_process_group(backend=...)` **before** any device touch (`"neuron"` on TorchNeuron proper, `"cpu:gloo,neuron:neuron"` on Lite's native lane); device = `neuron:<local_rank>` | Biggest structural change: CLI stages, DP router, serving worker all become process groups. `BackendCapabilities(requires_aot=False, single_process_multi_core=False, supports_torchrun_mpmd=True)` |
| 4 | NxD `parallel_state` + `torch.distributed.new_group(pg_options={"xla_pg_options": ...})` (`core/parallel_mesh.py`) | `init_device_mesh("neuron", (dp, cfg, cp, tp))` or `dist.new_group(ranks)` per axis from the backend-neutral `difflet.pipeline.parallel_mesh.MeshSpec` (already shared with the TPU backend); register every group's full world partition in the backend's mesh registry so compiled collectives are SPMD-identical across ranks | Honour the 4/8/16/32 replica-group alignment, contiguous-start constraints, and on `trn2.48xlarge` the torus-ring layout |
| 5 | `ColumnParallelLinear` / `RowParallelLinear` / `ParallelEmbedding` from `neuronx_distributed.parallel_layers` (`ops_impl/linear.py`) | Difflet-owned TP linears on plain `nn.Linear` shards + `all_reduce` / `all_gather` on the TP group (or DTensor `ColwiseParallel`/`RowwiseParallel`); AWS's Wan DiT uses raw parameters plus hand-written collectives, with row-parallel biases pre-divided by TP | The TPU backend already has a 252-line `ops_impl/linear.py` of this shape to crib from |
| 6 | NxD mappings (`gather_from_tensor_model_parallel_region_with_dim`, `reduce_scatter_to_sequence_parallel_region`, ...) and `xm.all_gather/reduce_scatter/all_to_all/collective_permute` (`ops_impl/collectives.py`, `ops_impl/attention.py`) | `torch.distributed` collectives inside compiled regions (`all_gather_into_tensor`, `reduce_scatter_tensor`, `all_reduce`, functional `all_gather_tensor`); `all_to_all_single` for Ulysses | `all_to_all_single` is tested in eager but was commented out of the compiled-collectives test; no `collective_permute`/`send`/`recv` at all; on Lite every device collective must be inside a compiled graph |
| 7 | `difflet.ops.attention` -> nkilib `attention_cte[2]` with mask bounds; `ring_attention` -> nkilib `ring_attention_spmd_fwd[2]`; joint ring via partial softmax + `collective_permute` | Default: nkilib `attention_cte` via a custom-op wrapper (AWS's choice for the Wan DiT), with `F.scaled_dot_product_attention` (built-in NKI flash, head dim 128/192) as the fallback; `attention_const_max` for QK-RMSNorm models; ring: NKI ring kernel with `nki.collectives` inside (`ring_attention_spmd_fwd` or AWS's `ring_attention_const_max_fwd`), `gather_kv` all-gather as the fallback | Ring CP is the single riskiest feature: it needs in-kernel collectives or point-to-point, and the native c10d backend has no send/recv; the joint-MMDiT ring (`collective_permute` + partial-softmax merge) needs re-expression as an in-kernel collective or an all-gather |
| 8 | `torch_neuronx.xla_impl.ops.RmsNorm` custom call; `AwsNeuronModuleMarker*` HLO markers | `torch.nn.functional.rms_norm` (fused `aten::_fused_rms_norm` on torch >= 2.9) / plain `nn.RMSNorm`; markers dropped (they exist to steer the HLO compiler; FX graphs have module scopes) | `layers/normalization.py:34` and `models/flux/modeling_flux.py:76-79` must stop importing the marker |
| 9 | 8 MX FP8 NKI kernels via `TorchXlaKernel` (`nki_kernels/mx.py`, `ops_impl/mx.py`); `split_along_dim0_kernel` | Same kernel bodies (NKI 0.6 `nki.*` namespace), registered as custom ops (`@nki_op` around a Python function that launches the kernel) with fake shape functions; call sites unchanged | Kernels must compile under NKI 0.6 (the venv has 0.5); the NKI library treats every MX kernel as Trainium3-only, so re-verify the trn2 MX path early; `neuronxcc.nki` legacy namespace is on the way out |
| 10 | FP8 PTQ via NxD `QuantizedColumnParallelLinear`/`QuantizedRowParallelLinear`, `quantize_traced_model_`, `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3` (`core/quant.py`, `wan/backbone.py`) | No `_scaled_mm` in the native suite; options are an NKI FP8 linear kernel registered as a custom op (nkilib `output_projection_cte`/`mlp`/`qkv` accept FP8 ROW/STATIC inputs; `rmsnorm_quant_kernel` produces the packed `[B,S,H+4]` activations), or weight-only FP8 storage with bf16 compute as a stop-gap | On trn2 only non-OCP e4m3 (max 240) exists; AWS's Wan 2.2 FP8 path is Trainium3-only and unvalidated; the FP8 verification plan in `docs/verification/2026-09-29-ptq-fp8-wan-plan.md` stays on the XLA path and on native becomes a kernel project |
| 11 | NxDI text encoders: `NeuronLlamaForCausalLM` (HunyuanVideo), `NeuronQwen2VLTextForCausalLM` / `NeuronQwen3VLTextForCausalLM` (Qwen-Image), forked validation utils (`utils/accuracy.py`, `utils/benchmark.py`) | HF `transformers` models run compiled on `neuron`, TP via Difflet TP linears or DTensor; AWS TP-shards UMT5 across all world ranks with the rank passed as a tensor so one NEFF serves every rank | Second-largest chunk of work; interim option is CPU text encoding |
| 12 | TeaCache adaptive probe = separate probe NEFF with `prev_mod` as aliased NEFF state (`core/teacache_probe.py`, `*/teacache_probe_fused.py`, CPU shadows) | Compute the probe signal inside the compiled DiT (extra output) and decide on host in Python (AWS's cache-dit integration does the same with a host `.item()` per step); eager makes the 51 ms probe-dispatch floor and the CPU shadow unnecessary | Simplification; the cache-key exclusions for runtime-only TeaCache modes stay |
| 13 | LTX-2 / HV1.5 segmented block streaming re-execs a worker per block (`ltx_2/segmented.py`, `hunyuan_video/segmented15.py`) | Stream blocks to device in eager, compile per block | Simplification |
| 14 | Shared weight store of presharded `tpN_sharded_checkpoint.safetensors` hardlinked per topology (`core/shared_weights.py`) | Still useful on the host side (shard once per TP degree, load per rank, prepare kernel weight layouts on CPU before `.to(device)`); on device, parameters are plain tensors shared by all compiled shape variants automatically | Keep; drop the "disabled when layout transformation is on" branch |
| 15 | `difflet serve` resident worker: one process, `application.load(path, start_rank_id=0, local_ranks_size=world)` | Worker = process group of `world_size` ranks (one per core) driven by rank 0 through a queue, exactly like vLLM's executor; readiness = all ranks warmed | `serving/engines/resident_worker.py` (1.3 K LOC) is the largest serving change |
| 16 | DP router spawns full CLIs with disjoint `NEURON_RT_VISIBLE_CORES` ranges (`cli/dp/router.py`) | Same idea, but each replica is a process group with its own `MASTER_PORT`/root comm id; or one big group with a `dp` mesh axis | Bit-identity across replicas should hold (same NEFF via the SPMD cache key) |
| 17 | Compile-cache key hashes `torch-neuronx, torch-xla, neuronx-cc, neuronx-distributed, nki, libneuronxla` (`compile_cache.py`) | Add `backend="neuron"` plus the native runtime's version (`torch_neuronx` or `libtorch-neuronx-lite`), `neuronx-cc`, `nki`, `nki-library`; drop the XLA packages for that backend via `_BACKEND_EXTRA_TOOLCHAIN_PACKAGES` | The additive-only policy already supports a non-default backend |
| 18 | Platform detection `torch_neuronx.utils.get_platform_target`; `NEURON_PLATFORM_TARGET_OVERRIDE` | `torch.neuron.get_device_name()` / `get_device_properties()` (Lite: `compile.platform.get_platform_target()`); keep the override env (the native compiler also reads it) | |
| 19 | `cli/prewarm.py` touching `privateuseone:i` to warm `nrt_init` | Device init happens in `init_process_group`; prewarm becomes "open the group early" | |
| 20 | Dependencies: `torch 2.9.1`, `torch-xla 2.9`, `torch-neuronx 2.9.0.2.15`, `libneuronxla`, `neuronx-distributed(-inference)`, `nki 0.5`, `neuronx-cc 2.26`, Python >= 3.10 | `torch 2.10+` (beta pins `~=2.10.0`; Lite builds go to 2.13), the native runtime (beta `torch-neuronx` or public `libtorch-neuronx-lite`, which still needs `torch-xla` of the same torch version), `nki 0.6`, `nki-library` (Python >= 3.12), `neuronx-cc` per the runtime's pin, Python >= 3.11 (the 2.27 compiler and nki 0.6 wheels are cp311-cp313 only) | Needs a second venv (`scripts/setup_env.sh` variant) and a new `requirements-neuron-native.lock` |

Explicitly **not** replaced: model definitions under `difflet/models/*/modeling_*.py` and `difflet/layers/*`
(they import `difflet.ops`, `torch`, `diffusers`, `transformers` only, per the DEVELOPER.md rule), the
pipeline/scheduler code, the CPU reference backend, the CLI surface, the serving API, the compile-cache
manifest machinery, the parallel-mesh math (`difflet/pipeline/parallel_mesh.py`), TeaCache controllers,
and the verification scripts' interfaces.

---

## 4. The changes to make (design)

### 4.1 A new backend, not an in-place rewrite

Add `difflet/backends/neuron/` registered as `DIFFLET_BACKEND=neuron` in `backends/registry.py`,
leaving `trainium` (XLA) intact until cut-over. Reasons: the two stacks cannot share a venv, the
verification campaign needs the XLA outputs as goldens, and the TPU backend already proved the
pattern (`backends/tpu/runtime.py`, `backends/tpu/core/application_base.py`, `backends/tpu/ops_impl/*`
mirror exactly the pieces needed here).

Put a single runtime-adapter module (`backends/neuron/runtime_api.py`, the role of vllm-omni-neuron's
`lite_compat.py`) between Difflet and the runtime: `compile_backend_name()`, `dist_backend()`,
`device_count()`, `register_replica_groups(name, groups)`, `nki_op(...)`/`launch_kernel(kernel, lnc)`,
`platform_target()`, `neff_cache_dir()`. Two implementations: TorchNeuron proper (`"neuron"`,
`init_process_group("neuron")`, `torch_neuronx.nki_op`, `torch_neuronx.distributed.mesh_registry`) and
Lite (`neuron_native_lite`, `"cpu:gloo,neuron:neuron"`, `libtorch_neuronx_lite._compiler.nki_op`,
`_MESH_REGISTRY`). Everything else in the backend talks to the adapter only.

Auto-detection must change: `registry._auto_detect_backend()` returns `trainium` whenever
`torch_neuronx` is importable, which is also true for the native package. Detect the native stack by
`hasattr(torch, "neuron")` / `torch.accelerator.current_accelerator().type == "neuron"` or by
`importlib.util.find_spec("torch_neuronx.neuron_dynamo_backend")` / `find_spec("libtorch_neuronx_lite")`.

### 4.2 Runtime and process model

```
class NeuronBackend(BackendRuntime):
    name = "neuron"
    capabilities = BackendCapabilities(
        requires_aot=False,                 # compile lazily, persist via NEFF cache
        single_process_multi_core=False,    # one process per logical core
        supports_torchrun_mpmd=True,
    )
    def prepare_runtime(self, parallel):
        # torchrun-style env already set by the launcher; init once per process
        dist.init_process_group(backend=runtime_api.dist_backend())  # sets NEURON_RT_VISIBLE_CORES
        init_parallel_mesh(parallel)            # dp/cfg/cp/tp groups + replica-group registration
```

Launcher: a small `difflet.cli.launch` that spawns `world_size` processes (multiprocessing spawn or
`torchrun`), one per logical core, with `LOCAL_RANK` = core index and one thread per process
(`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)` as AWS does); stages (`cli/runner.py`) become
process groups instead of single subprocesses; the DP router allocates disjoint core ranges and
ports per replica as today. Only rank 0 does I/O and host-side scheduling; other ranks execute the
same program (SPMD), which is also what makes CFG-parallel work (rank-dependent conditioning input).
Small control data (shapes, flags) travels over a gloo CPU group; device tensors are never passed to
eager collectives.

### 4.3 Component lifecycle (`backends/neuron/core/application_base.py`)

- `build_module()` on CPU from the HF checkpoint; shard TP weights on host (reuse
  `convert_hf_to_neuron_state_dict` and the shared weight store so every rank reads only its shard);
  prepare kernel weight layouts (transposes, fused QKV) on CPU; `.to("neuron")`.
- `compile(path)`: for each canonical shape, run the compiled forward once on synthetic inputs so the
  NEFF lands in the NEFF cache (pointed at `<DIFFLET_COMPILE_CACHE>/neff`), then write the Difflet
  manifest (`compile_cache.write_manifest`) plus the list of NEFF cache keys / a MegaCache bundle.
  `has_compiled_artifacts()` = manifest present and every key present in the cache.
- `load(path)`: weights to device, `torch.compile` wrappers created, warm each shape (cache hit, so
  seconds not hours), then `set_stance("fail_on_recompile")`; ready.
- `forward(*inputs)`: dispatch to the wrapper for the input's shape; raise the same
  `profile_mismatch` as today for unknown shapes. Timestep, guidance scale, scheduler coefficients and
  rank are tensors; Python-bool branches (first chunk / rest, tiled / untiled) get separate compiled
  objects.
- Keep `NeuronConfig`/`InferenceConfig` as the configuration objects but strip them of NxD-only fields
  behind a backend check; 21 non-backend files import them, so renaming would be churn without benefit.

Compiler flags: translate each application's `get_compiler_args()` into
`options={"compiler_args": [...]}` where the runtime supports it and into `NEURON_CC_FLAGS` otherwise;
keep `--model-type=transformer -O1 --auto-cast=none` for DiTs and `--model-type=unet-inference` for VAEs,
as AWS does for Wan 2.2 (plus `--hbm-scratchpad-page-size=2048` paired with
`NEURON_SCRATCHPAD_PAGE_SIZE=2048`).

### 4.4 `difflet.ops` on native (`backends/neuron/ops_impl/`)

| Module | Implementation |
|---|---|
| `collectives.py` | `MeshSpec`-driven groups; `gather_tp_dim` = `all_gather_into_tensor` + `cat`; `reduce_tp` = `all_reduce`; `scatter_*` = local slice by rank; `SPMDRank` = rank as a device tensor; `get_cfg_rank_spmd`/`get_cp_rank_spmd` = `MeshSpec.axis_rank(rank)`; all of it designed to be called inside compiled regions |
| `linear.py` | `ColumnParallelLinear`/`RowParallelLinear`/`ParallelEmbedding` on `nn.Linear` shards with the TP collectives above (gather-output / input-is-parallel semantics identical to NxD) |
| `attention.py` | `attention`/`cross_attention` = nkilib `attention_cte` custom op with mask-to-bounds, SDPA fallback; `ring_attention`/`joint_ring_attention` = NKI ring kernel with in-kernel collectives or all-gather fallback; `ulysses_attention` = `all_to_all_single` head/sequence redistribution |
| `norm.py` | `nn.RMSNorm` / `F.rms_norm`, `nn.LayerNorm` |
| `embeddings.py` | `apply_rotary_emb` as plain torch (the CPU reference already is) |
| `mx.py` | the eight MX kernels registered as custom ops, CPU reference for fake shapes |
| `platform.py` | `hardware` enum + `get_platform_target()` via the adapter with the override env |

Fallback rule: wherever the CPU reference implementation is plain torch, start from it on `neuron`
(eager fallback is free), then specialise with a kernel only after an A/B shows it matters. Every
kernel call site gets a capability check plus a shape check and a torch fallback so CPU-mode tracing
and unit tests keep working (AWS's `can_run_kernel` pattern).

### 4.5 Kernels

Port the first-party kernels to NKI 0.6 (`import nki`, `nki.language`, `nki.isa`) and register each
as a custom op: `@nki_op("difflet::<name>", mutates_args=())` around a Python function that launches
`kernel[2](...)` (never around the raw `@nki.jit` kernel, which would make Dynamo execute it during
fake-tensor tracing), with a meta/fake function copied from the kernel's shape/dtype output descriptor.
Non-tensor kernel arguments must be hashable constants (tuples, frozen dataclasses). Prefer library
kernels where they exist (`attention_cte`, `output_projection_cte`, `mlp`, `RoPE`, `conv3d`,
`cumsum`), mind their limits (`qkv` I <= 4096 forces TP >= 3 for H=3072 models; `output_projection_cte`
tested to 17 heads), and keep LNC=2 launch semantics from one environment variable. Adopt a per-kernel
simulator test (`nki.simulate(kernel)[lnc]`, `NKI_SIMULATOR=1`) so kernel bring-up does not need the
device; in-kernel collectives are not simulatable, so ring kernels need the device.

### 4.6 Serving, CLI, DP

- Resident worker: rank 0 owns the HTTP engine queue; it broadcasts request tensors to the group and
  gathers outputs; the warm-up and shape-set checks stay. Readiness only after all ranks warmed.
- `difflet compile|generate|run|serve`: unchanged flags; the backend selects the launcher.
- DP: unchanged UX; replica = process group.
- TeaCache: fixed-cadence unchanged; adaptive mode reads the probe from an extra compiled output.

### 4.7 Toolchain and repo plumbing

New venv recipe (`scripts/setup_env_native.sh`, `requirements-neuron-native.lock`): Python 3.12,
torch per the runtime (2.10 for the beta wheel; 2.11-2.13 for Lite), the native runtime, `nki 0.6`,
`nki-library`, `neuronx-cc` per the runtime, `diffusers`/`transformers` re-validated against the newer
torch. `pyproject.toml` `requires-python` bumps to >= 3.11 for the native extra. Compile-cache key: add
the backend and native toolchain packages. Unit tests: the `*_cp_mode_threading` stubs that fake
`torch_xla` need a native twin.

---

## 5. Verification after the migration

Freeze the baseline first: generate and commit (under `artifacts/`) the XLA-path outputs for the
canonical verification shapes of every model (`scripts/verify_cli.py` shapes), with the exact initial
latents saved as FP32 CPU tensors, plus per-step DiT latents for one prompt per model. These are the
goldens.

| Tier | What | Pass criterion | Tooling |
|---|---|---|---|
| 0 | Host and toolchain: venv, `torch.neuron.is_available()`, `init_process_group` on 4 ranks, `neuronx-cc --version`, NEFF and NKI cache dirs writable | all green | extend `scripts/env_check.sh` |
| 1 | Unit tests on CPU (`DIFFLET_BACKEND=cpu`) still pass; new backend unit tests for mesh/groups/launcher/manifest/adapter | 100 % | `scripts/test_unit.sh` |
| 2 | Op parity: every `difflet.ops` symbol, native vs CPU reference, bf16 and fp32, including TP=2/4 collectives and the eight MX kernels (simulator + device) | three-way rule: (BC >= 0.99 or sigma-ratio <= 1) and L-inf ratio < 5 and L2 ratio < 3 | new `tests/numerical/test_ops_native_parity.py` |
| 3 | Component parity: text encoder, DiT, VAE each against diffusers CPU (FP32 ref, BF16 expected, BF16 Neuron actual) at the canonical shape | same rule | per-model component tests |
| 4 | Single-step DiT parity against the frozen XLA per-step latents with identical latents/timestep | cosine >= 0.999 (today's ring-parity gate) and the three-way rule | reuse `*_ring_parity_smoke.py` harness shape |
| 5 | End-to-end: image models byte-compare with tolerance / SSIM, video models per-frame SSIM against goldens; adaptive TeaCache and `--quant fp8` as separate rows | SSIM floor agreed per model; no NaN; identical output across repeated runs | `scripts/verify_cli.py` |
| 6 | Parallelism matrix on trn2 (4 cores): `tp4 tp2cp2 tp2cp2ring tp2cp2ulysses tp2cfg tp4sp dp2tp2` for all five models; expected-outcome table written before running | PASS/SKIP/XFAIL per cell as declared | `scripts/run_cells.sh` under `supervise.sh` |
| 7 | DP bit-identity dp=2 vs dp=1 | byte-identical images / latents | `scripts/verify_dp_correctness.py` |
| 8 | Serving: `difflet serve --shapes A,B`, `/ready`, every compiled shape, one off-set shape -> `400 profile_mismatch`, 1 h soak with mixed shapes, Dynamo recompile counter flat after warm-up, CPU-fallback op list empty | as today | `scripts/serve_smoke.sh` + `get_dynamo_metrics()` / `get_fallback_ops()` |
| 9 | Performance: per-step DiT latency (`benchmark/step_realloop.py`), cold and warm e2e (`benchmark/cold_warm_e2e.py`), compile time per component and per shape, peak HBM, graph-break count zero for DiTs | within an agreed band of `benchmark/trn2/*.md` (propose: per-step <= 1.10x XLA; warm e2e faster because weights stay resident) | existing harness + new native adapter |
| 10 | Cache portability: compile on host A, load on host B with the same lockfile, no recompile; stale-NKI-cache drill (clear `NKI_COMPILE_CACHE_URL`, recompile, compare) | no compiles at load; identical outputs | `scripts/hardlink_proof.sh` analogue for NEFF keys |

Record every cell with compile seconds, generate seconds, toolchain versions and the exact command,
in the evidence-doc format of `.claude/skills/difflet-device-verify`. One bug, one commit.

---

## 6. Impact assessment

**Architecture and code.** The 23 K LOC Trainium backend shrinks: the NxD lifecycle, bucketing router,
world checks, compile-allocator and compile-retry workarounds, snapshot/metaneff tooling and the HLO
custom calls go away; what replaces them is a few thousand lines of eager/compile lifecycle, TP layers,
collectives, the runtime adapter and the launcher. The 20 K LOC of vendored NxDI forks (text-encoder
LLMs, validation utils) either get replaced by HF models or dropped. Model code is untouched by design.

**Performance (expected, to be measured in the spike).** `torch.compile` with `fullgraph=True` yields one
fused NEFF per component per shape, i.e. the same unit of work the compiler optimises today, so per-step
DiT latency should be comparable; AWS runs Wan 2.2 this way in production and reports ring-attention MFU
of ~75 % and a 1.8-2.0x CFG-parallel step speed-up. Risks: Dynamo-side Python overhead per call (small
for a 30-step loop), host round-trips for TeaCache decisions, eager fallbacks for any op the compiler
rejects, losing the CTE-specific NKI attention tuning until kernels are re-registered, and cold compiles
AWS measures at 40-50 minutes for Wan 2.2 at 480p/720p on 64 cores. Gains: weights stay on device across
shapes and requests (warm e2e today is load-dominated: 63 s warm for Qwen-Image, 144 s for HunyuanVideo in
`benchmark/trn2/*.md`, mostly weight reload), no AOT trace step, no `ModelBuilder` compile-time
multiplication across buckets, no `libjemalloc` and compile-retry hacks.

**Memory.** Eager tensors live in NRT HBM through a caching allocator with no segment caching
(`memory_reserved()` is always 0 in the suite); all shape variants share parameters. One process per
core means `world_size` copies of host-side Python state, so host RAM use grows with world size
(124 GB on this box is ample). The VAE decode is the HBM hotspot (AWS needs spatial tiling at 720p).

**Features at risk.** Ring CP (needs in-kernel collectives or send/recv; MHA-only kernels), Ulysses CP
(`all_to_all` topology limits, compiled `all_to_all_single` untested), FP8 PTQ (no scaled matmul; only
non-OCP e4m3 on trn2), MX kernels (NKI 0.6 port; library marks MX Trainium3-only), VAE ops
(conv3d/GroupNorm/upsample untested in the suite; channels_last rejected; `nn.Upsample` lowered badly in
AWS's experience), adaptive TeaCache (becomes easier), LTX-2 segmented mode (becomes easier), DP
bit-identity (should hold).

**Numerics.** bf16 everywhere with `--auto-cast=none` stays the policy; expect small per-op deltas
from different fusion decisions (StableHLO front end, different decompositions). The three-way
methodology is there to keep "ordinary bf16 noise" from being misread as a defect, and the end-to-end
SSIM gate is there because per-step noise compounds over the denoising loop.

**Operations and dependencies.** Two venvs during the transition (the native package replaces the
`torch_neuronx` namespace), Python >= 3.11 (3.12 for `nki-library`), torch 2.10+ with
diffusers/transformers re-validation, a compiler version change (section 8), a full compile-cache
rebuild, a one-process-per-core process model that changes how `difflet serve` and the DP router allocate
cores, and an upstream that is either closed beta (TorchNeuron) or Alpha with private APIs (Lite) and whose
GitHub-first development was "planned for early 2026" but whose repository is still private. The
compensating factor: this is the stack AWS is converging vLLM, vLLM-Omni, TorchTitan and NKI on, so
Difflet stops carrying a vendored NxDI fork and gains ecosystem kernels.

**Schedule risk.** Highest-uncertainty items, in order: runtime access and its torch/compiler pins;
ring CP; FP8; VAE op coverage; serving worker rewrite.

---

## 7. Migration plan

Phase gates are hard: do not start the next phase until the exit criteria are met and recorded.

| Phase | Scope | Exit criteria | Effort (one engineer, rough) |
|---|---|---|---|
| **0. Access and spike** | Request the native `torch-neuronx` beta wheel from the account team; meanwhile build the second venv on the public stack (Python 3.12, torch 2.11-2.13, `libtorch-neuronx-lite` native lane, `nki 0.6`, `nki-library`, `neuronx-cc 2.27.5334.0`) mirroring vllm-omni-neuron's setup guide; port the tiny-Wan PTQ probe model (`scripts/ptq_fp8_device_probe.py`) to `neuron` eager and to `torch.compile`; TP=2 and TP=4 process groups; one `attention_cte` custom op, one RMSNorm, one compiled all_gather, one MX kernel under `@nki_op`; a conv3d/GroupNorm/upsample VAE block; measure compile time and per-step latency vs the XLA artifact; confirm which `neuronx-cc` the runtime needs and whether it compiles the FLUX CLIP encoder | A written spike report: op coverage hits, compile times, step latency, collective constraints observed, compiler version decision, runtime decision (Lite vs beta), go/no-go | 2-3 weeks |
| **1. Backend scaffold** | `backends/neuron/` runtime, registry entry and detection, runtime adapter (Lite + TorchNeuron implementations), launcher, mesh/groups with replica-group registration, lifecycle base class, manifest/NEFF-cache integration, cache-key changes, env lock | `DIFFLET_BACKEND=neuron` runs a CPU-built dummy module on 4 ranks through compile/load/forward with a manifest; unit tests | 2 weeks |
| **2. Ops and kernels** | `ops_impl/*` per section 4.4; TP linears; attention custom op + SDPA fallback; `gather_kv` CP; Ulysses; MX kernels and `split_along_dim0` as custom ops; NKI simulator tests; op parity tier 2 | Tier 2 green for all `difflet.ops` symbols at TP=1/2/4 | 4-6 weeks |
| **3. Wan 2.1 end to end** | Wan backbone, UMT5 text encoder (TP-sharded, compiled), VAE (rank 0, `unet-inference` flags, tiling without `fullgraph`); multi-shape; CFG-parallel; TeaCache cadence + adaptive; `difflet generate` and `difflet serve` on the native worker; tiers 3-5, 8 for Wan | Wan tiers 3-5 pass; serving smoke passes; per-step latency within band | 4-6 weeks |
| **4. Remaining models** | FLUX (CLIP/T5, TAEF1), Qwen-Image (Qwen2.5-VL text encoder replaces NxDI class), HunyuanVideo (Llama + CLIP encoders replace NxDI class; VAE), LTX-2 (segmented path simplified; MX path); ring CP kernel port; FP8 kernel decision | Tiers 3-5 per model; parallelism matrix cells declared | 6-10 weeks |
| **5. Campaign and cut-over** | Full `difflet-device-verify` campaign (tiers 6-10) on trn2 (and trn3 if available); perf report vs `benchmark/trn2`; docs (README, QUICKSTART, DEVELOPER, lockfiles); flip the default backend; keep `trainium` behind a flag | Evidence doc committed; default flipped; release notes | 3-4 weeks |
| **6. Decommission** | Remove the XLA backend, NxDI forks, torch_xla stubs in tests, old venv recipes, once two releases have shipped on native | tree free of `torch_xla`/`neuronx_distributed` imports | 1-2 weeks |

Parallelisable: phases 2 and 3 can overlap once the attention/linear ops exist; model ports in phase 4
are independent of each other.

Decision points:

1. End of phase 0: which runtime to build on first. Recommendation: code against the TorchNeuron API
   shape through the adapter, run on Lite's native lane until the beta wheel (or the public PyTorch 2.10
   release) is available, and treat Lite's private APIs as a temporary dependency pinned by exact version.
2. End of phase 0: go/no-go versus waiting for the public release; compiler version.
3. End of phase 3: whether ring CP ships via an NKI-collectives kernel or is documented as
   `gather_kv`-only on native for the first release.
4. End of phase 3: FP8 strategy (NKI linear kernel with non-OCP e4m3 on trn2, weight-only FP8, or defer).

Questions for the AWS Neuron account team (ask during phase 0):

- Access to the native `torch-neuronx` wheel; its torch pin (2.10 only, or 2.11-2.13 like Lite);
  Python 3.12 wheels; the planned Neuron release for GA; whether third parties should build on Lite's
  native lane in the meantime and how stable its `nki_op` / mesh-registry entry points are.
- The `neuronx-cc` version the runtime requires; whether the public compiler accepts StableHLO input
  (section 8); status of the FLUX CLIP encoder ICE on 2.27.
- Roadmap for `send/recv` or `collective_permute`, `all_to_all_single` inside compiled graphs, replica
  groups of size 2 under LNC=2, and `nki.collectives` inside custom-op kernels.
- `_scaled_mm` / FP8 GEMM plans; OCP FP8 and MX on Trainium2 for `torch.compile` graphs.
- Op coverage for conv3d, GroupNorm, upsample/interpolate (VAE decoders); channels_last.
- Whether `torch.compiler.save_cache_artifacts()` bundles are the supported way to ship precompiled
  NEFFs between hosts.

---

## 8. The compiler question

**Current compiler.** `neuronx-cc 2.26.6360.0+6f180f47` (Neuron SDK 2.31 line), from
`requirements-neuron.lock`, pinned on purpose: commit `23674df` "pin neuronx-cc 2.26 — 2.27 ICEs on the
Flux CLIP encoder". It is driven by `libneuronxla` inside NxD `ModelBuilder.trace()`, fed XLA HLO
(`--framework XLA`, `.hlo.pb`), with per-application flag strings (`--model-type=transformer`, `-O1`,
`--auto-cast=none`, `--tensorizer-options=...`, `--internal-hlo2tensorizer-options=...`, `--lnc 2`,
`--target trn2`, the FP8 `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`). The runtime on the host is
already from the 2.32 line (`aws-neuronx-runtime-lib 2.34.10.0`), so compiler and runtime are skewed by
one release today.

**Does native change the compiler?** No new compiler binary: TorchNeuron's eager kernels and its Dynamo
backend both spawn `neuronx-cc` found on `PATH` (the develop suite's `NeuronCompiler::CompileHloToNeff`
and `neuronx_cc_wrapper` tests; Lite's `CompilerSubprocess.find_neuronx_cc()`), with `--target` and
`--lnc` derived from the instance, `-O1` by default, and extra flags from `NEURON_CC_FLAGS` /
`compiler_args`. The pieces that *do* change:

1. **Input IR.** The native backend lowers FX -> torch-mlir -> StableHLO and hands the compiler a
   `module.mlir` (Lite writes it and still passes `--framework XLA`; the develop suite's own CLI wrapper
   distinguishes `--framework XLA` for `.hlo.pb` from `--framework StableHLO` for `.mlir`). The documented
   `--framework` value in the 2.32 CLI reference is `XLA` only, and the installed 2.26 `--help` lists
   `--framework {XLA}` only, while both the 2.26 and 2.27 frontend binaries (`driver/jobs/Frontend`,
   `starfish/bin/hlo2penguin`, `hlo_convert`) contain StableHLO support strings. The develop suite carries
   an xfail reading "Latest public compiler does not support stableHLO lowerings" for the StableHLO-direct
   path. Conclusion: StableHLO ingestion by the public compiler is real but version-sensitive; the spike
   must confirm it on the exact version the runtime pins.
2. **Version.** AWS's Wan 2.2 plugin pins `neuronx-cc==2.27.5334.0` (Neuron 2.32) with Lite 2.11; the
   `nki-library` package requires `neuronx-cc>=2.26`; the beta wheel will have its own requirement. Plan
   on **moving to 2.27 or newer** for the native path, which means re-testing the FLUX CLIP encoder ICE
   (and, on native, the CLIP encoder could run eager or with graph breaks around the offending op as a
   workaround).
3. **Flag plumbing.** Per-application flag strings become `options={"compiler_args": [...]}` (where
   supported) or `NEURON_CC_FLAGS`; `--model-type`, `--auto-cast`, `-O`, `--hbm-scratchpad-page-size` carry
   over unchanged; `--lnc`/`--target` are derived automatically; the `--internal-*` tensorizer options
   should be re-justified one by one on the new front end.
4. **Python.** `neuronx-cc 2.27.5334.0`, `2.26.6360.0` and `2.25.3371.0` wheels are cp311/cp312/cp313 only;
   the Difflet venv is 3.12, so no interpreter change on this host, but the repo's "Python 3.10+" claim
   must become 3.11+ (3.12 with `nki-library`), and the hardcoded `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
   paths in `benchmark/models.py`, `benchmark/step_realloop.py`, `tests/numerical/test_mx_ops_neff.py`,
   `scripts/ptq_fp8_device_probe.py` should be retired at the same time.

NKI is the other compiler in play: the `nki` package (0.6.0 in Neuron 2.32; 0.5.0 in the Difflet venv) is
an MLIR-based kernel compiler that produces kernel IR the Neuron compiler back end consumes, and it has its
own cache (`NKI_COMPILE_CACHE_URL`). The MX and split kernels must be validated against 0.6, and the
legacy `neuronxcc.nki` namespace used by some research scripts is being retired.

---

## Appendix A. Native API cheat sheet (from the develop test suite and the Lite/vLLM code)

```python
# device & eager
import torch, torch_neuronx            # explicit import only needed on torch < 2.9
x = torch.randn(4096, 4096, device="neuron")
y = (x @ x.T).softmax(-1)              # eager: per-op NEFFs, fused by adaptive eager
torch.neuron.synchronize()             # the mark_step analogue: wait for queued ops + compiles

# compile (static shapes only)
f = torch.compile(model, backend="neuron", fullgraph=True, dynamic=False,
                  options={"model_name": "wan_transformer"})
# Lite's native lane: backend=libtorch_neuronx_lite.compile.native_backend.register()  (-> "neuron_native_lite")
#   options={"model_name": ..., "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1",
#            "--hbm-scratchpad-page-size=2048"], "compiler_workdir": ...}

# distributed: one process per logical core, before touching the device
# env: MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE
import torch.distributed as dist
dist.init_process_group(backend="neuron")             # TorchNeuron proper
# dist.init_process_group(backend="cpu:gloo,neuron:neuron")   # Lite native lane
mesh = dist.device_mesh.init_device_mesh("neuron", (cfg, cp, tp), mesh_dim_names=("cfg", "cp", "tp"))
tp_group = mesh["tp"].get_group()
# register every group's full partition for SPMD lowering of compiled collectives:
#   TorchNeuron: torch_neuronx.distributed.mesh_registry ; Lite: _compiler.distributed.mesh_registry._MESH_REGISTRY[name] = groups
dist.all_reduce(t, group=tp_group)                     # inside a compiled region on Lite; eager OK on TorchNeuron
from torch.distributed._functional_collectives import all_gather_tensor

# NKI kernel as a custom op (eager + compile)
import nki, nki.language as nl, nki.isa as nisa
from torch_neuronx import nki_op       # Lite: libtorch_neuronx_lite._compiler.nki_op + nki.nki_hop.wrap_nki

@nki.jit
def add_kernel(a, b):
    sa = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf); nisa.dma_copy(dst=sa, src=a)
    sb = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf); nisa.dma_copy(dst=sb, src=b)
    sc = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sc, data1=sa, data2=sb, op=nl.add)
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm); nisa.dma_copy(dst=out, src=sc)
    return out

@nki_op("difflet::nki_add", mutates_args=())           # wrap the launcher function, not the raw kernel
def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return add_kernel[2](a, b)                         # LNC=2 launch grid
```

Environment variables seen in the suite, Lite and the Wan plugin: `NEURON_CC_FLAGS`,
`NEURON_COMPILER_OPT_LEVEL`, `NEURON_LOGICAL_NC_CONFIG`, `NEURON_RT_NUM_CORES`, `NEURON_RT_VISIBLE_CORES`,
`NEURON_RT_ROOT_COMM_ID`, `NEURON_PLATFORM_TARGET_OVERRIDE`, `NEURON_SCRATCHPAD_PAGE_SIZE`,
`NEURON_LAUNCH_BLOCKING` (sync mode), `NEURON_FALLBACK_ENABLED`, `NEURON_EXECUTION_BACKEND` (Lite: lite|native),
`VLLM_NEURON_BACKEND=neuron_native`, `TORCH_NEURONX_DYNAMO_BACKEND_ONLY`, `TORCH_NEURONX_NEFF_CACHE_DIR`,
`TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR`, `TORCH_NEURONX_NEFF_DISABLE_CACHE`, `TORCH_NEURONX_HLO_CACHE_DIR`,
`TORCH_NEURONX_DEBUG_DIR`, `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS`, `TORCH_NEURONX_DUMP`,
`TORCH_NEURONX_ENABLE_CONCATENATION`, `TORCH_NEURONX_ENABLE_PROLOGUE`, `TORCH_NEURONX_MLIR_ATEN_OPS`,
`TORCH_NEURONX_SPMD_DISABLE`, `TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION`,
`TORCH_NEURONX_DYNAMO_DISABLE_CPU_AUTOCOPY`, `TORCH_NEURONX_METRICS_ENABLED`, `NKI_COMPILE_CACHE_URL`,
`NKI_SIMULATOR`, `NKILIB_USE_TORCH_REF`, `NEURON_LIBTORCH_CACHE_ROOT`, `NEURON_LIBTORCH_COMPILATION_TIMEOUT`,
`NEURON_LIBTORCH_CPU_MODE`.

## Appendix B. References

- Native PyTorch for AWS Trainium (overview):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html
- About PyTorch on AWS Neuron (three implementations, package-name note):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/about/index.html
- Transition announcement (PyTorch 2.9 last XLA version):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-transition-pytorch-trainium.html
- What's New: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html
- Compiler release notes (2.27.5334.0 in Neuron 2.32.0):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/compiler.html
- Compiler CLI reference (`--framework XLA`):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/index.html
- OSS repositories (TorchNeuron private beta, nki-library, vllm-neuron):
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/oss/index.html
- TorchNeuron repository (private beta): https://github.com/aws-neuron/torch-neuronx
- vLLM Omni Neuron (Wan 2.2 reference on the native stack): https://github.com/aws-neuron/vllm-omni-neuron
- NKI Library: https://github.com/aws-neuron/nki-library ; NKI samples: https://github.com/aws-neuron/nki-samples
- torchtitan RFC on NKI kernels via `@nki_op`: https://github.com/pytorch/torchtitan/issues/2391
- Public pip index: https://pip.repos.neuron.amazonaws.com/ (`torch-neuronx`, `neuronx-cc`, `nki`,
  `libtorch-neuronx-lite`, `vllm-neuron`, `vllm-omni-neuron`)
