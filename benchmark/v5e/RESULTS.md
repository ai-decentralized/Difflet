# Cloud TPU v5e (v5litepod-4) — difflet benchmark

Measured on a Cloud TPU VM: `v5litepod-4`, 4 × v5e chips, 2x2 topology,
16 GB HBM per chip, us-west4-a.

## Models measured

| model | detail | harness | resident weights |
|---|---|---|---|
| [Qwen-Image](qwen_image.md) | 1024x1024, 20 steps, tp=4 | `benchmark/adapters/tpu.py` | 9.62 GB per chip |
| [Wan 2.2 A14B](wan_2_2.md) | 480x832x9, 20 steps, tp=4, **single expert** | standalone runner (the TPU adapter is Qwen-specific) | 7.5-8.1 GB per chip |
| Wan 2.1 14B | 480x832x9, 20 steps, tp=4 | `benchmark/wan_tpu_run.py --model-dir <2.1 snapshot>` | 6.99 GB per chip (single transformer) |
| [HunyuanVideo](hunyuan_video.md) | 320x512x61, 20 steps, tp=4 | `benchmark/hunyuan_tpu_run.py` | 10.84 GB per chip |
| [LTX-2](ltx_2.md) | 480x704x49, 20 steps, tp=4 | `benchmark/ltx2_tpu_run.py` | 9.3 GB per chip (+2.4 GB video VAE on rank 0) |
| [FLUX.1-dev](flux_1_dev.md) | 1024x1024, 28 steps, tp=4 | `benchmark/flux_tpu_run.py` | 10.18 GB per chip (+0.3 GB VAE on rank 0) |

HunyuanVideo, LTX-2 and FLUX.1-dev were ported on 2026-09-11/12 (`docs/plans/2026-09-11-tpu-port-hunyuan-ltx2-flux.md`).

**Both models now use the Pallas fused attention kernel**, which needs Python
3.12; see [wan_2_2.md](wan_2_2.md) for the toolchain and the measurements. The
e2e figures below for Qwen-Image predate it and have not been re-run.

Qwen-Image runs through `benchmark/adapters/tpu.py`, the same harness and
metric schema as the other device folders. Wan does not yet: that adapter
drives `QwenImageServingStageAdapter` directly, so the Wan numbers come from a
standalone one-process-per-chip runner (`benchmark/wan_tpu_run.py`) using
difflet's own `WanOrchestrator` and the same frozen `MATRIX` config. Folding
Wan into the adapter is outstanding.

## Configuration — Qwen-Image

Wan 2.2's configuration and its TPU-specific rows are in
[wan_2_2.md](wan_2_2.md); the two differ in more than shape (text-encoder
dtype, VAE placement, expert residency).

The model and config rows are hardware-agnostic — any backend must match these
to reproduce. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Qwen/Qwen-Image` |
| HF revision (pinned) | `75e0b4be04f60ec59a75f475837eced720f823b6` |
| model type | `qwen_image` |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape | 1024 × 1024 |
| steps | 20 |
| guidance scale | 4.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |

All of the above come from `benchmark/models.py::MATRIX`, unchanged, so the
numbers sit beside the other device folders without adjustment.

### What is TPU-specific

| key | value |
|---|---|
| accelerator | Cloud TPU v5litepod-4, 4 chips, 2x2 topology, 16 GB HBM per chip |
| execution | eager (lazy XLA), one worker process per chip |
| attention | `torch.nn.functional.scaled_dot_product_attention` |
| matmul precision | `highest` (XLA's TPU default is reduced precision) |
| text encoder | host, bf16 |
| VAE | host between requests, moved to device for the decode |
| torch / torch-xla / libtpu | 2.9.0+cpu / 2.9.0 / 0.0.21 |
| diffusers / transformers | 0.38.0 / 5.15.0 |

`MATRIX`'s `config_label` reads "joint attention via attention_cte". That is
the **Trainium** recipe — `attention_cte` is a Neuron kernel and did not run
here. The tp=4/cp=1 part *is* accurate: the DiT is sharded across the 4 chips.

## Results — Qwen-Image, 1024x1024, 20 steps, tp=4, bf16

Re-measured 2026-08-24 with the fused attention kernel and on the cross-device
per-step rule (see the correction below).

| metric | value |
|---|---|
| `compile_seconds` | 0 (eager — see caveats) |
| `e2e_cold_seconds` | 80.7 |
| **`e2e_warm` mean (synced)** | **13.32 s** (n=3) |
| **`e2e_warm` mean (natural)** | **8.98 s** (n=3) |
| **`step_latency` mean (synced)** | **505.4 ms** (n=19) |
| `step_latency` (natural) | 289.9 ms |
| `peak_device_mem_gb` | 9.62 per chip |
| stages (warm) | text_encode 2.00 s / denoise 10.14 s / vae_decode 1.16 s |
| `output` | `[1, 4096, 64]` packed latents, finite |

The text encode was 5.74 s — a third of the e2e — because all four replicas
encoded the same prompt, in bf16, which this VM emulates (AMD EPYC, no
AVX512-BF16). One rank now encodes in fp32 and broadcasts: **2.00 s**, and
e2e warm 17.44 -> 13.32 s. See the 2026-08-24 entry in the plan.

**Validated through `difflet serve`**, not only the harness: health/ready 200,
a real 1024x1024 / 20-step HTTP generation in 8.75 s, an over-long prompt
returning 400 in 10 ms without hanging any replica, and a clean SIGTERM
shutdown with the chips released.

## Results — Wan 2.2 A14B, 480x832x9, 20 steps, tp=4, bf16, single expert

| metric | value |
|---|---|
| `load_seconds` | 4.2-37 (page-cache dependent; see caveats) |
| **`e2e_warm`** | **41.66 s** |
| **`step_latency` mean (synced)** | **608.5 ms** (n=38, std 2.1) |
| same loop, unsynced | no different basis — the loop round-trips to the host every step by construction |
| `peak_device_mem_gb` | 7.5-8.1 per chip |
| stages (warm) | text_encode 0.78 s / denoise 12.20 s / vae_decode 26.8-42.7 s |
| `output` | `[1, 16, 3, 60, 104]` latents, finite; coherent 9-frame video |

**Only one of the two 14.288B experts is resident** — each is 7.15 GB per chip
at tp=4 and two would not fit. Matches trn2's configuration, not the GPU rows.
**The host VAE decode (26.77 s) is now larger than the entire denoise loop**;
it is blocked on device by an unsupported negative index in `AutoencoderKLWan`
under torch_xla, not by memory.

## Cross-device — DiT per-step, all on the same rule

Device-synced inter-step deltas of a real generate loop, step 0 excluded
(`benchmark/harness.py::RealLoopStepTimer`).

| device | Qwen-Image | Wan 2.2 A14B | Wan 2.1 14B | HunyuanVideo | LTX-2 | FLUX.1-dev |
|---|---|---|---|---|---|---|
| B300 SXM6 | **140.0 ms** | **240.7 ms** | — | — | — | — |
| H100 PCIe | 297.7 ms | 553.7 ms | — | — | — | — |
| trn3 (4 cores) | 324.1 ms | — | — | — | — | — |
| trn2 (4 cores) | 447.1 ms | 554.8 ms | 554.8 ms | 850.6 ms | 441.8 ms | 267.6 ms |
| **v5e x4** | **508.4 ms** | **608.5 ms** | **608 ms** | **1006 ms** | **1453 ms** | **187 ms** |

### v5e vs trn2, whole request (same MATRIX rows, 2026-09-11/12)

| model | trn2 warm e2e | v5e denoise | v5e served request | where the v5e time goes |
|---|---|---|---|---|
| Qwen-Image 1024², 20 st | 63 s | 10.1 s (natural 5.7 s) | 7.6 s (cadence 2) | encode 2 s, denoise, decode 1.2 s |
| Wan 2.2 480×832×9, 20 st | 57 s | 12.2 s | 30.4 s | **24 s host VAE decode** |
| Wan 2.1 480×832×9, 20 st | 56 s | 12.2 s | 33.5 s | **24 s host VAE decode** |
| HunyuanVideo 320×512×61, 20 st | 144 s | 20.1 s | 191 s | **165 s host VAE decode** |
| LTX-2 480×704×49, 20 st | 58 s | 29 s (+20 s Gemma-3 fp32 encode) | 51 s | decode 0.2 s on chip; the encode |
| FLUX.1-dev 1024², 28 st | 35 s | 5.4 s (+3.3 s T5-XXL fp32 encode) | 10.1 / 9.3 s | decode 0.31 s on chip; the encode |

Per DiT step the v5e is 1.1× (Qwen, Wan) to 3.3× (LTX-2) slower than trn2's 4 cores and 1.4×
faster on FLUX (4 608 tokens of dense matmul, little attention); per request it is faster
wherever the decode is on the chip (Qwen, LTX-2, FLUX) and slower where it is not (the video VAEs on the host: Wan's 24 s, HunyuanVideo's 165 s). Putting those two VAEs on
the chip is the single biggest open item — LTX-2's went from >14 min (host, 121 frames) to
0.7 s.

TeaCache cadence 2 (probe-free) skips 5 of 20 steps and delivers **0.75× denoise on every
model** (Qwen 7.60 s, Wan 9.10 s, Hunyuan 15.07 s, LTX-2 DiT 22 s; FLUX at 28 steps skips 9 →
0.69×, 5.39 → 3.73 s); online-delta is a net loss on
the Qwen device-resident loop (per-full-step sync) and neutral on the host-looped video models.

On this basis v5e is last on both. Qwen-Image has a second, faster basis —
290.9 ms with no per-step sync, level with the H100's synced 298 ms — because
its denoise loop is device-resident. **Wan does not**: its orchestrator holds
the latents on the host in fp32 for UniPC, so every step ends in a `.cpu()`
whether the harness asks for a sync or not. Removing that was measured and is
worth ~2 ms/step; see [wan_2_2.md](wan_2_2.md). And no GPU has been measured on
a natural basis, so that pairing is not like-for-like either way.

Everything else about the two backends is a different story: v5e uses ~9.6 and
~7.5 GB **per chip** against 58 and 71 GB on the GPUs, and pays 0 s of AOT
compile against trn2/trn3's 20-131 minutes.

## Reproduce

```bash
# Toolchain (this host has no system torch; see scripts/setup_neuron_venv_offhost.sh
# for the Trainium-side equivalent):
python -m virtualenv tpuenv
tpuenv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
tpuenv/bin/pip install numpy 'torch_xla[tpu]==2.9.0'
tpuenv/bin/pip install 'diffusers==0.38.0' transformers accelerate safetensors \
    torchvision==0.24.0 huggingface_hub

# Weights (pinned revision), under a large volume — the snapshot is ~54 GiB:
HF_HOME=/mnt/models/hf tpuenv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen-Image',
                  revision='75e0b4be04f60ec59a75f475837eced720f823b6',
                  allow_patterns=['transformer/*','vae/*','scheduler/*',
                                  'tokenizer/*','text_encoder/*','model_index.json'])"

# Benchmark (writes benchmark/v5e/qwen_image.{json,md}):
DIFFLET_BENCH_DEVICE=v5e DIFFLET_BACKEND=tpu HF_HOME=/mnt/models/hf \
  tpuenv/bin/python -m benchmark.bench --model qwen_image --backend tpu \
      --skip-download --iters 3

# The synced-per-step variant discussed under Caveats:
DIFFLET_BENCH_SYNC_STEPS=1 DIFFLET_BENCH_DEVICE=v5e DIFFLET_BACKEND=tpu \
  tpuenv/bin/python -m benchmark.bench --model qwen_image --backend tpu \
      --skip-download --iters 3
```

The chips must be free before a run: a previous worker that outlived its parent
still holds `/dev/vfio/*` and the next run fails with "Device or resource busy".
`tpu-info` lists the holding PIDs.

## Caveats

**`compile_seconds = 0` does not mean compilation is free.** The TPU backend
runs eagerly, so no AOT artifact is built or reused, which is what the harness
metric records. XLA still compiles on each process's first execution — that
cost sits inside `e2e_cold_seconds` (87.3 s against a 19.1 s warm run), and it
is paid again on every process start. torch_xla cannot persist compiled
executables (`UNIMPLEMENTED: Deserializing serialized executable not
supported`), so no configuration avoids it today.

**Per-step is measured without forcing a device sync each step.** The harness
asks for device-synced inter-step deltas, which is the right rule for eager
backends. Under XLA's lazy execution that extra sync breaks pipelining and
slows the loop it is measuring — it becomes a different workload, not just a
different clock. Measured both ways:

| | e2e_warm | step |
|---|---|---|
| forced per-step sync | 24.44 s | 0.858 s |
| no forced sync (reported) | 19.10 s | 0.276 s |

**Not yet optimized.** Two known items, both measured rather than assumed:
the text encoder runs on the host, and attention has no fused kernel — the TPU
Pallas flash-attention kernel needs jax 0.7.1, which requires Python >= 3.11
while this toolchain is on 3.10.

**Two models, small n.** Qwen-Image and Wan 2.2 are the only models ported to
the TPU backend so far. Qwen-Image: `n=3` warm iterations, one run. Wan 2.2:
`n=2` warm iterations, one run, single expert.

**Neither is numerically validated.** Both are checked only for "finite" and
"the output looks right". Per DEVELOPER.md the oracle must be the diffusers
reference, never TPU-vs-Trainium.
