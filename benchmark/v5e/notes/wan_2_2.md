> Port-era notes (2026-08/09, standalone runner). The measured row on the shared
> cold/warm/per-step protocol is [`../wan_2_2.md`](../wan_2_2.md) (2026-09-12,
> `benchmark.bench --backend tpu`); the cross-device tables are in
> [`../RESULTS.md`](../RESULTS.md). Kept for the TPU-specific detail
> (what is on the chip, parity, TeaCache A/B, profiling) that the generated report does not carry.

# Benchmark — Wan-AI/Wan2.2-T2V-A14B-Diffusers

**Status:** ok
**Backend:** tpu
**Device:** Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip
**Timestamp:** 2026-08-23 22:38 UTC

> Configuration: tp=4, bf16, **single expert resident** (high-noise
> `transformer`), eager. See "Expert residency" — this matches what trn2/trn3
> measured and is not the both-experts configuration the GPU rows use.

## Configuration

| key | value |
|---|---|
| model type | wan |
| HF revision (pinned) | `5be7df9619b54f4e2667b2755bc6a756675b5cd7` |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 832, 'num_frames': 9} |
| steps | 20 |
| guidance scale | 1.0 (no CFG second pass) |
| seed | 42 |
| latent grid | 3 x 60 x 104 → 4680 patch tokens |

Model, shape, steps, guidance and seed come from `benchmark/models.py::MATRIX`
unchanged, so the numbers sit beside the other device folders. `MATRIX`'s
`config_label` says "attention_cte" — that is the **Trainium** kernel and did
not run here; TPU attention is
`torch.nn.functional.scaled_dot_product_attention`.

### What is TPU-specific

| key | value |
|---|---|
| execution | eager (lazy XLA), one worker process per chip |
| attention | **Pallas flash attention** (`torch_xla.experimental.custom_kernel`), SDPA below 32 M score elements |
| matmul precision | `highest` (XLA's TPU default is reduced precision) |
| text encoder | host, **fp32** — see "Text encode" |
| VAE | host — the device decode path is blocked, see Caveats |
| python | **3.12.14** — required for the fused kernel, see "Attention" |
| torch / torch-xla / libtpu | 2.9.0+cpu / 2.9.0 / 0.0.20 |
| jax / jaxlib | 0.7.1 / 0.7.1 |
| diffusers / transformers | 0.38.0 / 5.15.1 |

## Expert residency

Wan 2.2 A14B ships **two 14.288B experts** — `transformer` (high-noise) and
`transformer_2` (low-noise) — selected at a timestep boundary
(`boundary_ratio` 0.875 in `model_index.json`). Each is 57.2 GB fp32 on disk,
28.6 GB as bf16, **7.15 GB per chip at tp=4**.

Only one is resident here. Measured: one expert occupies 6.985 GB of the
15.748 GB a v5e chip exposes. Two would be 14.3 GB, leaving 1.4 GB — less than
the DiT forward's own transient footprint. This matches what Trainium serving
already does (`difflet/serving/models/wan.py` sets `enable_transformer_2:
False`), so the high-noise expert runs every step.

The GPU rows in the cross-device table below are **not** the same
configuration: an 80 GB H100 holds both experts resident (71.1 GB peak).

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 0 (eager — see caveats) |
| weight load (sharded, per rank) | 4.19 s |
| text encode (host, umT5) | 2.43 s |
| **denoise — first run** (XLA compile inside) | **56.30 s** |
| **denoise — warm** | **12.20 s** |
| text encode (host, fp32, one rank + broadcast) | 0.78 s |
| VAE decode (host) | 26.77 s |
| **e2e warm** (encode + denoise + decode) | **41.66 s** |
| peak device memory | **8.10 GB** per chip (of 15.748 GB) |

Before the fused attention kernel was wired in, the same run measured 18.19 s
warm at 7.02 GB. The kernel is 1.49x on the denoise loop and costs ~1.1 GB of
padding buffers.

The first denoise pays XLA's compile of the DiT graph inside step 0
(21.10 s against a 0.90 s warm step). That cost is per process start, not per
request: torch_xla cannot persist compiled executables.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 608.5 ms | 608.1 ms | 611.0 ms | 605.4 ms | 38 |
| denoise (warm, 20 steps) | 12.20 s | 12.20 s | 12.21 s | 12.19 s | 2 |

**Throughput:** 1.643 steps/s

Per-step basis: **synced** — device-synced inter-step deltas of a real generate
loop, step 0 excluded, produced by the shared
`benchmark/harness.py::RealLoopStepTimer` so it is the same rule the other
device folders use. Unlike Qwen-Image, Wan has no meaningfully different
unsynced basis — its loop round-trips to the host every step by construction.
See "Two bases" below.

### Where the per-step goes

The orchestrator holds latents on the host in fp32 (UniPC's order-2 corrector
collapses in bf16), so every step round-trips to the chip. Measured two ways on
the same graph, before and after the fused kernel:

| | SDPA | fused |
|---|---|---|
| the real loop, host round-trip per step | 887.7 ms | **610 ms** |
| a scheduler-less probe, device-resident | 739.7 ms | 441.7 ms |

**The second row is not reachable by this pipeline** and must not be quoted as
a Wan result — that probe (`benchmark/wan_tpu_stepcost.py`) chains raw DiT
calls with no scheduler at all. Removing the round-trip was measured directly
and is worth ~2 ms; see "Two bases" below. The gap between the rows, ~170 ms,
is what running UniPC on the chip costs.

## Attention: what was slow, and the fix

Profiled by ablating one real `WanTransformerBlock` at the exact per-rank
shapes (`benchmark/wan_block_profile.py`). On the original SDPA path the full
block was **20.0 ms/layer** and the cost was badly skewed:

| removed | block time | cost of the piece | share |
|---|---|---|---|
| — (full block) | 20.0 ms | | |
| self-attention | 6.3 ms | **13.7 ms** | **68%** |
| cross-attention | 17.1 ms | 2.9 ms | 15% |
| FFN | 16.4 ms | 3.6 ms | 18% |
| the 2 row-parallel all-reduces | 16.9 ms | 3.1 ms | 15% |
| the cross-rank qk-norm reduce | 19.9 ms | 0.1 ms | 0.7% |

Self-attention was **43% of the FLOPs but 68% of the time**, while the FFN was
40% of the FLOPs and 18% of the time.

**The MXU was never the problem.** Isolated matmuls at these shapes:

| op | time | MFU |
|---|---|---|
| `(1,4680,5120) @ (5120,1280)` (a Q/K/V projection) | 0.38 ms | **82%** |
| `(1,4680,5120) @ (5120,3456)` (the FFN up-projection) | 0.93 ms | **90%** |
| all-reduce `(1,4680,5120)` bf16 | 0.99 ms | — |

`F.scaled_dot_product_attention` was, at 6-7% MFU. On XLA it has no fused
lowering: it materializes the full 10x4680x4680 score matrix (219 M elements)
and runs the softmax in fp32, so ~0.82 GiB is written and re-read several
times. Its cost is a shape-independent **42 ps per score element** — it scales
with the matrix, not the arithmetic.

### The fused kernel

`torch_xla.experimental.custom_kernel.flash_attention` ships in torch_xla 2.9
and does work here; it needs `jax`, which needs Python >= 3.11. Measured per
rank against an fp32 reference:

| | time | MFU | rel. err vs fp32 |
|---|---|---|---|
| Wan self-attn (10 heads, 4680), SDPA | 7.99 ms | 7.1% | 3.31e-3 |
| Wan self-attn, **fused** | **1.19 ms** | **47.9%** | **2.76e-3** |
| Qwen joint-attn (6 heads, 5120), SDPA | 5.65 ms | 7.2% | 3.37e-3 |
| Qwen joint-attn, **fused** | **0.73 ms** | **56.3%** | **2.80e-3** |

**6.7-7.8x faster and more accurate** — it keeps the softmax statistics in fp32
without ever materializing the matrix, which is exactly what SDPA pays for.

**The trap: an unaligned sequence is silently wrong.** The kernel accepts a
length that is not a multiple of its block size without complaint and pads
internally with zeros, which then take part in the softmax. At Wan's 4680 that
returns a result with **5.39e-2** relative error — 20x worse, no error raised,
and the images still look plausible. The fix is to pad to a multiple of 512 and
mark the pad rows as a separate segment via `q_segment_ids`/`kv_segment_ids`;
that restores 2.76e-3 and costs nothing measurable (1.18 ms either way). Note
4736 — a multiple of 128, but not of 512 — raises instead.

**Small shapes still go to SDPA.** The kernel is near-flat in the score-matrix
size while SDPA grows linearly, so below a threshold SDPA wins. Swept at
q=4680, head_dim 128, 10 heads:

| score elements | SDPA | fused | ratio |
|---|---|---|---|
| 24 M (Wan's cross-attention) | 0.51 ms | 0.89 ms | **0.58x** |
| 48 M | 1.90 ms | 0.89 ms | 2.14x |
| 96 M | 3.49 ms | 0.91 ms | 3.82x |
| 144 M | 5.12 ms | 0.94 ms | 5.42x |
| 219 M (Wan's self-attention) | 7.99 ms | 1.19 ms | 6.70x |

`FLASH_MIN_SCORE_ELEMENTS` is set to 32 M, in the middle of the crossover.
Masked and causal calls also stay on SDPA — see `_should_flash`.

### End-to-end effect, both models

| model | basis | SDPA | fused | speedup |
|---|---|---|---|---|
| Wan 2.2 | synced (its only basis) | 899 ms | **610 ms** | 1.47x |
| Qwen-Image | synced | 861 ms | **510 ms** | 1.69x |
| Qwen-Image | natural (no per-step sync) | 620 ms | **291 ms** | 2.13x |

Against an H100 PCIe on the same models (`throughput` is the fair basis for a
lazy backend; the GPU figures are device-synced real-loop deltas):

| model | v5e fused, synced | H100, synced | |
|---|---|---|---|
| Wan 2.2 | **610 ms** | 554 ms | v5e 1.10x slower |
| Qwen-Image | **510 ms** | 298 ms | v5e 1.71x slower |

Qwen gains more from the kernel than Wan (2.13x vs 1.47x) because attention was
a larger share of its step — 64% against Wan's 56%. On Qwen's natural basis
(no per-step sync, which its device-resident loop supports and Wan's does not)
it reaches 291 ms, level with the H100's synced 298 ms — but no GPU has been
measured on a natural basis, so that pairing is not a like-for-like result.

**Requires Python 3.12.** On 3.10, `pip install jax[tpu]` resolves to jax 0.6.2
and downgrades libtpu 0.0.21 -> 0.0.17, breaking torch_xla. jax 0.7.1 needs
>= 3.11 and pulls libtpu 0.0.20, which torch_xla 2.9 runs on fine. Without jax
the op falls back to SDPA automatically — the backend stays correct, just
slower — and `DIFFLET_TPU_FLASH=0` forces that fallback for A/B runs.

**What is left.** The two row-parallel all-reduces were 15% of the block before
and are a larger share now that attention has shrunk; sequence parallelism
would convert them to reduce-scatters. The cross-rank qk-norm reduce, which
looked like a plausible suspect at 4 extra collectives per layer, costs 0.1 ms
and is not worth touching.

Whole-forward arithmetic, for scale: one DiT forward is **133.5 TFLOP**.

## Cross-device (same model, same shape/steps; **see the expert note**)

| device | denoise warm (s) | step (ms) | peak GB | experts resident |
|---|---|---|---|---|
| H100 PCIe | 33.7 (e2e) | 554.2 | 42.4 (2.1: 14B) / 71.1 (2.2) | both |
| **v5e x4** | **12.20** | **610** | **7.5-8.1 per chip** | **one** |
| trn2 (4 cores) | — | 554.8 ᵈ | — | one |

**Comparing per-step needs a stated basis**, because under lazy XLA the same
loop yields three different numbers, and one of them is not device time at all
(see the correction in [RESULTS.md](RESULTS.md)). `throughput` — N steps, one
wait, divided — is the fair comparison against the GPU and trn2 figures, which
are device-synced inter-step deltas of a real generate loop:

### Two bases — and Wan does not actually have a second one

| per step | Qwen-Image | Wan 2.2 |
|---|---|---|
| **synced** (the cross-device rule) | **510.4 ms** | **610.2 ms** |
| natural (same loop, no per-step sync) | **290.9 ms** | **610.5 ms** |
| H100 PCIe, synced | 297.7 ms | 553.7 ms |

Qwen's loop keeps its latents on the chip, so removing the sync lets the
tracing of step N+1 overlap the execution of step N and the step nearly halves.
**Wan's does not move.** Its orchestrator holds the latents on the host in fp32
(UniPC's order-2 corrector collapses in bf16), so every step ends in a
``.cpu()`` — a hard sync whether or not the harness asks for one. There is no
unsynced mode to measure; the two bases agree to 0.3 ms because they are the
same run.

**That round-trip was measured, and removing it is not worth it.**
`benchmark/wan_device_loop_probe.py` runs the real orchestrator four ways at
8 steps:

| variant | true per step | note |
|---|---|---|
| host, as shipped | **610 ms** | one compile at step 0 |
| device latents, scheduler sigmas left on CPU | ~15 600 ms | **4 recompiles, ~196 s** |
| device latents + sigmas on device, still synced | 645 ms | no recompiles |
| device latents + sigmas on device, no per-step wait | **612 ms** | no recompiles |

Two things fall out. First, the trap: diffusers deliberately keeps the sigmas
on the host (`self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU
communication`), and with the latents on device those per-step CPU scalars fold
into the graph as constants, so XLA recompiles until UniPC's solver order
stabilises — the same failure mode the plan already records for a Python scalar
in the loop. Moving the sigmas across fixes it completely.

Second, and the reason not to bother: **612 ms against 610 ms.** The DiT is
610 ms of device work and a 1.2 MB latent round-trip disappears next to it.

The 441.7 ms that `benchmark/wan_tpu_stepcost.py` reports is *not* what this
would reach, and must not be quoted as a Wan result: that probe chains raw DiT
calls with no scheduler at all. The difference — ~170 ms/step — is what running
UniPC on the chip actually costs.

Against an H100 on the rule both were measured with, v5e is **1.71x slower on
Qwen-Image and 1.10x slower on Wan**. On Qwen's natural basis it is level
(290.9 vs 297.7), but no GPU has been measured that way, so that comparison is
not yet available in either direction.

## Text encode

umT5 runs on the host, and two things about *this* host matter:

| variant | 512 tokens | 64 tokens |
|---|---|---|
| bf16 | 16.12 s | 2.00 s |
| fp32 | 3.56 s | 0.94 s |

- **fp32 beats bf16 by 4.5x.** The VM is an AMD EPYC (`n2d`) without
  AVX512-BF16, so a bf16 matmul is emulated. The checkpoint is fp32 anyway.
- **Encoding at the prompt's real length beats padding to 512 by ~4x.** The
  benchmark prompt is 16 tokens. difflet's shared `encode_prompt` pads to
  max_length because Trainium compiles a fixed 512-token text stage; T5
  attention is masked and the padded positions are zeroed afterwards, so
  encoding 16 tokens and zero-padding the embeddings is the same tensor.

Together: **39.44 s → 2.43 s**. Then encoding on one rank and broadcasting the
embeddings to the other three (they were computing the same tensor from the
same prompt) took it to **0.78 s**, with the output bit-identical.

## Output validity

| field | value |
|---|---|
| latents | [1, 16, 3, 60, 104] |
| finite (no NaN/Inf) | True |
| range | -2.136 .. 1.995 |
| decoded | [1, 3, 9, 480, 832], finite |
| visual | coherent — a red fox running through a snowy forest, matching the prompt |

## Reproduce

```bash
# Toolchain: Python 3.12 is required for the fused attention kernel (jax 0.7.1).
uv python install 3.12 && uv venv --python 3.12 tpuenv312
uv pip install --python tpuenv312/bin/python torch==2.9.0 \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python tpuenv312/bin/python numpy "torch_xla[tpu]==2.9.0"
uv pip install --python tpuenv312/bin/python "jax[tpu]==0.7.1"
uv pip install --python tpuenv312/bin/python "diffusers==0.38.0" transformers \
    accelerate safetensors torchvision==0.24.0 imageio imageio-ffmpeg
uv pip install --python tpuenv312/bin/python -e . --no-deps
# uv's CPython ships libpython as a shared object outside the venv:
export LD_LIBRARY_PATH=<uv-python-dir>/lib:$LD_LIBRARY_PATH

# Weights (pinned revision), under a large volume — the snapshot is ~118 GiB
# because both experts are fp32:
HF_HOME=/mnt/models/hf tpuenv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.2-T2V-A14B-Diffusers',
                  revision='5be7df9619b54f4e2667b2755bc6a756675b5cd7',
                  allow_patterns=['transformer/*','transformer_2/*','vae/*',
                                  'scheduler/*','tokenizer/*','text_encoder/*',
                                  'model_index.json'])"

# Benchmark (one process per chip; writes <out>.json/.mp4/.npy):
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  tpuenv/bin/python benchmark/wan_tpu_run.py --steps 20 --iters 3 \
      --out /mnt/models/wan_tpu_opt

# The per-step breakdown under "Where the per-step goes":
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  tpuenv/bin/python benchmark/wan_tpu_stepcost.py

# The per-layer ablation under "Why it is ~1.6x slower per step":
HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu \
  tpuenv/bin/python benchmark/wan_block_profile.py
```

Both read `DIFFLET_WAN_SNAPSHOT` if the checkpoint is not at the default path.

The chips must be free before a run: a worker that outlives its parent still
holds `/dev/vfio/*` and the next run fails with "Device or resource busy".
`tpu-info` lists the holding PIDs.

## Caveats

**Single expert, so this is not the full A14B recipe.** The high-noise expert
denoises every step, including the range below the 0.875 boundary where the
model is meant to switch. It is the trn2 configuration, not the GPU one.

**`compile_seconds = 0` does not mean compilation is free** — the same caveat
as Qwen-Image. XLA compiles on each process's first execution, and torch_xla
cannot persist compiled executables. The fused kernel makes this *worse* in
absolute terms: the first denoise is 56.07 s against 38.09 s on the SDPA path,
because Pallas compiles the kernel too. Warm runs are unaffected.

**The VAE decode runs on the host** (36.67 s) and is now by far the single
largest phase — larger than the entire 12.19 s denoise. Moving it to a chip is
not blocked by memory — the DiT leaves 7.6 GB free —
but `AutoencoderKLWan` raises `Value out of range (expected to be in range of
[-1, 0], but got -2)` under torch_xla, an unsupported negative index in the
decoder. Qwen-Image's VAE has no such problem and does run on device.

**Not numerically validated at the model level.** The attention op itself is
checked against an fp32 reference at the real shapes (2.76e-3 relative, better
than the SDPA path it replaced), and the fused and SDPA runs agree to a mean
absolute 0.002 per pixel on the decoded video with the same seed. But the full
model still has only "finite" and "the video looks right"; per DEVELOPER.md the
oracle must be the diffusers reference, never TPU-vs-Trainium.

**Only self-attention is fused.** Wan's cross-attention (24 M score elements)
stays on SDPA because the kernel is slower there; that is measured, not assumed.

**n=2 warm iterations, one run, one prompt.**
