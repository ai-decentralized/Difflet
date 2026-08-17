# Cloud TPU v5e (v5litepod-4) — difflet benchmark

Measured on a Cloud TPU VM: `v5litepod-4`, 4 × v5e chips, 2x2 topology,
16 GB HBM per chip, us-west4-a. Run through `benchmark/adapters/tpu.py`, the
same harness and metric schema as the other device folders.

```bash
DIFFLET_BENCH_DEVICE=v5e DIFFLET_BACKEND=tpu \
  python -m benchmark.bench --model qwen_image --backend tpu --skip-download --iters 3
```

## Qwen-Image, 1024x1024, 20 steps, tp=4, bf16

| metric | value |
|---|---|
| `compile_seconds` | 0 (eager — see below) |
| `load_seconds` | 7.8 |
| `e2e_cold_seconds` | 87.3 |
| **`e2e_warm` mean** | **19.10 s** (n=3, std 0.055) |
| **`step_latency` mean** | **0.276 s** (n=19, std small) |
| `throughput` | 3.6 steps/s |
| `peak_device_mem_gb` | 9.62 per chip |
| `output` | `[1, 4096, 64]` packed latents, finite; 1.06 MB PNG after decode |

Toolchain: torch 2.9.0+cpu, torch-xla 2.9.0, libtpu 0.0.21, diffusers 0.38.0.

## Cross-device (same harness, same model config)

| device | e2e_warm (s) | step (s) | compile (s) | load (s) | peak GB |
|---|---|---|---|---|---|
| B300 SXM6 | 10.65 | 0.140 | 0 | 5.2 | 58.29 |
| H100 PCIe | 18.32 | 0.298 | 0 | 9.9 | 58.38 |
| **v5e x4** | **19.10** | **0.276** | **0** | **7.8** | **9.62** |
| trn3 (4 cores) | 54.56 | 0.324 | 1189.4 | 372.8 | — |
| trn2 (4 cores) | 62.71 | 0.447 | 1316.4 | 452.9 | — |

Reading it:

- **3.3x faster than trn2 and 2.9x faster than trn3** on warm end-to-end, and
  faster per step than either.
- **Level with an H100 PCIe** (19.10 vs 18.32 warm; 0.276 vs 0.298 per step),
  at roughly a sixth of the peak device memory because tp=4 shards the weights
  across chips while the GPU runs dense on one.
- **~1.8x slower than a B300.**
- Compile and load are the largest contrast with Trainium: 0 s and 7.8 s here
  against ~1200-1300 s and ~370-450 s there.

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

Set `DIFFLET_BENCH_SYNC_STEPS=1` to reproduce the synced variant.

**Not yet optimized.** Two known items, both measured rather than assumed:
the text encoder runs on the host, and attention has no fused kernel — the TPU
Pallas flash-attention kernel needs jax 0.7.1, which requires Python >= 3.11
while this toolchain is on 3.10.

**Single sample of a single model.** Only Qwen-Image has been ported to the
TPU backend so far, so there is no matrix here yet.
