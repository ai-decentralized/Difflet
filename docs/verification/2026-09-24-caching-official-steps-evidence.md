# Device evidence — step caching at each model's official step count

Branch: `feat/wan-chunked-vae-and-caching-eval` (based on `main` @ `0f9ef0f`).
Hardware: trn2.3xlarge (`i-034e1f17ce3c1c3dd`), 1 Trainium2 chip, 4 NeuronCores,
96 GiB HBM, LNC=2.
Dates: 2026-09-23 → 2026-09-24.

Toolchain, rebuilt from `requirements-neuron.lock` into `<repo>/.venv` because
this DLAMI no longer ships `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| torch / torch-xla | 2.9.1 / 2.9.0 |
| torch-neuronx | 2.9.0.2.15.32035 |
| neuronx-cc | 2.26.6360.0 |
| neuronx-distributed | 0.19.28492 |
| nxd-inference | 0.10.18399 |
| nki / nkilib | 0.5.0 |
| driver / runtime / collectives | 2.30.2.0 / 2.34.10.0 / 2.34.10.0 |

**The raw logs are not committed.** They exist on the measurement host:

| What | Host path |
|---|---|
| HunyuanVideo + Wan compile | `/tmp/logs/compile_hv_wan.log` |
| Wan VAE single-chunk validation compile | `/tmp/logs/validate_vae_chunk.log` |
| Wan 81-frame chunked-VAE compile attempts | `/tmp/logs/wan81.log` |
| HunyuanVideo caching sweep | `/tmp/logs/sweep_hv_wan.log` |
| Wan 81-frame caching sweep | `/tmp/logs/wan81_host.log` |
| Per-run outputs, logs and timings | `cclogs/caching-official-steps/` (gitignored) |
| Curated copy of the HunyuanVideo results | `/home/ubuntu/difflet-results-backup/` |

## What was measured, and why the step counts changed

The paper's caching table ran every video model at 20 steps. Each model's own
default is higher — HunyuanVideo 50 (`hyvideo/config.py`, `--infer-steps`),
Wan 2.1 T2V-14B 50 (`generate.py`), LTX-2 40 — and the step count changes the
result, because the implementation keeps 5 warm-up and 5 cool-down full steps at
either end. At cadence 2 those fixed windows cap the loop speedup at

    20 steps -> skip 5  -> 1.33x
    50 steps -> skip 20 -> 1.67x

so a 20-step measurement understates what caching does at the count users run.
Step count is a generate-time argument and not part of the NEFF shape key
(`difflet/cli/main.py`: `--steps` is in `_add_generate_flags`, shapes in
`_add_shape_flags`), so the same compiled artifacts serve both.

Modes measured: the caching-off reference plus the two probe-free controller
modes. Calibrated adaptive is **not** measured here: no calibration JSON exists
in the tree, and `scripts/calibrate_teacache.py`'s hardware collection path is
HunyuanVideo-only (`_collect_hunyuan_video_pairs`, `--collect-hv-pairs-out`), so
fitting one for Wan needs a pairs collector that does not yet exist.

## Where the loop column comes from

The table's per-step figure must include the steps the cache skipped, so it can
come neither from a DiT-forward benchmark (which times executed calls only) nor
from subtracting two end-to-end runs — `benchmark/step_latency.py`'s own header
records an LTX-2 pair that measured 293 s and 628 s for the same configuration.
`difflet/pipeline/step_timing.py` times the scheduler loop itself; it is inert
unless `DIFFLET_STEP_TIMING` is set, so no measured path changes unless a
measurement asks for it.

## HunyuanVideo — 320x512x61, 50 steps, every component on device

Components: `clip`, `llama`, `transformer`, `vae_decoder`, all Neuron. Compile
86 min. Three repetitions per mode; the table reports medians.

| mode | skipped | loop (s) | ms/step | speedup | e2e (s) | PSNR (dB) | SSIM |
|---|---|---|---|---|---|---|---|
| caching off | 0/50 | 41.15 | 823.4 | ref. | 144.3 | ref. | ref. |
| fixed cadence 2 | 20/50 | 24.77 | 495.4 | 1.66x | 128.1 | 33.0 | 0.937 |
| online-delta (α=0.6) | 20/50 | 24.78 | 495.6 | 1.66x | 128.4 | 25.3 | 0.866 |

Three things the numbers establish:

1. **The measured speedup lands on the ceiling.** 1.66x against a 1.667x bound
   (50 steps, 20 skipped). The 20-step rows could not exceed 1.33x.
2. **A DiT call costs the same either way.** `median_excl_step0` is 813 ms in all
   three modes, so caching saves the calls it skips, not time per call.
3. **online-delta trades quality, not time.** Identical skip count and loop time,
   7.7 dB below fixed cadence, with SSIM agreeing (0.937 → 0.866). The 20-step
   measurement showed the same 7.7 dB gap (31.8 vs 24.1), so this is a property
   of the mode, not of the step count.

Repeatability: the caching-off loop measured 41.168 / 40.979 / 41.305 s across
three runs — 0.3% spread, against the ±2–3 s the end-to-end column carries.

## Wan 2.1 14B — 480x832x81, 50 steps

Frame count is Wan's own default (81; `generate.py` requires 4n+1). The paper's
earlier 9-frame shape is 3 latent frames and ~4.7k attention tokens against 21
latent frames and ~33k at 81, a different regime for what caching can save. At
81 frames the measured denoise loop is 336.7 s for 50 steps (6734 ms/step),
against 575.5 ms/step at 9 frames.

**Component placement: UMT5 and the transformer are on device; the VAE decodes on
the host.** The reason, and everything tried to avoid it, is in the next section.
This leaves the caching columns intact — the cache acts inside the denoise loop
and the decode runs once per generation outside it, and PSNR compares cached
against uncached output decoded the same way — and costs only the end-to-end
column, where host decode adds roughly 400–500 s.

Results: see `cclogs/caching-official-steps/results.json` once the sweep
completes; this document is updated with the table when it does.

## Why Wan's VAE has no device build at 81 frames

`WanVAEDecoderModel.forward` loops over latent frames, carrying a causal
`feat_cache` between them, exactly as upstream diffusers does
(`AutoencoderKLWan._decode`). Tracing flattens that loop, so the graph grows with
the frame count against neuronx-cc's ceiling. Measured at 480x832, each a
separate compile of the `vae` stage:

| latent frames | output frames | instructions | outcome |
|---:|---:|---:|---|
| 3 | 9 | — | compiles (111 min, 370 MiB artifact) |
| 4 | 13 | 6,675,705 | NCC_EBVF030, limit 5,000,000 |
| 5 | 17 | 8,871,691 | NCC_EBVF030 |
| 6 | 21 | 11,164,146 | NCC_EBVF030 |
| 7 | 25 | 11,823,804 | NCC_EBVF030 |
| 9 | 33 | 15,719,922 | NCC_EBVF030 |
| 21 | 81 | 39,093,968 | NCC_EBVF030 |

The transformer is unaffected and compiles at 81 frames (38.7 GiB artifact): its
attention is tiled by the NKI kernel, so it does not unroll. The decoder's
convolution stack does, and its last stages run at full 480x832 resolution.

Compilation cost at the one size that fits: 111 min wall, and a `walrus_driver`
peak of 126.7 GB — more than the host's 124 GB of RAM, so 96 GB of swap had to
be added for it to complete at all. HLO generation is not the cost: 1.2 s of the
6670 s.

### Paths tried to avoid the ahead-of-time compile

| Path | Result |
|---|---|
| Chunked AOT (this branch) | Works and is bit-identical; ~111 min per graph |
| `torch_xla` eager | Compiles every operator separately; a toy decoder did not finish in 900 s |
| `torch.compile(backend="openxla")` | Dynamo fake-tensor propagation fails on `silu`: "Expected all tensors to be XLA tensors. Got: XLAFloatType". Unaffected by swapping `nn.SiLU` for `F.silu`; `torch._dynamo.config.fake_tensor_propagation` no longer exists to disable |
| `torch_xla.compile` | Reaches the compiler and fails inside it: NCC_INLA001 on a `concatenate`, which the compiler asks be reported as a bug |
| NKI `conv3d` kernel | Available in nki 0.5.0; not evaluated. `conv3d_temporal_unroll` (nki 0.6.0) does not apply: its own `should_use_temporal_unroll` returns False for every layer here, wanting `C_out <= 32` and multiple temporal positions where the decoder has 96–384 channels and decodes one latent frame at a time |
| Newer toolchain | `neuronx-cc` on the public index tops out at 2.27; the NKI Library warns its kernels "are not guaranteed to be compatible with the latest release of the Neuron compiler" |

Reproduce the two execution-mode findings with the probes on branch
`exp/wan-vae-execution-modes`.

For contrast, the one public Trainium2 Wan deployment we found
(`malinich1/wan22-pytorch-native-trn2`) decodes its VAE eagerly at 768x1280 and
81 frames — a larger shape than the one that fails here — using PyTorch Native
`device='neuron'`. That is a different execution framework, not a setting, and
it reports VAE decode at 201.7 s eager and 112.7 s under `torch.compile`.

This is the same shape as the paper's existing context-parallel finding: a
capability exists upstream but not in a combination the pinned stack can build.

## The chunked VAE decoder

`difflet/models/wan/vae/chunked.py` lifts the causal cache out of the traced
region and decodes in fixed-size chunks, which is an equivalence rather than an
approximation because the loop body was already per-frame.
`scripts/wan_vae_chunked_parity.py` and
`tests/unit/models/wan/test_wan_vae_chunked.py` check it is bit-identical
(`maxdiff = 0.000e+00`) at 3, 6, 9 and 21 latent frames and at chunk sizes 3 and
6, including a trailing short chunk.

Two graphs, not one, and of different sizes:

* **First chunk, 3 latent frames.** The cache has three regimes, not two: frame 0
  leaves the string sentinel `"Rep"` in the upsample3d slot, frame 1 consumes it
  and takes a different `time_conv` call, and from frame 2 on every slot is a
  fixed-shape tensor. A 3-frame first chunk swallows all three, so neither graph
  contains a branch on cache state.
* **Later chunks, 2 latent frames.** The later graph also reads the 32 written
  cache slots as inputs, which the first does not. At 3 latent frames that cost
  5,426,220 instructions — 8.5% over the ceiling — while the first chunk at the
  same size compiled. Two frames brings it under. 81 frames decomposes as
  3 + 2x9 = 21 latent frames, ten graph launches.

The cache stays in HBM. At 480x832 its 32 written slots hold 1,889 MB in bf16,
the largest four being `(1, 96, 2, 480, 832)` at 153 MB each near the end of the
decoder, so returning them to the host each chunk would move ~13 GB per decode —
more than the host decode this replaces. Each slot is an aliased `nn.Parameter`,
updated in place, as the TeaCache probe holds `prev_mod`. One slot
(`conv_out`'s) is never written and needs no Parameter: `_clear_cache` sizes the
list by counting every `WanCausalConv3d`, but the last one is called outside the
cache-writing path.

The first-chunk graph holds **no** Parameters. It reads no prior cache, and a
Parameter that never enters the computation is absent from the lowering context,
which fails the trace with "Unable to lower HLO: parameter not found in lowering
context". It returns its cache as ordinary outputs, which seed the later graph's
Parameters once per decode.

**Status: the implementation is verified on CPU and by unit tests; its device
compile has not completed.** Three attempts were made. The first two failed on
defects in this branch's code, both since fixed and pinned by tests
(`tests/unit/backends/test_wan_vae_chunked_app.py`): a wrapper that did not
inherit `ShapeBucketedInputGenerator`, so `ModelWrapper.input_generator` built
LLM-shaped inputs and called `prepare_sampling_params`, which is None for a VAE;
and the first-chunk Parameters above. The third was stopped after 92 minutes to
free the machine for the measurement sweep.

## How to inspect manually

```bash
# The measured ceiling, one frame count at a time (each is a fresh compile)
python -m difflet.cli.stage --orchestrator Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --stage vae --stage-mode compile --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --tp-degree 4 --height 480 --width 832 --num-frames 13

# Chunked decode equals single-shot decode
python scripts/wan_vae_chunked_parity.py --latent-frames 21

# The sweep, and the table it produces
DIFFLET_WAN_FRAMES=81 DIFFLET_RERUN_MODES="off cadence2 online" \
  bash scripts/rerun_caching_official_steps.sh wan
python scripts/collect_caching_results.py --root cclogs/caching-official-steps --latex

# Quality of one model's cached modes against its own caching-off output
python scripts/psnr_compare.py --sweep cclogs/caching-official-steps/hunyuan
```

A stale lock is worth knowing about: a compile killed mid-run leaves
`model.hlo_module.pb.lock` under `/var/tmp/neuron-compile-cache/`, and the next
compile waits on it indefinitely, logging only
`[INFO]: Another process must be compiling ...` once a minute. Remove the lock
when no `neuronx-cc` process is running.
