# Neuron flash attention: softmax exp on the Scalar engine (trn3)

`DIFFLET_NEURON_ATTENTION_EXP=scalar` (the new default) against `=vector` (attention_cte's own
default, the previous behaviour), one checkout, back to back.

| | |
|---|---|
| Code | `b453247d` (this branch's code change; the scripts here were added after the runs) |
| Host | trn3.48xlarge, LNC=2, Neuron SDK 2.33: torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631, nki 0.7.0; diffusers 0.38.0, transformers 4.57.6 (`env.txt` of the run) |
| Date | 2026-11-13, a shared host (other jobs running; load average 30-200) |

## Kernel (`kernel_sweep.py`, one logical core)

`attention()` itself, bf16, d=128; median of 9 synchronized calls; relative L2 error against fp64
on 256 random query rows.

| shape | B*H | Vector exp | Scalar exp | change | rel. error vs fp64 (Vector / Scalar) |
|---|---|---|---|---|---|
| 1,024 tokens | 24 | 1.02 ms | 0.76 ms | -25.5% | 2.256e-03 / 2.256e-03 |
| FLUX 1024^2 (4,608) | 12 | 2.51 ms | 1.93 ms | -22.9% | 2.320e-03 / 2.322e-03 |
| Wan 480p 9 frames (4,680) | 12 | 2.69 ms | 2.44 ms | -9.5% | 2.304e-03 / 2.302e-03 |
| Qwen-Image 1328^2 (~7.4k) | 12 | 4.82 ms | 4.69 ms | -2.6% | 2.324e-03 / 2.322e-03 |
| 16,384 tokens | 8 | 13.48 ms | 11.77 ms | -12.7% | 2.596e-03 / 2.594e-03 |
| Wan 480p 81 frames (32,760) | 5 | 32.80 ms | 26.97 ms | -17.8% | 3.085e-03 / 3.090e-03 |
| cross-attention 32,760 x 512 | 5 | 2.09 ms | 2.15 ms | +2.9% | 2.255e-03 / 2.254e-03 |
| H3 768p TP4xU2, bounded (38,272) | 7 | 69.88 ms | 57.78 ms | -17.3% | 3.782e-03 / 3.781e-03 |

- Both engines are equally accurate at every shape.
- Long sequences (16k and up) gain 13-18%, consistently: an earlier run on another branch gave
  -17.4% at 32,760 and -19% at H3's shape.
- The few-millisecond shapes are not stable between runs on this host: an earlier run gave
  Qwen-Image -11% and cross-attention -2%. Read them as "not slower by much", not as figures.

## End to end (`ab_e2e.sh`, `compare.py`)

TP4 on one device per run, seed 42, prompt in `ab_e2e.sh`. Step time: the model's logged in-run
step median; for HunyuanVideo and LTX-2, which log none, (DiT stage - load and warm-up) / steps,
which also holds the prompt encode.

| model | shape, steps | Vector exp | Scalar exp | change | PSNR Vector vs Scalar | md5 Vector / Scalar |
|---|---|---|---|---|---|---|
| FLUX.1-dev | 1024^2, 50 | 222.2 ms | 215.3 ms | -3.1% | 45.5 dB | 4e90b75b5d53 / bdf69eef98b3 |
| Qwen-Image | 1024^2, 50 | 304.0 ms | 296.6 ms | -2.4% | 46.3 dB | cf4adeab6186 / e3688bac5bd4 |
| Wan 2.2 A14B | 480x832x81, 40 | 5066.8 ms | 4618.1 ms | -8.9% | 27.5 dB | baabb380176e / 8eceacac05ef |
| HunyuanVideo 1.5 480p | 480x848x121, 30 | 10026.0 ms | 8468.0 ms | -15.5% | 18.0 dB | 0c8ab7b722b8 / 97170fac6c40 |
| HunyuanVideo | 544x960x61, 30 | 5870.0 ms | 5270.0 ms | -10.2% | 36.9 dB | 8ea2bfc346d5 / cf5e9c535ada |
| LTX-2 | 512x768x121, 30 | 2243.3 ms | 2210.0 ms | -1.5% | 21.5 dB | 8875c653a923 / 1a30bf5d3d1b |

- The gain follows the attention share of a step: short image sequences 2-3%, long videos 9-16%.
- PSNR here is between two trajectories, not a distance to a reference. Inspected frames show the
  same scene, composition and quality on both sides for every model; the low values (HunyuanVideo
  1.5, LTX-2, Wan) are the same subject in a different pose or stride. For scale, MiniMax-H3 at
  768p: Vector vs Scalar 20.4 dB, the same Vector code with only the reduction order changed
  (TP8 instead of TP4 x Ulysses2) 22.4 dB.
- Reruns: an earlier run of the same A/B (a branch without #156) produced byte-identical files
  for FLUX, Qwen-Image, HunyuanVideo 1.5 and LTX-2 on both engines; Wan and HunyuanVideo differ
  from that run because #156 changed their timestep dtype.

## Not checked

- One seed and one prompt per model; no distance to a higher-precision reference (both engines'
  kernel error against fp64 is the accuracy evidence).
- The NKI Library ring kernel (#159) does not take the flag and was not measured.
- Kernel timings of a few milliseconds are wall-clock with a host round trip per call; a
  back-to-back measurement was not run (no free device at the end of the session).
