# FLUX cache rebuild parity audit — 2026-07-31

## Decision

The rebuilt FLUX cache execution and evaluation loop is behaving consistently, but the
pre-rebuild quality row cannot be reproduced as a like-for-like oracle. The strongest
available explanation for the reported `35.44 dB` versus the current `25.98 dB` is a
different prompt/case set. Coordinate choice and a short cooldown do not explain the
difference.

This conclusion deliberately separates:

- facts reproduced from current tensor artifacts;
- hypotheses disproved by controlled Trainium A/B runs; and
- historical questions that cannot be proved because the old prompt/seed manifest,
  tensors, model revision, and metric configuration were never saved.

## Locked environment

All controlled runs in this audit were collected from a clean
`feature/cache-system` worktree at commit
`1182921ca62f4393dc3c92aa30077a32624f9b1e`.

- model: `black-forest-labs/FLUX.1-dev`
- resolved revision: `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`
- compile-cache key: `65af1cd08dfa90cc`
- hardware: Trainium, TP=4
- dtype: bfloat16
- shape: 1024×1024
- denoise steps: 50
- guidance: 3.5
- prompt split: `legacy_parity`
- seeds: 0 and 1
- candidate: periodic anchor + full-noise-prediction TaylorSeer,
  warmup=14, interval=4, phase=1, order=1

Each artifact directory contains the protected collection manifests, raw latent
trajectories, final latents, decoded images, and an independent offline evaluation.

## 1. Trajectory metric definition

The production gate is `minimum-per-step-flattened-v1`: flatten each denoise step,
compute its cosine, then take the minimum. Recomputing alternative diagnostics in
float64 from the exact same index-coordinate artifacts gives:

| Sample | Stacked-trajectory cosine | Mean per-step cosine | Minimum per-step cosine |
|---|---:|---:|---:|
| p000-s0 | 0.999848112362 | 0.999870338583 | 0.998741478269 |
| p000-s1 | 0.999695613668 | 0.999741691806 | 0.997481826481 |
| p001-s0 | 0.998750961085 | 0.998921641424 | 0.990478023576 |
| p001-s1 | 0.997989781532 | 0.998075006276 | 0.983091682971 |

The evaluator's float32 minimum for the last sample is `0.983079910278`.

Result: the definition change is confirmed and materially raises a stacked/average
diagnostic relative to the production worst-step metric. It does **not** reproduce the
old `0.996803` exactly: the current worst stacked value is `0.997989782`. There is no
fixed conversion factor between these metrics; both depend on the distribution of error
over steps.

## 2. Per-step error shape

For every sample, baseline and candidate trajectory tensors are bit-identical at steps
0–13. The first unequal tensor is step 14, the first predicted step. Every sample reaches
its minimum cosine at step 49.

For the worst sample, p001-s1, selected values are:

| Step | 14 | 22 | 30 | 38 | 43 | 45 | 47 | 48 | 49 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine | .999999 | .999960 | .999796 | .998106 | .994245 | .991638 | .988013 | .985692 | .983092 |

Result: error starts at the first forecast, accumulates through the trajectory, and
accelerates late. It is not a failure that appears only in the final five steps.

The exploratory plot is stored at
`/home/ubuntu/flux-cache-parity-protocol-v1-final-20260731/trajectory-cosine-by-step.png`.

## 3. Shifted FlowMatch coordinate hypothesis

Reconstructing the exact scheduler from the locked model revision gives `mu=1.15`.
Contrary to the original hypothesis, adjacent sigma distances expand near the end:

- first distance: `0.006420493126`
- middle distance: `0.014315128326`
- final distance: `0.055738490075`
- final/first ratio: `8.681340978`

Timestep is approximately `1000 × sigma` with maximum observed discrepancy
`3.01e-05`. For first-order Newton extrapolation, sigma and timestep should therefore be
equivalent apart from floating-point rounding.

The controlled hardware results are:

| Coordinate | Speedup | Worst trajectory/final cosine | Worst PSNR | Worst LPIPS |
|---|---:|---:|---:|---:|
| index | 2.0238× | 0.98307991 | 25.9777 dB | 0.08892 |
| sigma | 2.0236× | 0.98283297 | 25.8138 dB | 0.08476 |
| timestep | 2.0245× | 0.98283297 | 25.8138 dB | 0.08476 |

Result: sigma/timestep do not narrow the terminal latent error. Index is slightly better
on worst cosine and PSNR; sigma/timestep are slightly better on LPIPS. Coordinate choice
is not the source of the roughly 9.5 dB discrepancy.

## 4. Cooldown hypothesis

Holding `coord=index` fixed:

| Cooldown | Full/skip steps per sample | Speedup | Worst cosine | Worst PSNR | Worst LPIPS |
|---|---:|---:|---:|---:|---:|
| 1 | 24 / 26 | 2.0238× | 0.98307991 | 25.9777 dB | 0.08892 |
| 2 | 25 / 25 | 1.9475× | 0.98296803 | 26.0322 dB | 0.08810 |
| 3 | 25 / 25 | 1.9474× | 0.98296803 | 26.0322 dB | 0.08810 |
| 4 | 26 / 24 | 1.8921× | 0.98299038 | 26.1129 dB | 0.08658 |

Cooldown 2 and 3 have the same effective anchor mask because step 47 is already a
periodic anchor. Their four saved trajectories are byte-for-byte identical. Cooldown 4
adds another real anchor but improves worst PSNR by only `0.1353 dB` relative to
cooldown 1.

Result: a short terminal protection window is not the missing quality mechanism.

## 5. Prompt/case stratification

The current index/cooldown-1 arm has the following per-case results:

| Prompt | Seed | Final/trajectory cosine | PSNR | LPIPS |
|---|---:|---:|---:|---:|
| fox in snow | 0 | 0.99872494 | 39.7919 dB | 0.00968 |
| fox in snow | 1 | 0.99746573 | 35.5945 dB | 0.01937 |
| night market | 0 | 0.99046493 | 28.9539 dB | 0.06333 |
| night market | 1 | 0.98307991 | 25.9777 dB | 0.08892 |

The worst value over only the fox cases is `35.5945 dB`, just `0.1545 dB` above the old
summary's `35.44 dB`. The full four-case result is instead controlled by night-market
seed 1.

Result: this is strong numerical support for the recollection that the reported quality
improved after changing cases. It is not historical proof: the old raw manifest was
never saved, so the exact old case binding cannot be recovered.

## 6. Remaining historical uncertainty

Model snapshot and compiler drift cannot be measured retrospectively without the old
revision and artifacts. They remain possible secondary contributors. They are less
diagnostic than the observed case split because both sides of each current A/B run share
the same model and stack.

The current metric pipeline is internally coherent:

- final latent equals `trajectory[-1]` by an exact tensor check;
- the minimum trajectory step is the final step for all four samples;
- PSNR, SSIM, and LPIPS degrade together on the difficult cases;
- a same-configuration index rerun reproduced every quality value exactly; and
- cooldown 2/3, which materialize the same mask, produced byte-identical trajectories.

The calibrator's fail-closed rejection is therefore correct. This arm is fast, but it
does not meet the registered latent or PSNR gates on the frozen four-case split.

## Evidence locations and evaluation-manifest hashes

| Experiment | Artifact directory | SHA-256 of `quality-evaluation-v1.json` |
|---|---|---|
| index/c1 | `/home/ubuntu/flux-cache-coord-index-20260731` | `0e8f648a8d3f5a34edceb3a4ecc17d8848cdec1c3f35231bae0622350fdb4906` |
| sigma/c1 | `/home/ubuntu/flux-cache-coord-sigma-20260731` | `bca82ece5dddf0f5452f10c8ee17d263f6848d77efb43d61f71f74fdf3c2bf0c` |
| timestep/c1 | `/home/ubuntu/flux-cache-coord-timestep-20260731` | `bf23f576f6127efd69cd3f5a10e4283f74a096351461319f4a6d99cbc0cd08a5` |
| index/c2 | `/home/ubuntu/flux-cache-cooldown-c2-20260731` | `e3ca42743209587f4102ec2b1e05c50a45a8f9364643afe23b011606844fdc8e` |
| index/c3 | `/home/ubuntu/flux-cache-cooldown-c3-20260731` | `c522405c9eae82c469b66288998d861efb4d5fd656a61f8519b06d3978abfafa` |
| index/c4 | `/home/ubuntu/flux-cache-cooldown-c4-20260731` | `c10b0befd01f71ee1dce6a0afef3fb945026d6901ea24da1e9c8b7b91b1e9bb3` |

These directories are local experiment evidence, not source-controlled fixtures.
