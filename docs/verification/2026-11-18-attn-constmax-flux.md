# Const-max attention at model level, FLUX.1-dev lane (four arms, one session, trn3)

Campaign: nkilib `attention_const_max` (bf16 attention with a fixed per-layer softmax max) on
every model, branch `exp/attn-constmax-models`. This lane re-runs FLUX.1-dev's four arms back to
back. E1 (`docs/verification/2026-11-18-attn-constmax.md`) ran three of them and compared
mxfp8 + const-max with an mxfp8 run from another session. Its numbers are a cross-check here.

Every number is **measured** on this host in this run unless it is marked *inferred* or
*evidence* (quoted from an earlier doc, not re-run).

## 1. Summary

FLUX.1-dev, 1024x1024, tp4, 50 steps, seed 42, guidance 3.5. Device 1 (cores 4-7), CPUs
54-59,150-155. Tree `63b0ae2c`, clean under `difflet/`, for all four timed arms.

| arm | warm DiT step (ms) | ratio | warm e2e (s) | output md5 | vs A: PSNR / SSIM / LPIPS / latent cos | vs CPU fp32: PSNR / SSIM / LPIPS |
|---|---|---|---|---|---|---|
| A bf16 (shipped `attention_cte`) | **222.6** | 1 | 11.55 | `67da9e98…` = evidence A1 | - | 23.36 / 0.9170 / 0.0748 |
| B bf16 + const-max | **201.1** | **B/A 0.903** | 10.47 | `23f8c63d…` = E1 | 41.40 / 0.9931 / 0.0024 / 0.99957 | 23.31 / 0.9156 / 0.0761 |
| C `--quant mxfp8` | **185.7** | C/A 0.834 | 9.71 | `7b6a23dc…` = evidence B1 | 22.59 / 0.9079 / 0.0978 / 0.96935 | 21.30 / 0.8846 / 0.1392 |
| D `--quant mxfp8` + const-max | **165.1** | **D/A 0.742, D/C 0.889** | 8.65 | `5fd4a295…` = E1 | 24.18 / 0.9224 / 0.0683 / 0.97928 | 22.11 / 0.8918 / 0.1177 |

- **Gain.** Const-max saves 21.6 ms per step on bf16 (0.903x) and 20.6 ms on top of MXFP8
  (0.889x). With MXFP8 linears and const-max together, a step takes 0.742x of shipped bf16.
- **Quality holds.** B is 41.4 dB from A, 12 dB tighter than bf16's own tp2-vs-tp4 floor
  (29.4 dB). Against fp32, B is 0.05 dB below A, and D is 0.81 dB *above* C. Every arm is above
  the CPU bf16 floor (21.47 dB, evidence) except C, which is 0.17 dB below it. C is the shipped
  default MXFP8, unchanged by this experiment (evidence: 21.30 dB).
- **Reproducibility.** All four arms are deterministic, and each md5 equals its earlier run:
  A = the evidence's A1, C = the evidence's B1 (the shipped MXFP8 default), B and D = E1.
  Within each arm, both requests are bit-identical.

## 2. Method

**Hook.** E1's, unchanged for FLUX (E1 section 2). With `DIFFLET_EXP_ATTN_CONSTMAX=1`:

- `NeuronFluxTransformer2DModel` carries each attention's sigma as a non-persistent `[128, 1]`
  buffer.
- The non-CP wrapper passes the sigma to `ops_impl.attention`.
- Unbounded, non-causal, bf16, d = 128 calls go to `difflet_neuron::attention_const_max` (nkilib
  2.14 `experimental/attention/attention_const_max.py`, unmodified).
- softmax_max = 0.88 sigma sqrt(2 ln Sk) is computed in the graph at the call's own Sk (4,608).

All 57 FLUX attention calls are routed: 19 joint blocks and 38 single blocks.

**This run's tree.** The KERNEL lane's `2d6714cf` (in the timed tree) changed `attention.py`. It
adds the key-bounded fork and refactors the unbounded condition into `_const_max_shape_ok`, with
the same predicate. FLUX's calls are unbounded, so they take E1's op. B's and D's outputs are bit
for bit E1's (`23f8c63d`, `5fd4a295`), which confirms it.

**Table provenance.**

- Table: `difflet/models/flux/attn_constmax.json`, written by `scripts/attn_constmax/table.py`
  from the QK-RMSNorm gains of hub snapshot `3de623fc…`.
- 57 layers, sigma 0.608-2.092, largest gap (hard_bound - softmax_max) 42.7 at
  `single_transformer_blocks.36`. All 57 pass the gap <= 80 rule.
- Re-run in this lane on the CPU:
  `table.py flux table_regen.json /mnt/nvme/difflet-mxattn/probe/constmax/bounds.json`.
  - The output is byte-identical to the committed table.
  - The cross-check against section 12's `bounds.json` passes (38 of 38 one-pair layers).
- No value was tuned against any output.

**Arms.**

- `scripts/attn_constmax/arms.sh` (this lane added the `mx` arm and the `CACHE` root, `3a570e2f`).
- Arm names: A = `bf16`, B = `cm` (`DIFFLET_EXP_ATTN_CONSTMAX=1`), C = `mx` (`--quant mxfp8`),
  D = `mxcm` (both).
- arms.sh unsets `DIFFLET_EXP_ATTN_CONSTMAX` and every cache location first, then sets the flag
  only for B and D.
- bench.sh's `flux` key flags: `--model-id black-forest-labs/FLUX.1-dev --height 1024 --width
  1024 --tp-degree 4 --steps 50 --seed 42`, prompt "a cat sitting on a bench".
- Driver: `/mnt/nvme/difflet-exp/constmax-models/flux/run_lane.sh`, run under
  `flock .../devlocks/neuron1.lock`. It does three things:
  1. compile and warm the four arms (untimed);
  2. run the arm-B gate;
  3. wait for the gate verdict, then run the four timed arms back to back.
- The device lock is held across the gate, so no lane took device 1 between the gate and the
  timing. Before each process starts, the driver checks that `fuser /dev/neuron1` is empty and
  logs the slice's `mpstat` (`lane.log`: every timed start at 0% on all 12 CPUs).
- `NEURON_RT_VISIBLE_CORES=4-7`, `taskset -c 54-59,150-155` and `NEURON_NUMA_PIN=0`.

**Caches.**

- E1's per-arm caches were copied (`cp -a`, E1's dirs untouched) to
  `/mnt/nvme/difflet-exp/constmax-models/flux/cache/flux-{bf16,cm,mxcm}`. C got a copy of
  mxcm's quantized directory (`mxfp8-809df193b011ba1f`, the shipped default set).
- Compile walls: A 1:20, B 2:19, C 13:41 (new NEFFs: no earlier C cache), D 1:29.
- No timed process compiled anything (no neuronx-cc line in any `timed.log`).

**Timing.**

- Each timed arm is one process that runs two identical requests (`--requests`).
- warm DiT step = the median over the two requests of each request's DiT step median (n = 49,
  steps 2-50), as printed by the orchestrator.
- spread = |r1 - r2| / that value.
- warm e2e = request 2's wall, which includes the in-process decode (0.33 s after the last step).
- FLUX has no per-step dump hook (`DIFFLET_STEP_TIMES_FILE` is not wired into FLUX's loop). The
  per-request median, max and n are the finest timings kept.

**Quality.**

- Done with `scripts/attn_constmax/score.sh` (`ROOT=/mnt/nvme/difflet-exp/constmax-models`),
  which runs `scripts/compare_difflet_runs.py` on request 2's PNG and its final packed latents.
- References: this run's A, and the CPU fp32 diffusers reference
  `/home/ubuntu/fp8-bench/fp32_ref/flux` (same bf16 noise).
- D against C was scored separately with the same tool.

## 3. Gate (before any timing)

- 10-step arm-B generate: `arms.sh flux cm smoke`, at 09:47 on device 1, tree `63b0ae2c`.
- Latents are finite: `[1, 4096, 64]` bf16, mean -0.167, std 1.071, |max| 4.53.
- `smoke.png` is a clean cat on a bench, with no noise or artifacts.
- md5 `2b3b752b…` = E1's smoke, bit for bit. Step median 203.0 ms (n=9).
- **Gate: passed** (`gate.go`).

## 4. Timing (measured; device 1, back to back 09:49-09:55)

| arm | start | request 1: median / max (ms) | request 2: median / max (ms) | **warm step (ms)** | spread | request 1 / 2 wall (s) |
|---|---|---|---|---|---|---|
| A bf16 | 09:49:10 | 222.9 / 226.0 | 222.3 / 229.4 | **222.6** | 0.27% | 11.72 / 11.55 |
| B bf16 + cm | 09:50:48 | 201.0 / 206.3 | 201.1 / 204.9 | **201.05** | 0.05% | 10.63 / 10.47 |
| C mxfp8 | 09:51:58 | 185.3 / 189.2 | 186.1 / 191.9 | **185.7** | 0.43% | 17.27 / 9.71 |
| D mxfp8 + cm | 09:53:36 | 165.4 / 170.8 | 164.8 / 168.2 | **165.1** | 0.36% | 8.81 / 8.65 |

Ratios: **B/A 0.903, C/A 0.834, D/A 0.742, D/C 0.889.**

- C's first request spent 7.8 s before the end of its first step (the other arms: 0.2-0.5 s).
  That is one-off first-request work in that process. Request 2 (the warm one) does not show it.
- *Cross-check with E1 and the evidence (same host, other sessions):*
  - A 222.6 here; 220.5 in E1; 221.6 in the evidence's A1.
  - B 201.05 here; 200.0 in E1.
  - C 185.7 here; 182.9 / 183.3 in the evidence's B1.
  - D 165.1 here; 162.2 in E1.
  - Every arm here is 0.5-1.8% slower than in its earlier run. The ratios agree: E1's B/A 0.907
    and D/A 0.736. E1's cross-session D/C was 162.2 / 183.1 = 0.886 (*inferred* there), against
    0.889 measured here.
  - The host's load average was 27-33 when the lane started (`uptime`, 09:28), with other lanes'
    compiles on other CPUs. It was not sampled during the timed arms.
- *Inferred:*
  - Const-max saves 21.55 ms per step in bf16 and 20.6 ms on top of MXFP8, about 0.36-0.38 ms per
    call over 57 calls. The saving is nearly the same with or without MXFP8 linears: the
    attention kernel is not changed by `--quant mxfp8`.
  - So the gains compose: (C/A) x (B/A) = 0.753 predicts D/A, and the measured value is 0.742.

## 5. Quality (measured; request 2 of each arm)

| pair | PSNR (dB) | SSIM | LPIPS | latent cos | latent rel L2 |
|---|---|---|---|---|---|
| B vs A | **41.40** | 0.9931 | 0.0024 | 0.99957 | 0.029 |
| C vs A | 22.59 | 0.9079 | 0.0978 | 0.96935 | 0.249 |
| D vs A | 24.18 | 0.9224 | 0.0683 | 0.97928 | 0.204 |
| D vs C | 23.56 | 0.9278 | 0.0663 | 0.97777 | 0.211 |
| A vs fp32 | 23.36 | 0.9170 | 0.0748 | 0.97568 | 0.222 |
| **B vs fp32** | **23.31** | 0.9156 | 0.0761 | 0.97532 | 0.223 |
| C vs fp32 | 21.30 | 0.8846 | 0.1392 | 0.95605 | 0.299 |
| **D vs fp32** | **22.11** | 0.8918 | 0.1177 | 0.96446 | 0.269 |
| *evidence* floor: bf16 tp2 vs tp4, same image (6a) | 29.4 | 0.963 | 0.021 | - | - |
| *evidence* floor: CPU bf16 vs fp32 (`fp32_ref/flux_cpu_bf16`) | 21.47 | 0.890 | 0.135 | 0.95756 | 0.293 |

- **A's md5 equals the evidence's bf16 md5.** `67da9e98`
  (`docs/verification/2026-11-07-mxfp8-video-models.md` 6a, A0/A1). Every score with an earlier
  counterpart (E1, evidence 6a and 10) equals it to the printed digits, as the md5s predict.
- **B holds.** It is closer to A than bf16 at another TP degree is (41.4 vs 29.4 dB). Against
  fp32 it is 0.05 dB below A, inside the 0.22 dB spread between bf16's two TP degrees against
  fp32 (E1 section 5).
- **D vs C.** At 23.6 dB, D is a different trajectory from C, as C is from A (22.6 dB). FLUX at
  50 steps turns the cat's head under any small perturbation (evidence 6b: twelve images). The
  strip `arms_strip.png` shows four clean cats, with no noise in any arm.
  - C's head is turned.
  - B's and D's heads are close to A's.
  - D's distance to fp32 (22.11 dB) is better than C's (21.30) and between the CPU bf16 floor and
    A. One image cannot rank D above C (evidence 6b), but nothing here says const-max costs
    quality on top of MXFP8.
- **Broken arms: none.** Every arm's latents are finite and every image is clean.

## 6. Caveats

- **One image, one seed, one prompt.** B's 41.4 dB from A makes the bf16 verdict robust. The
  MXFP8 arms' distances are trajectory drift, and the twelve-image set (evidence 6b) was not run
  for D.
- **Per-layer fidelity is lower** than cte: up to 6x cte's error at FLUX's largest sigma, from
  E1's kernel probe (E1 3a, *evidence*). The softmax_max rule uses the gains only, and only the
  hard bound guarantees no overflow. FLUX's largest gap is 42.7, well inside the 80 rule. No
  overflow was seen.
- **Scope.** Measured: tp4, non-CP, 1024x1024. CP paths (ring / ulysses / gather_kv), other
  resolutions, other prompts and LoRAs are not routed or not tested.
- **Shared host.** Load average 27-33 at the lane's start, with other lanes compiling on other
  CPUs. This lane's 12
  CPUs were idle at every timed start, and the four arms ran back to back on one device within
  6 minutes.
- **Spread** is over two requests in one process, not over processes. E1's same-arm numbers from
  another hour sit 0.5-1.8% lower.

## 7. Paths

- Evidence: `/mnt/nvme/difflet-exp/constmax-models/flux/`:
  - `results.json` (from `collect.py`);
  - `lane.log` (per-start `mpstat` and run ends), `run_lane.sh`, `gate.go`;
  - `<arm>/{compile,smoke,timed}.log` and `<arm>/*.env.txt` (time, head, dirty paths, `fuser`,
    `mpstat`);
  - `<arm>/warm_{1,2}.png`, `<arm>/work/worker_0/latents_req000{0,1}.pt`;
  - `cm/smoke.png`, `cm/gate_latents.txt`;
  - `scores/*.json`, `arms_strip.png`, `table_regen.json`;
  - `cache/flux-<arm>` (kept).
- Code: E1's hook (`difflet/backends/neuron/ops_impl/attention.py`,
  `difflet/models/flux/modeling_flux.py`) and `scripts/attn_constmax/{arms.sh,score.sh,table.py}`.
