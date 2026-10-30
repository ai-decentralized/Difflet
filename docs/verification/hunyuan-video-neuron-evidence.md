# HunyuanVideo 1.0 on the neuron backend: on-device evidence (DiT parity per layout, end to end, CLI)

Campaign: 2026-10-24, device runs 21:31-22:35 UTC. Host: `trn2.3xlarge`, 1 Neuron device, 4 logical
NeuronCores (logical config 2), 24 GiB HBM per logical core, 12 vCPUs, 124.8 GiB RAM. Branch
`feat/hunyuan-neuron` at `88c8e48`; no code changed during stage 2.

This doc covers Task 8 of `docs/superpowers/plans/2026-10-24-hunyuan-video-neuron.md`. Stage 2
checks DiT parity on the device at every layout, at truncated depth and at full depth, and runs the
compiled mask-bounds regression (sections Summary to Bug ledger). Stage 3, the end-to-end parity
check, ran right after it (2026-10-24 22:41-23:02 UTC, code unchanged: `b38f3bc` only added this
doc). The CLI runs (stage 4) ran on 2026-10-25 at the commits that
`benchmark/trn2/hunyuan_video_neuron.md`'s header lists: the TP4 row is `tp4_devvae`, run at
`b38f3bc` with an explicit `--device-vae` (the placement `d6ffb9a` then made the default); TP2,
TP2xCP2 and the first 720p attempt ran at `4c38a58` (no code change since `d6ffb9a`); the other
three 720p attempts ran at `f4d9ac1`, `15b503f` and `ce9d212`, and the trace-cache runs at
`f4d9ac1`, `ce9d212` and `195fc8c`, each of which changes code (the fixes in the bug ledger). See
"End to end" and "CLI runs" below; the timings are reported in that benchmark doc. A compiler-level
deep dive into the one open parity result ran on 2026-10-25 14:28-15:25 UTC at `eef92d6`, with no code change
(section "Compiler-level cause of the full-depth t=900 result"; its files are under `J/s3/`).
The VAE work of the plan's F1/F2 items (2026-10-25/26: the device's bf16 GroupNorm fix `2e04372`,
the compiled decoder `1c10dfc`, the decode in the generate launch `ecd1b2d`/`74fc979`) and the CLI
re-measure at the new default (2026-10-25 23:42 - 2026-10-26 00:25 UTC at `eadddec`, clean tree)
are in "VAE decode after F1/F2" below; their evidence is under `B26 = /home/ubuntu/bench-2026-10-26/`.

Logs, result JSONs, oracles and diagnosis dumps are kept outside the repo, as the job rules
require, under `J = /home/ubuntu/.claude/jobs/28e8164e/tmp/hv1_device/`; the CLI runs' evidence is
under `B = /home/ubuntu/bench-2026-10-25/`. Every number below comes from a file in `J/` or `B/`.
Statements marked *inference* are explanations that no run measured directly.

Statuses: **PASS** (ran on the device in this campaign and met its gate) · **FAIL** (ran and
missed the gate) · **NOT MEASURED** (not run in this stage).

## Summary

* **Truncated depth (2 dual + 4 single blocks), all 8 cases PASS.** The cases are TP2 eager, TP2,
  TP4, TP2+SP, and TP2xCP2 with gather_kv, ulysses and ring (all compiled), plus TP2 at 3 valid text
  tokens. rel-L1 against fp32 diffusers is 0.948x-0.981x the bf16 control when compiled and 1.060x
  in eager mode; the gate is 1.15x. Every rank's output is identical, there are no CPU fallbacks and
  no unwarmed shapes, and all 6 of 6 blocks are compiled.
* **Full depth (20 + 40 blocks), oracle t=900 with 20 valid tokens: every compiled layout FAILs the
  1.15x gate, at 1.188x-1.191x.** That holds for TP2, TP4, TP2+SP and TP2xCP2 with gather_kv,
  ulysses and ring. Cosine is 0.999995. All the other checks pass: identical ranks, no fallbacks,
  60 of 60 blocks compiled. **TP2 eager passes at 1.006x.**
* **The failure is specific to that one input. It is not a layout, sharding or checkpoint bug, and
  this campaign fixed nothing.** We drew eight full-depth samples of the compiled TP2 DiT: t =
  950, 900, 850, 700, 500, 300 and 100 at 20 valid tokens, plus t=900 at 3 valid tokens. Their
  ratios are 0.838, **1.190**, 1.054, 0.940, 0.891, 0.943, 0.998 and 1.049x, with a mean of 0.988x.
  Only t=900/20 fails. Eager mode on the same 7 timesteps at 20 valid tokens gives 0.989-1.125x.
  The excess is confined to compiling the dual-stream block as a whole graph, and to the 12
  massive-activation channels of the residual stream. The diagnosis is below.
* **Cause found (deep dive): neuronx-cc's default mixed-precision accumulation.** When the dual
  block's two gated residual updates are in one graph, the compiler does not round the
  intermediate stream `h1 = h + attn*gate_msa` to bf16 before adding `ff*gate_mlp`: one rounding
  per block instead of diffusers' two. A graph break anywhere between the two updates gives
  1.000-1.002x, a break before the first update does not (1.186-1.188x); a host CPU emulation of
  the single rounding reproduces the device's pattern and its t=900 excess (1.198x at t=900,
  device 1.190x; close at 5 of 7 timesteps, not at t=850 or t=300); rounding `h1` explicitly
  inside the compiled graph gives 0.966-1.085x on all 8 samples.
  This rounding is **not less accurate**: over the 7 timesteps the compiled DiT's mean rel-L1 is
  0.973x the control's, against 1.027x for eager. The t=900 sample fails because the paths that
  round like diffusers (control, eager) share a lucky low error there (2.515e-03, the lowest of
  7), not because the compiled error is high (2.994e-03, inside its own 2.58-3.78e-03 range).
  **No product change is recommended; the recommendation is a multi-sample gate** (see the
  deep-dive section). Flux's compiled dual blocks, with the same compiler behaviour, score
  0.897-0.962x on 3 samples.
* **The other gate fails nowhere.** The cosine floor, finite outputs, identical ranks, zero
  fallbacks and every block compiled all pass in every case, and the cosine is at least 0.999970
  in every case.
* **HBM.** At full depth, peak HBM at 5x320x512 is 15.61 GiB per rank at TP2 and 11.10 GiB at
  TP4. At the default 61x320x512 (10,240 image tokens), timing only, it is 16.18 GiB at TP2 and
  11.67 GiB at TP4. **TP2 fits at the default shape, so amendment A9 (column-parallel modulation)
  is not needed.**
* **Time per full-depth forward (compiled, median of 2x5).** At 5x320x512: TP2 207.9 ms, TP4
  265.0 ms, TP2+SP 207.3 ms, TP2xCP2 gather_kv 649.7 ms, ring 672.7 ms, ulysses 1148.3 ms, and TP2
  eager 1802.1 ms (median of 2, `--iters 2`). At 61x320x512: **TP2 1460.8 ms, TP4 805.6 ms.**
* **Compiled mask bounds: 72/72 counts exact on a NeuronCore.** The checks cover the key bounds,
  ulysses `key_valid_len` and ring `text_valid_len`, at 1,536, 10,496 and 20,096 keys.
* **End to end (stage 3), 3 steps at full depth: PASS** at TP2 (1.003x the control), with our
  host encoder (encoder outputs equal to diffusers'), and at TP2xCP2 gather_kv (1.021x).
* **CLI (stage 4), 320x512x61, 50 steps: TP2, TP4 and TP2xCP2 gather_kv all PASS** (playable
  61-frame mp4 showing the prompt, bit-identical outputs within each cell). Step 1461.8 / 808.3 /
  905.1 ms; warm e2e 300.9 / 239.1 / 250.0 s. TP4 is the fastest layout.
* **Two bugs fixed during the CLI runs:** the decode default (`d6ffb9a`) and the missing NKI
  trace-cache opt-in that made 720p trace the attention kernel for 30+ minutes per process
  (`f4d9ac1`). A third (`15b503f`, `ce9d212`) splits the bounded attention kernel over heads
  past a size whose NKI compile overflows. See the bug ledger.
* **720x1280x129 at TP4: BLOCKED.** After those fixes the block graph's neuronx-cc compile needs
  more than the host's 124.8 GiB and is OOM-killed; no forward ran (see "CLI runs").
* **VAE (F2): the device's bf16 GroupNorm was the cause of the 20.8 dB bf16 decode; fixed by
  `2e04372`.** Bisection on one tile found every op but GroupNorm at the host's bf16 error; the
  device's GroupNorm error grows with the elements per group (0.20-0.38 rel-L1 at 1.2M-2.4M,
  1.35-2.05 at 4.5M-8.9M; host bf16 2.2e-3-3.8e-3). With its statistics in fp32 the device's
  eager bf16 decode is at 55.9 dB against the fp32 host decode, as bf16 on the host is; fp32 is
  bit-identical to before. See "VAE decode after F1/F2".
* **VAE (F1): the compiled bf16 decoder in the generate launch is the CLI default (`74fc979`)**
  and passes the default-change rule: 56.07 dB against the fp32 host decode (gate 50 dB), run at
  TP2, TP4 and TP2xCP2. At TP4 (full method, `eadddec`): warm e2e 126.7 s (was 239.1 s; AoT
  132.9 s, *reused*), cold e2e 457.8 s (was 570.4 s), compile cold 274.5 s (was 2877.2 s), decode
  16.0 s, same latents as before; rank 0's core 14.74 / 17.95 GiB. The earlier default still runs
  by flag and gives the same mp4 as before.

| item | value |
|---|---|
| instance | trn2.3xlarge, `neuron-ls`: 1 device, 4 cores (ids 0-3), 96 GB, logical-neuroncore-config 2 |
| python / venv | 3.12.3, `/home/ubuntu/venvs/native-312` (its `bin/` on PATH: `neuronx-cc` is spawned by name) |
| torch / torch-neuronx | 2.14.0+cu130 / 2.14.3.0.1+b1b6c66 |
| neuronx-cc | 2.28.7631.0+8f7b0cf5 |
| runtime / collectives / dkms | aws-neuronx-runtime-lib 2.35.30.0, aws-neuronx-collectives 2.35.43.0, aws-neuronx-dkms 2.31.7.0 |
| diffusers | 0.38.0 |
| checkpoint | `hunyuanvideo-community/HunyuanVideo` snapshot `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` (transformer in bf16) |
| launch env (added by the scripts) | `HF_HUB_OFFLINE=1 DIFFLET_DISABLE_PREWARM=1 DIFFLET_NEURON_PG_TIMEOUT_S=10800 PYTHONPATH=<worktree>` |

The first attempt (`J/dit_l2s4_tp2_eager.nopath.log`) failed with `posix_spawn failed: No such
file or directory` while compiling the first eager kernel. The interpreter had been called by
absolute path without the venv's `bin/` on PATH, so `neuronx-cc` could not be found. This was a
spurious environment failure, not a Difflet bug. Every later run goes through `J/run.sh`, which
exports the PATH.

## Commands

Every command runs from the worktree, as `J/run.sh <name> <args>`. That is
`PYTHONPATH=. PATH=/home/ubuntu/venvs/native-312/bin:$PATH python <script> <args>`, with the
output in `J/<name>.log` ending in `WALL` and `EXIT` lines. Outside torchrun, the DiT check
starts torchrun itself and prints the exact `launch:` line into the log.

```
# oracles (host CPU, fp32 reference + bf16 control; 5x320x512, 1,280 image tokens, seed 0)
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_l2s4_v20.pt --layers 2 --single-layers 4
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_l2s4_v3.pt  --layers 2 --single-layers 4 --valid-text 3
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_full_v20.pt --threads 6
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_full_v3.pt  --valid-text 3 --threads 6
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_full_v20_3t.pt --timesteps 900 500 100 --threads 6
check_neuron_hv1_dit.py --make-oracle J/hv1_dit_full_v20_4t.pt --timesteps 950 850 700 300 --threads 12
# device, one per layout (O = an oracle above)
check_neuron_hv1_dit.py --oracle O --tp-degree 2 --exec-mode eager   --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle O --tp-degree 2 --exec-mode compile --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle O --tp-degree 4 --exec-mode compile --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle O --tp-degree 2 --sp                --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle O --tp-degree 2 --cp-degree 2 --cp-mode {gather_kv,ulysses,ring} --out-json J/<name>.json
check_neuron_hv1_dit.py --oracle none --tp-degree {2,4} --frames 61 --iters 5 --out-json J/dit_f61_tp{2,4}_timing.json
check_neuron_hv1_mask_bounds.py --core 0 --work-dir J/mask-bounds
```

Oracle costs, from `J/oracle_*.log`:

| oracle | depth | t | control rel-L1 | load / fp32 / bf16 (s) | max RSS |
|---|---|---|---|---|---|
| `hv1_dit_l2s4_v20.pt` | 2+4 | 900 | 6.821e-03 | 10.8 / 2.7 / 1.8 | 7.4 GiB |
| `hv1_dit_l2s4_v3.pt` | 2+4 | 900 | 6.787e-03 | 0.4 / 2.7 / 1.9 | 7.4 GiB |
| `hv1_dit_full_v20.pt` | 20+40 | 900 | 2.515e-03 | 173.4 / 42.1 / 22.5 | 51.7 GiB |
| `hv1_dit_full_v3.pt` | 20+40 | 900 | 2.452e-03 | 5.1 / 43.5 / 22.1 | 51.7 GiB |
| `hv1_dit_full_v20_3t.pt` | 20+40 | 900/500/100 | 2.515 / 3.105 / 3.789e-03 | 5.0 / 130.0 / 54.7 | 51.7 GiB |
| `hv1_dit_full_v20_4t.pt` | 20+40 | 950/850/700/300 | 3.083 / 2.852 / 3.307 / 3.013e-03 | 5.2 / 118.2 / 42.8 | 51.7 GiB |

The low load times on the later full-depth oracles are a page-cache effect (*inference*: the first
full load read the 25.6 GB checkpoint from disk).

## Results

Columns: ours and the control are rel-L1 against fp32 diffusers. "ratio" is ours / control, and
the gate is at most 1.15x. `err_cos` is the cosine between our error and the control's error.
"load+warmup (warm-up fwd)" is seconds to build, load the checkpoint, compile and run the warm-up
forward. "compiles" is torch-neuronx's `TotalCompilations` counter. ms/fwd is the median device
forward over 2 rounds of 5 (`--iters 5`), except full-depth TP2 eager, timed with `--iters 2`
(median of 2 samples, one round). Peak HBM is the maximum over ranks. Logs are in `J/`.
Every row also passed: checkpoint (0 missing, 0 unexpected), fallbacks (none in warm-up, forwards
or benchmark), unwarmed shapes (none), identical ranks (output sha256 equal on every rank),
compiled blocks (all, compile mode) or no compiled blocks (eager), and finite output.

**Compile time was measured on a warm cache.** The persistent NEFF cache already held most of
these graphs: the first compiled run reports `PersistentHits` 111 with 1 compilation. The
load+warmup column is therefore the warm figure. The only runs that compiled new block graphs are
the 61-frame timing runs: 16 compilations and 174.0 s warm-up at TP2, 6 compilations and 93.9 s at
TP4. A cold compile was not measured.

### Truncated depth: 2 dual + 4 single blocks, 5x320x512, t=900

| layout | sample | gate | ours | control | ratio | cosine | err_cos | load+warmup (warm-up fwd) s | compiles | ms/fwd | peak HBM GiB | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP2 eager | vt=20 | **PASS** | 7.231e-03 | 6.821e-03 | 1.060x | 0.999970 | 0.724 | 103.8 (74.3) | 193 | 205.5 | 2.21 | `dit_l2s4_tp2_eager.log` |
| TP2 compile | vt=20 | **PASS** | 6.498e-03 | 6.821e-03 | 0.953x | 0.999976 | 0.618 | 39.7 (38.1) | 1 | 40.6 | 2.02 | `dit_l2s4_tp2_compile.log` |
| TP4 compile | vt=20 | **PASS** | 6.521e-03 | 6.821e-03 | 0.956x | 0.999976 | 0.595 | 30.8 (19.6) | 7 | 42.6 | 1.48 | `dit_l2s4_tp4_compile.log` |
| TP2 + SP compile | vt=20 | **PASS** | 6.694e-03 | 6.821e-03 | 0.981x | 0.999975 | 0.637 | 20.3 (19.1) | 1 | 39.3 | 2.04 | `dit_l2s4_tp2_sp.log` |
| TP2xCP2 gather_kv | vt=20 | **PASS** | 6.464e-03 | 6.821e-03 | 0.948x | 0.999977 | 0.616 | 23.0 (21.6) | 3 | 80.4 | 1.98 | `dit_l2s4_tp2cp2_gather.log` |
| TP2xCP2 ulysses | vt=20 | **PASS** | 6.467e-03 | 6.821e-03 | 0.948x | 0.999977 | 0.622 | 24.7 (23.2) | 1 | 134.2 | 2.00 | `dit_l2s4_tp2cp2_ulysses.log` |
| TP2xCP2 ring | vt=20 | **PASS** | 6.486e-03 | 6.821e-03 | 0.951x | 0.999976 | 0.590 | 30.1 (28.6) | 0 | 85.8 | 2.00 | `dit_l2s4_tp2cp2_ring.log` |
| TP2 compile | vt=3 | **PASS** | 6.496e-03 | 6.787e-03 | 0.957x | 0.999977 | 0.624 | 6.2 (4.9) | 0 | 39.1 | 2.02 | `dit_l2s4_tp2_v3.log` |

`skipped=1080` in each checkpoint line is the truncated blocks' keys, which are left out on
purpose (amendment A2).

### Full depth: 20 dual + 40 single blocks, 5x320x512, t=900 (plan Step 2 cases)

| layout | sample | gate | ours | control | ratio | cosine | err_cos | load+warmup (warm-up fwd) s | compiles | ms/fwd | peak HBM GiB | log |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TP2 eager | vt=20 | **PASS** | 2.530e-03 | 2.515e-03 | 1.006x | 0.999996 | 0.680 | 8.6 (3.8) | 0 | 1802.1 | 15.79 | `dit_full_tp2_eager.log` |
| TP2 compile | vt=20 | **FAIL** | 2.994e-03 | 2.515e-03 | 1.190x | 0.999995 | 0.495 | 10.3 (5.0) | 0 | 207.9 | 15.61 | `dit_full_tp2_compile.log` |
| TP4 compile | vt=20 | **FAIL** | 2.997e-03 | 2.515e-03 | 1.191x | 0.999995 | 0.494 | 13.0 (7.4) | 0 | 265.0 | 11.10 | `dit_full_tp4_compile.log` |
| TP2 + SP compile | vt=20 | **FAIL** | 2.992e-03 | 2.515e-03 | 1.190x | 0.999995 | 0.494 | 12.9 (6.7) | 0 | 207.3 | 15.60 | `dit_full_tp2_sp.log` |
| TP2xCP2 gather_kv | vt=20 | **FAIL** | 2.995e-03 | 2.515e-03 | 1.191x | 0.999995 | 0.497 | 16.6 (9.1) | 0 | 649.7 | 15.56 | `dit_full_tp2cp2_gather.log` |
| TP2xCP2 ulysses | vt=20 | **FAIL** | 2.989e-03 | 2.515e-03 | 1.188x | 0.999995 | 0.494 | 14.4 (7.7) | 0 | 1148.3 | 15.57 | `dit_full_tp2cp2_ulysses.log` |
| TP2xCP2 ring | vt=20 | **FAIL** | 2.992e-03 | 2.515e-03 | 1.190x | 0.999995 | 0.493 | 11.5 (4.8) | 0 | 672.7 | 15.57 | `dit_full_tp2cp2_ring.log` |
| TP2 compile | vt=3 | **PASS** | 2.572e-03 | 2.452e-03 | 1.049x | 0.999996 | 0.423 | 14.2 (8.4) | 0 | 213.2 | 15.61 | `dit_full_tp2_v3.log` |

The failure reason is quoted verbatim from `dit_full_tp2_compile.log`: `rel_l1 2.994e-03 is
1.190x the control's 2.515e-03, above max_ratio_vs_control 1.15`. The note with it reads
`err_cos 0.495 < 0.5: the result's error is decorrelated from the bf16 control's, so it does not
round like upstream bf16`.

### Full depth: more timesteps, TP2 (is t=900/vt=20 representative?)

| timestep | compile ratio | eager ratio | control rel-L1 | logs |
|---|---|---|---|---|
| 950 | 0.838x PASS | 0.989x PASS | 3.083e-03 | `dit_full4t_tp2_{compile,eager}.log` |
| **900** | **1.190x FAIL** | 1.006x PASS | 2.515e-03 | `dit_full3t_tp2_{compile,eager}.log` |
| 850 | 1.054x PASS | 1.125x PASS | 2.852e-03 | `dit_full4t_tp2_{compile,eager}.log` |
| 700 | 0.940x PASS | 0.999x PASS | 3.307e-03 | `dit_full4t_tp2_{compile,eager}.log` |
| 500 | 0.891x PASS | 1.025x PASS | 3.105e-03 | `dit_full3t_tp2_{compile,eager}.log` |
| 300 | 0.943x PASS | 0.991x PASS | 3.013e-03 | `dit_full4t_tp2_{compile,eager}.log` |
| 100 | 0.998x PASS | 1.054x PASS | 3.789e-03 | `dit_full3t_tp2_{compile,eager}.log` |
| 900, 3 valid tokens | 1.049x PASS | NOT MEASURED | 2.452e-03 | `dit_full_tp2_v3.log` |

Compiled: mean 0.988x over the 8 samples, range 0.838-1.190x. Eager: mean 1.027x over 7
samples, range 0.989-1.125x. The t=900 control has the smallest error of the seven 20-token
samples.

### Default shape 61x320x512 (10,240 image tokens), full depth, timing only (`--oracle none`)

| layout | gate (no oracle: finite, ranks, fallbacks, blocks) | load+warmup (warm-up fwd) s | compiles | ms/fwd | peak HBM GiB | log |
|---|---|---|---|---|---|---|
| TP2 compile | **PASS** | 180.5 (174.0) | 16 | 1460.8 | 16.18 | `dit_f61_tp2_timing.log` |
| TP4 compile | **PASS** | 100.1 (93.9) | 6 | 805.6 | 11.67 | `dit_f61_tp4_timing.log` |

### Compiled mask bounds (one NeuronCore)

**PASS, 72/72 counts exact on neuron.** The counts are the key bounds, the joint count (ulysses
`key_valid_len`) and the text count (ring `text_valid_len`). They were checked at joint lengths
1,536, 10,496 and 20,096, including 19,858 valid keys at 20,096, the Trainium failure case.
Wall time 36 s. Log `J/mask_bounds.log`, results `J/mask-bounds/results.json`.

### End to end (stage 3): latents after 3 steps against diffusers' fp32 pipeline

`tests/manual/check_neuron_hv1_e2e.py`, full depth, 5x320x512, 3 steps, seed 42, the script's
prompt ("A cat walks on the grass, realistic style.", 11 valid of 256 tokens). The reference is
diffusers' `HunyuanVideoPipeline` in fp32 on the host and the control the same pipeline in bf16,
both from the same bf16 prompt embeddings (`J/e2e_ref_f5.log`: fp32 76 s, bf16 28 s, control
rel-L1 7.575e-02, cos 0.996937, written to `J/hv1_e2e_ref_f5.pt`). The gate is the same 1.15x of
the control's rel-L1, plus `err_cos >= 0.5`. One reference sample was drawn.

| layout | gate | ours rel-L1 | ratio | cosine | err_cos | other checks | step times (s) | peak HBM GiB | log |
|---|---|---|---|---|---|---|---|---|---|
| TP2 compile | **PASS** | 7.601e-02 | 1.003x | 0.996888 | 0.994 | finite, no fallbacks, no unwarmed shapes, 3 steps | 1.55, 0.22, 0.21 | 15.61 | `J/e2e_tp2.log` |
| TP2 compile, our rank-0 host encoder (`--encode`) | **PASS** | 7.601e-02 | 1.003x | 0.996888 | 0.994 | encoder hidden states and pooled projection rel-L1 0 against diffusers' `encode_prompt`, mask equal (11 valid) | 2.19, 0.22, 0.22 | 15.61 | `J/e2e_tp2_encode.log` |
| TP2xCP2 gather_kv compile | **PASS** | 7.737e-02 | 1.021x | 0.996789 | 0.995 | as TP2, on all 4 ranks | 2.09, 0.62, 0.62 | 15.56 | `J/e2e_tp2cp2_gather.log` |

The full-depth DiT failure on the single t=900 / 20-token input (above) does not show at the
integration level: after 3 denoising steps the compiled TP2 latents are 1.003x the control.

### CLI runs (stage 4): `difflet generate --backend neuron`, 320x512x61, 50 steps

Seed 42, guidance 6.0, prompt "a cinematic shot of a red fox running through a snowy forest",
`HF_HUB_OFFLINE=1`, pinned revision, default placement (host encode on rank 0, decode in the
one-core fp32 vae launch). Each cell: a cold compile on a fresh root, a generate with the page
cache dropped, a discarded warm generate, then n timed warm generates, through the harness
`B/scripts/hvbench.py` (method and numbers in `benchmark/trn2/hunyuan_video_neuron.md`). Gate:
exit 0, a playable mp4 of 61 frames at 320x512 with moving content (every frame's pixel std at
least 49 grey levels, mean frame-to-frame difference 32 levels), and a mid frame that shows the
prompt (opened: a red fox running through snow in every cell), `unwarmed_shapes=[]`, no
fallbacks, 60 of 60 blocks compiled, and the same latents in every generate of a cell.

| layout | gate | DiT step median (ms) | e2e warm (s) | e2e cold (s) | peak HBM alloc / neuron-monitor (GiB) | latents (all runs of the cell) | evidence |
|---|---|---|---|---|---|---|---|
| TP2 | **PASS** | 1461.8 (n=2) | 300.9 | 629.4 | 16.18 / 18.71 | `3d0b647e…`, 4 of 4 equal | `B/hunyuan_video/tp2/` |
| TP4 | **PASS** | 808.3 (n=3) | 239.1 | 570.4 | 11.67 / 12.98 | `b62d0891…`, 5 of 5 equal | `B/hunyuan_video/tp4_devvae/` |
| TP2xCP2 gather_kv | **PASS** | 905.1 (n=2) | 250.0 | 567.8 | 15.89 / 18.00 | `b50ce0c8…`, 4 of 4 equal | `B/hunyuan_video/tp2cp2_gather/` |
| TP4, 720x1280x129 | **BLOCKED** | — (no forward ran) | — | — | 11.83 / 12.29 (weights only) | — | `B/hunyuan_video/tp4_720p.*/` |

**720x1280x129 (119,056 joint tokens) does not compile on this host.** Four `difflet compile`
attempts at TP4 (`benchmark/trn2/hunyuan_video_neuron.md`, "720x1280x129"): (1) the attention
kernel's NKI trace ran for 30+ minutes in every process, fixed by `f4d9ac1`; (2) the bounded
kernel's NKI compile failed, `BIR emission failed: colz: body blob exceeds uint32 offset range`,
at 6 heads x 119,168^2 per call; (3) the same at 3 heads (`15b503f`); (4) at one head per call
(`ce9d212`) the kernel compiles, and neuronx-cc compiling the block graph was OOM-killed at
109.8 GiB resident (dmesg `anon-rss:115166160kB`) of the host's 124.8 GiB (`[F137]`, exit code
70). The DiT weights load (11.83 GiB per
core); the activation HBM at this shape was never reached.

The plan's Step 4 names TP2xCP2 ulysses; the CP cell ran gather_kv instead because it was the
fastest CP mode in the DiT checks above (649.7 ms per full-depth forward against ring 672.7 and
ulysses 1148.3). Ulysses and ring have the same DiT parity results as gather_kv (PASS at truncated
depth, 1.188x and 1.190x on the full-depth t=900 input), but their CLI runs are **NOT MEASURED**.

### VAE decode after F1/F2 (2026-10-25/26)

Evidence under `B26 = /home/ubuntu/bench-2026-10-26/`; timings and the full tables are in
`benchmark/trn2/hunyuan_video_neuron.md` ("VAE: compiled decoder and in-process decode (F1/F2)",
"F2", "F1a", "F1b"). Reference for every accuracy row: the fp32 host decode of the same latents
(diffusers' tiled decode, `J/vae_cpu_fp32.pt` for `J/cli/tp2_host.work/latents.pt`).

**F2 finding.** Bisection on the first tile of the tiled decode (`latents[:, :, 0:5, 0:32, 0:32]`
through `post_quant_conv` + decoder, forward hooks on all 174 module calls, device bf16 and host
bf16 against host fp32; `B26/f2/bisect/`, `B26/scripts/f2_bisect.py`): the tile's output is 0.329
rel-L1 on the device against 3.3e-3 for host bf16, with the jumps at GroupNorms
(`up_blocks.2.resnets.0.norm1` 0.385 vs 1.8e-2, `up_blocks.3.resnets.0.norm1` 1.55 vs 1.0e-2).
Each module alone on the fp32 input cast to bf16: Conv3d, SiLU, the upsampling and `to_out` match
the host (1.00x), `to_q/k/v` are 0.82-0.93x, and GroupNorm grows with the elements per group
(2.8-6.6x the host at 81,920, 542-685x at 4.5M-8.9M). The GroupNorm micro-test on the device
(`B26/f2/gn_micro.log`) shows the raw bf16 kernel at 3.11-4.83 rel-L1 at 8.9M elements per group;
through `F.group_norm` with the workaround it equals host bf16 to within 1e-7 at every size.
*Inference:* the device kernel accumulates the group statistics at bf16 precision.
**Fix `2e04372`:** `install_group_norm_fp32` (`difflet/backends/neuron/core/workarounds.py`,
installed by `prepare_runtime`) computes bf16 `F.group_norm` on neuron tensors in fp32; other dtypes
and devices are untouched. CPU tests in `tests/unit/backends/test_neuron_bool_bitwise_guard.py` and
`test_neuron_backend.py` (the three new tests failed before the fix).

**Decode accuracy** (latents `J/cli/tp2_host.work/latents.pt` unless the row says otherwise):

| decoder | gate | result | status | evidence |
|---|---|---|---|---|
| device, eager fp32 (the CLI's decoder before `74fc979`) | rel-L1 <= 1e-6 | rel-L1 3.5e-7, 136.7 dB; after `2e04372` bit-identical (`torch.equal`), no new kernel | **PASS** | `B/hunyuan_video/vae/neuron_fp32*/`, `B26/f2/f2_neuron_fp32_gnfp32/` |
| device, eager bf16, before `2e04372` | >= 50 dB | rel-L1 0.226, 20.8 dB, visible banding | **FAIL** (fixed) | `B/hunyuan_video/vae/neuron_bf16_{cold,warm}/` |
| device, eager bf16, after `2e04372` | >= 50 dB | rel-L1 3.80e-3, 55.9 dB, bit-identical across processes | **PASS** | `B26/f2/f2_neuron_bf16_gnfp32{,_warm}/` |
| host, bf16 (control) | | rel-L1 3.8e-3, 55.9 dB | | `B/hunyuan_video/vae/cpu_bf16/` |
| device, compiled bf16, fixed tiles (`1c10dfc`; the CLI default from `74fc979`) | >= 50 dB | rel-L1 3.73e-3, 56.07 dB, bit-identical across repeats and processes | **PASS** | `B26/f1a/fixed_bf16_v2_{cold,warm}/`, `fixed_bf16_commit_check/` |
| device, compiled fp32, fixed tiles | >= 60 dB | rel-L1 2.5e-7, 139.3 dB | **PASS** | `B26/f1a/fixed_fp32_v2_{cold,warm}/` |
| host, fp32, the AoT path's tiling (edge tiles padded, decoder not confined to the window) | >= 60 dB | rel-L1 1.34e-2, 42.5 dB against diffusers' tiled decode | **FAIL** (why `1c10dfc` has the valid-extent decoder) | `B26/f1a/aot_semantics_full.log` |
| CLI default after F1/F2, TP4 latents (`s4_tp4/gen1`, `b62d0891…`) vs their fp32 host decode (`B26/s4/hostfp32/s4_tp4.0.pt`) | >= 50 dB | rel-L1 3.73e-3, 56.08 dB | **PASS** | `B26/s4/frames_vs_fp32.log` |
| CLI default after F1/F2, TP2xCP2 gather_kv latents (`b50ce0c8…`) vs their fp32 host decode | >= 50 dB | rel-L1 3.72e-3, 56.09 dB | **PASS** | `B26/s4/frames_vs_fp32.log` |
| CLI default after F1/F2, TP2 latents (`3d0b647e…`, equal to `J/cli/tp2_host.work/latents.pt`) | >= 50 dB | rel-L1 3.73e-3, 56.07 dB | **PASS** | `B26/s4/frames_vs_fp32.log` |

The CLI rows compare the standalone compiled bf16 decode of each layout's latents (`B26/f1b/standalone_vs_inproc/dec/`), whose mp4 is byte-identical to the in-process run's (`cmp` with this stage's gen1 mp4s); the in-process decode is not saved as a tensor.

**CLI runs at the default after F1/F2** (`eadddec`, no flag: decode on rank 0's core in the generate
launch with the compiled bf16 decoder; harness `B26/s4/scripts/hvbench_s4.py`, runs listed by
`B26/s4/scripts/queue_s4.sh`). Same gate as the stage-4 table above. TP4 ran the full method
(cold compile on a fresh root, cold e2e, gen0, n=3); TP2 and TP2xCP2 gather_kv ran gen0 and n=2 on
copies of the F1b roots. Mid frames opened in every cell: a red fox running through snow.

| layout | gate | DiT step median (ms) | e2e warm (s) | e2e cold (s) | compile cold (s) | decode (s) | rank 0's core alloc / neuron-monitor (GiB) | latents (all runs of the cell) | evidence |
|---|---|---|---|---|---|---|---|---|---|
| TP4 | **PASS** | 808.2 (n=3) | 126.7 (was 239.1) | 457.8 (was 570.4) | 274.5 (was 2877.2) | 16.0 | 14.74 / 17.95 | `b62d0891…`, 5 of 5 equal (as before F1/F2); mp4 `edf8d6be…` | `B26/hunyuan_video/s4_tp4/` |
| TP2 | **PASS** | 1461.6 (n=2) | 160.0 (was 300.9) | NOT MEASURED | NOT MEASURED | 16.0 | 19.24 / 23.50 | `3d0b647e…`, 3 of 3 equal (as before) | `B26/hunyuan_video/s4_tp2/` |
| TP2xCP2 gather_kv | **PASS** | 905.0 (n=2) | 126.0 (was 250.0) | NOT MEASURED | NOT MEASURED | 16.0 | 19.24 / 22.85 | `b50ce0c8…`, 3 of 3 equal (as before) | `B26/hunyuan_video/s4_tp2cp2/` |
| TP4, `--device-vae --vae-dtype fp32 --vae-exec-mode eager` (the earlier default) | **PASS** | 808.9 (n=1) | 213.1 (gen0 228.8) | | | 74.9 | vae core 11.53 / 20.89 | `b62d0891…`, 2 of 2; mp4 byte-identical to `tp4_devvae/gen1` (`68adae23…`) | `B26/hunyuan_video/s4_tp4_oldpath/` |

Every run: exit 0, `compiled_blocks=60 unwarmed_shapes=[] fallbacks: load=[] generate=[]` on every
rank (50 of 50 rank lines), 61 frames at 320x512, per-frame std at least 49.4 grey levels. The
first generate after the compile (`s4_tp4/cold`) ran three neuronx-cc subprocesses on rank 3 (6.4 s
before the denoise), as the 2026-10-25 cold runs did (open issue 6 of the benchmark doc); no warm
run compiled anything. The plan's step-4 targets: warm e2e at TP4 about 150 s or less, **met**
(126.7 s, n=3, max 130.6 s); decode 25 s or less, **met** (16.0 s). TP2's warm e2e (160.0 s) is
above the 150 s figure.

## Diagnosis of the full-depth t=900 failure

We followed the systematic-debugging protocol and gathered evidence before trying any fix. The
scripts and their outputs are in `J/`.

1. **The port and the checkpoint match at full depth.** TP2 eager on the device gives 1.006x. Our
   port in bf16 on the host CPU at tp1 (`cpu_port.py`, log `cpu_port_full_bf16.log`) also gives
   1.006x, with err_cos 0.763. Loading uses every key (0 missing, 0 unexpected, 0 skipped).
2. **The layout does not matter.** TP2, TP4, SP and the three CP modes all land at
   2.989-2.997e-03, so the cause is shared by every compiled layout. The block graph is the one
   thing they all share.
3. **Compiling the dual-stream blocks whole reproduces the failure; nothing smaller does.** These
   are eager-mode runs with selected pieces compiled in place (`debug_dit.py` `sub:`, logs
   `dbg_sub_*.log`, `dbg_blk_*.log`):

   | compiled pieces | ratio |
   |---|---|
   | whole dual blocks only (singles eager) | **1.186x** |
   | whole single blocks only (duals eager) | 1.000x |
   | every `ff`, `ff_context`, `proj_mlp`, `act_mlp`, `proj_out_mlp` | 1.009x |
   | every `attn` | 1.007x |
   | every `norm1`, `norm1_context`, `norm` | 1.001x |
   | every `norm2`, `norm2_context` | 1.006x |

4. **The error sits in the massive-activation channels.** We compared per-block outputs on the
   device with diffusers fp32 and bf16 on the host (`cpu_blocks.py`, `analyze_blocks.py`,
   `analyze_blocks2.py`; output `analyze_blocks_full_v20.txt`, `analyze_blocks2.txt`). In the
   image stream, 12 channels exceed 20x the median magnitude (386, 453, 915, 1028, 1111, 1181,
   1213, 1461, 1541, 1907, 1997, 2573), and channel 1028 reaches -4.8e7. In those channels the
   compiled path's accumulated error is 1.52-2.93x the control's from single block 4 onward. In
   the other 3,060 channels it stays at 0.82-0.99x over all 60 blocks. Eager stays at 0.97-1.16x
   in the massive channels and 1.01-1.08x in the rest.
   *Inference:* these channels dominate the LayerNorm statistics of every later norm and of
   `norm_out`, so a relative error there shifts the whole output.
5. **Locally, each compiled block is as accurate as bf16 diffusers or more so.** We fed each block
   the device's own input and compared its output with fp32 and bf16 diffusers on that input
   (`local_blocks.py`, `local_dual.py`, `local_dualx.py`; outputs `local_blocks*.txt`,
   `local_dual.txt`, `local_dualx.txt`):
   * Single blocks are at 0.98-1.01x the control.
   * Dual-block outputs are at 0.50-0.94x in the massive channels.
   * The dual block's intermediates (norm1 output, attention, first residual, norm2, modulation,
     ff) are within 0.55-1.31x when compiled, against 0.84-1.28x in eager mode.
   * In dual blocks 10-19 the compiled block's signed local error in channel 1028 is negative in
     7 of 8 sampled blocks. That makes it magnitude-increasing, because the channel is negative.
     The control's errors in the same blocks have mixed signs.
   * *Inference:* the compiled graph rounds the residual stream once per gated update where
     diffusers rounds twice. Its errors in the massive channels are smaller but stay correlated
     from block to block, so they add up, and the final metric for one input is luck-dependent.
     For this input it came out at 1.19x; err_cos is 0.495.
6. **Numerics changes tried. None reached 1.15x at t=900/vt=20, and none was committed:**

   | experiment (compiled TP2, full depth) | ratio | log |
   |---|---|---|
   | baseline | 1.190x | `dit_full_tp2_compile.log` |
   | GELU in fp32 (A9's accuracy contingency) | 1.190x | `dbg_full_tp2_gelu.log` |
   | single-block tail in fp32 (sum, all-reduce, bias, gate, residual) | 1.168x | `dbg_full_tp2_tail_a.log` |
   | single-block bias/gate/residual in fp32 | 1.178x | `dbg_full_tp2_tail_b.log` |
   | single-block partial sum + all-reduce in fp32 | 1.181x | `dbg_full_tp2_tail_c.log` |
   | every TP all-reduce in fp32 | 1.196x (and 408 ms/fwd) | `dbg_full_tp2_red32.log` |
   | dual-block residual updates explicit fp32, rounded once | 1.188x | `dbg_glue_dualfused.log` |
   | `NEURON_CC_FLAGS=--internal-disable-fma-on-ios` (the eager kernels' flag) | 1.190x | `dbg_full_tp2_nofma.log` |
   | LayerNorm as explicit two-pass fp32 | 1.190x | `dbg_glue2_ln2pass.log` |
   | (host CPU) port bf16 with fused residual rounding | 0.978x | `cpu_port_full_bf16_fused.log` |

   Two device probes ruled out further suspects:
   * `round_probe.py` (`round_probe.txt`): bf16 rounding at exact ties is round-to-nearest-even on
     the host CPU, in eager mode and in compiled graphs.
   * `ln_graph_probe.py`: the fp32 LayerNorm workaround reaches the Dynamo graph (`float` ->
     `layer_norm` -> `to(bfloat16)`).
7. **Conclusion.** This is not a Difflet bug, so there is no fix commit. Under the plan's gate as
   written, every compiled layout FAILs on this one oracle, at 1.188-1.191x. Across 8 full-depth
   samples the compiled DiT averages 0.988x the bf16 control, with 7 of 8 under the gate. The
   suspected cause is a decorrelated (err_cos ~0.5) but locally no-worse rounding pattern of the
   whole-graph compiled dual block. It concentrates in HunyuanVideo's massive-activation channels,
   where one input's final metric scatters by about ±15%; eager scatters too, reaching 1.125x at
   t=850. **Follow-ups:**
   * **Gate over several samples.** Gate the full-depth DiT on the mean or median over several
     timesteps and inputs instead of a single sample (the HV1.5 and Wan checks gate per timestep).
     Cost: one longer oracle, about 2 min of host fp32 per extra timestep, plus seconds on the
     device.
   * **Find the compiler cause.** If an exact pin of the compiled rounding is wanted, the next
     step is an HLO-level diff of the dual-block graph to find which convert the compiler fuses
     differently. Cost: tooling time; no product change is expected.
   * **The e2e check (stage 3) is the integration-level judge,** with 3 steps and the gate on the
     latents.

   The deep dive below confirmed the suspected cause (one rounding of the residual stream per
   dual block) at the compiler level and located it exactly. Two notes on the experiments above,
   from what it found: the `dualfused` patch could not test the cause on the device, because the
   compiler removes an explicit `bf16 -> fp32` round trip the same way; and
   `--internal-disable-fma-on-ios` was a no-op, because torch-neuronx already passes it to every
   compile (it is in the tensorizer options of every neuronx-cc log, e.g.
   `J/s3/art_none/neuronx-cc/*/log-neuron-cc.txt`).

## Compiler-level cause of the full-depth t=900 result (deep dive)

Ran on 2026-10-25 14:28-15:25 UTC (about 1 h of device time) at `eef92d6`. Nothing in the repo
changed. Every run is TP2, full depth, 5x320x512, unless the table says otherwise, and uses the
stage-2 oracles. Scripts, logs and JSONs are in `J/s3/`. `debug3.py` wraps
`check_neuron_hv1_dit.py` and `runq.py` queues its cases; `cpu_acc.py` and `acc_patch.py` run
the host emulation; `flux_check.py` and `flux_runs.py` run the Flux comparison; `summary.py`
builds the per-timestep table (`summary.txt`).

### 1. What the compiler is given

With `TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1` and `TORCH_NEURONX_DEBUG_DIR=J/s3/art_none`,
the dual block's FX graph, StableHLO and neuronx-cc work dir were kept (`J/s3/art_none/`). The
StableHLO is op for op the eager sequence, with every bf16 rounding explicit:

```
%263 = stablehlo.multiply %254, %262 : tensor<1x1280x3072xbf16>   # attn_out * gate_msa
%264 = stablehlo.add %arg3, %263 : tensor<1x1280x3072xbf16>       # h1 = h + ...   (bf16)
%269 = stablehlo.convert %264 : (...xbf16) -> ...xf32             # norm2 input
...
%355 = stablehlo.multiply %354, %332 : tensor<1x1280x3072xbf16>   # gate_mlp * ff
%356 = stablehlo.add %264, %355 : tensor<1x1280x3072xbf16>        # h_out = h1 + ...
```

The compiler command is `neuronx-cc compile module.mlir --framework XLA --target trn2 --lnc 2 -O2
--model-type transformer --auto-cast=none` (`compiler.txt`). Its log shows the tensorizer options,
which include `--accumulate-on-alu-dtype`. That is what neuronx-cc's default
`--enable-mixed-precision-accumulation` turns on. Its help text says: "Always use full precision of
the accelerator ALU when we do accumulation, never downcast the input of an accumulation.
(Default: true)". The counterpart flag `--disable-mixed-precision-accumulation` drops it
(`J/s3/cc_noacc/`, a host-only recompile of the same module).

### 2. Bisecting the dual block with graph breaks

The dual blocks are compiled with `fullgraph=False`, with `torch._dynamo.graph_break()` at chosen
points, and the single blocks stay eager. A break forces its tensors out of the graph as bf16.
The points are p1 after norm1, p2 after attn, p3 after the first gated residual, p4 after norm2,
p5 after the modulation, and p6 after the ff (`J/s3/brk_*.log`, `q1.log`).

| breaks | ratio at t=900, 20 valid tokens |
|---|---|
| none (the whole block in one graph) | **1.186x** (`brk_none.log`, as stage 2's 1.186x) |
| p1 | 1.188x |
| p2 | 1.186x |
| p3 | **1.000x** |
| p4 | **1.002x** |
| p5 | **1.001x** |
| p6 | **1.000x** |
| p1-p6 | 1.003x |

**Reading.** The excess goes away when `h1` crosses a graph boundary before the second residual
add. That holds for any break from p3 to p6, so `h1` is stored as bf16 before `h1 + gate_mlp*ff`.
It stays when only the attention side is split (p1, p2). With p4, p5 or p6, `norm2` still reads
`h1` inside the same graph as the first add, so norm2 reading an unrounded `h1` is not the cause.
**The cause is the second residual add taking an `h1` that was never rounded to bf16.** The block
then rounds its residual stream once, `bf16(h + attn*g1 + ff*g2)`, where diffusers and eager
round twice.

### 3. Host emulation reproduces the device's pattern and its t=900 excess

`cpu_acc.py` runs our port in bf16 on the host CPU (tp1, eager), with the dual block's residual
chain rounded as `acc_patch.py` selects (`J/s3/cpu_acc.log`). Mode A is
`bf16(f32(h) + f32(bf16(attn*g1)) + f32(bf16(ff*g2)))`, which only drops the `h1` rounding.
Mode B also keeps the products in fp32. Mode C also gives norm2 the fp32 `h1`.

| timestep | 950 | 900 | 850 | 700 | 500 | 300 | 100 |
|---|---|---|---|---|---|---|---|
| host, stock rounding | 1.005x | 1.005x | 1.004x | 0.998x | 1.007x | 0.998x | 0.982x |
| host, mode A (no `h1` rounding) | 0.822x | **1.198x** | 0.969x | 0.939x | 0.889x | 0.979x | 0.983x |
| host, mode B | 0.818x | 1.197x | 0.965x | 0.933x | 0.902x | 0.981x | 0.980x |
| host, mode C | 0.821x | 1.200x | 0.967x | 0.931x | 0.899x | 0.982x | 0.978x |
| device, compiled (stage 2) | 0.838x | **1.190x** | 1.054x | 0.940x | 0.891x | 0.943x | 0.998x |

Dropping the one `h1` rounding is enough to move the host from about 1.00x to the device's
pattern, including the 1.19-1.20x at t=900. Mode A is within 0.016 of the device at t=950, 900,
700, 500 and 100, but not at t=850 (0.969x host, 1.054x device) or t=300 (0.979x against
0.943x). Products and norm2 precision barely matter (modes B
and C). This is the off-device pin of the mechanism. A CPU unit test was not added: no code
changed, and the emulation needs the 25.6 GB checkpoint.

### 4. Fix candidates measured on the device (all 8 full-depth samples)

Each row covers the 7 timesteps at 20 valid tokens. The 8th sample (t=900, 3 valid tokens) is
listed separately. Cells give rel-L1 x1e3 and the ratio to the control
(`summary.txt`; JSONs `J/dit_full{3t,4t}_tp2_*.json`, `J/s3/{nomacc,rbh1}_full_*.json`).

| variant | t950 | t900 | t850 | t700 | t500 | t300 | t100 | mean ratio | max ratio | mean rel-L1 / mean control | t900 vt=3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 control rel-L1 (e-3) | 3.083 | 2.515 | 2.852 | 3.307 | 3.105 | 3.013 | 3.789 | | | | 2.452 |
| compiled (default) | 2.584 (0.838x) | 2.994 (**1.190x**) | 3.007 (1.054x) | 3.108 (0.940x) | 2.765 (0.891x) | 2.841 (0.943x) | 3.781 (0.998x) | 0.979 | 1.190 | **0.973** | 1.049x |
| eager | 3.049 (0.989x) | 2.530 (1.006x) | 3.210 (1.125x) | 3.304 (0.999x) | 3.183 (1.025x) | 2.984 (0.991x) | 3.995 (1.054x) | 1.027 | 1.125 | 1.027 | NOT MEASURED |
| compiled, `NEURON_CC_FLAGS=--disable-mixed-precision-accumulation` | 3.224 (1.046x) | 2.697 (1.072x) | 2.798 (0.981x) | 2.908 (0.879x) | 3.057 (0.984x) | 2.884 (0.957x) | 3.254 (0.859x) | 0.968 | 1.072 | 0.961 | 1.078x |
| compiled, `h1` rounded explicitly in the graph | 3.137 (1.018x) | 2.506 (0.996x) | 3.096 (1.085x) | 3.271 (0.989x) | 3.152 (1.015x) | 2.909 (0.966x) | 3.973 (1.049x) | 1.017 | 1.085 | 1.018 | 0.991x |

"`h1` rounded explicitly" (`HV1_RB=h1` in `debug3.py`) rounds `h1` to bf16 with integer ops: bit
manipulation for round-to-nearest-even, which the compiler cannot fuse away. The rest is the stock
compiled path. err_cos against the control's error then comes back to 0.66-0.83, the eager range
(0.68-0.83), from 0.40-0.75 for default compile. So this one rounding is the whole decorrelation.
With the flag off, the ratios pass but err_cos stays at -0.10-0.56. *Inference:* the flag changes
more than this rounding, and other intermediates in other graphs are still kept at higher
precision.

Speed and HBM, median forward over 2x5 (`J/s3/*.log`; the default-flag 61-frame rows are stage 2's,
at `88c8e48`):

| variant | TP2 5x320x512 | TP4 5x320x512 | TP2 61x320x512 | TP4 61x320x512 |
|---|---|---|---|---|
| default | 206.8 ms (`base_full_v20`) | 229.4 ms (`base_tp4_full_v20`) | 1460.8 ms, 16.18 GiB | 805.6 ms, 11.67 GiB |
| `--disable-mixed-precision-accumulation` | 205.4 ms | **368.7 ms** (twice: `nomacc_tp4_full_v20{,_r2}`; parity 1.081x) | 1453.8 ms, 16.18 GiB | 803.9 ms, 11.67 GiB |
| `h1` rounded explicitly | 235.1 ms (+13.7%) | NOT MEASURED | NOT MEASURED | NOT MEASURED |

The default TP4 5x320x512 forward is 229.4 ms here (`eef92d6`, spread 1.0%) but 265.0 ms in
stage 2 (`dit_full_tp4_compile`, `88c8e48`, spread 8.7%); `ops_impl/attention.py` changed between
the two commits. Against the same-day 229.4 ms the flag costs +61% at TP4; against stage 2's
265.0 ms it would be +39%. TP2 agrees across the two (206.8 against 207.9 ms).

The flag also recompiles every graph once (its value is part of the NEFF cache key) and is
process-global (`NEURON_CC_FLAGS`). torch-neuronx has no per-graph compiler option.

### 5. Flux, the same compiler behaviour in another dual-stream model

FLUX.1-dev (19 dual + 38 single blocks), 512x512, TP4 compiled, through `check_neuron_flux_dit.py`
with the oracle's sigma varied (`flux_check.py`; oracles `J/s3/flux_oracle_512_s*.pt`, logs
`J/s3/flux_tp4_{default,nomacc}.log`):

| sigma | 0.95 | 0.75 | 0.30 | forward |
|---|---|---|---|---|
| bf16 control rel-L1 | 8.771e-03 | 7.712e-03 | 1.038e-02 | |
| compiled (default) | 0.897x | 0.920x | 0.962x | 96.7 ms |
| compiled, `--disable-mixed-precision-accumulation` | 0.950x | 1.017x | 0.989x | 97.4 ms |

Flux's dual block has the same chain (`h + gate*attn`, norm2, `h + gate*ff`). The default
compile, which keeps the stream unrounded, is more accurate than bf16 diffusers on all 3 samples,
and more accurate than with the flag off.

### 6. Conclusion and recommendation

* **Cause.** neuronx-cc's default mixed-precision accumulation keeps the dual block's
  mid-block residual stream `h1` unrounded when the block is one graph, so the stream is rounded
  once per block instead of twice. It is documented, default-on compiler behaviour. It is not a
  Difflet bug and not a miscompile: per sample it is as accurate as bf16 diffusers or more (mean
  0.973x over 7 timesteps; Flux 0.897-0.962x). It is **decorrelated** from diffusers' rounding
  (err_cos about 0.5), so a single sample's ratio to the bf16 control scatters by about ±15%.
  At t=900 with 20 tokens, diffusers' own rounding happens to land unusually close to fp32
  (control 2.515e-03, the lowest of 7), and only the paths that round the same way (eager 1.006x,
  host port 1.005x) share that luck.
* **No product change.** The two fixes that pass every sample both cost something, and neither
  improves accuracy against fp32:
  * The flag is +61% at TP4 5x320x512 against the same-day default (229.4 ms at `eef92d6`; +39%
    against stage 2's 265.0 ms at `88c8e48`) and global to the process. It is neutral at the default
    61-frame shape and at TP2.
  * Explicit rounding is +13.7% at TP2 and only buys back the correlation with diffusers.

  Both stay available: the flag through `NEURON_CC_FLAGS`, the rounding as a 3-line `rne` in
  `debug3.py`.
* **Recommended gate policy for full-depth compiled DiT parity:** pass when the **mean ratio over
  at least 5 timesteps** (as `--timesteps` already draws) is at most 1.15x, and **every sample is
  at most 1.25x** (a guard against a real regression in one regime). Keep the single-sample 1.15x
  gate for truncated depth (all 8 cases pass at 0.948-1.060x) and for the end-to-end latents
  (1.003x). Under that policy the compiled DiT passes at full depth: mean 0.979x, max 1.190x over
  7 timesteps; with the 8th sample, mean 0.988x. Under the plan's gate as written, the full-depth
  TP2 and TP4 rows above stay **FAIL**. Choosing between them is the plan owner's decision; this
  deep dive changed no status.

## Bug ledger

Stage 2 (DiT parity) and stage 3 (end to end) found no Difflet bugs and made no fix commits.
The compiler-level deep dive found no Difflet bug either: the full-depth t=900 excess is
neuronx-cc's default mixed-precision accumulation (see its section).
The one environment failure, the missing PATH entry for `neuronx-cc`, is recorded under
Environment. The CLI runs found three bugs, each fixed in its own commit, and one suspected bug that the device refuted.
The F1/F2 stage (2026-10-25/26) found one Difflet-side precision bug on the device (row 5) and four
neuronx-cc limits that the compiled decoder works around (row 6):

| # | layout | symptom | root cause | fix |
|---|---|---|---|---|
| 1 | TP4, 320x512x61 | the default host decode took 410 s of a 540 s warm generate | the decode default copied LTX-2's host decode before any HunyuanVideo measurement | `d6ffb9a`: the one-core vae launch is the default (warm e2e 239.1 s); `test_device_vae_is_the_default` |
| 2 | TP4, 720x1280x129 | `difflet compile` had all 4 ranks at 100% CPU, no neuronx-cc running, for 30 minutes after the small graphs compiled; killed at 1806 s (`B/hunyuan_video/tp4_720p.killed_nki_trace/compile/run.log`) | py-spy (`.../compile_pyspy_rank1_blocking.txt`): nki tracing nkilib's `attention_cte` under torch-neuronx's `get_kernel_config` in fake-tensor mode, from `flash_attention_bounded`. That trace grows with the joint length (119,056 tokens) and is repeated in every process unless torch-neuronx's NKI trace cache keeps it; Difflet turns that cache off unless `DIFFLET_NKI_TRACE_CACHE` opts in, which HunyuanVideo 1.5's CLI does and 1.0's did not | `f4d9ac1`: the neuron launches opt in; `test_launches_opt_into_the_nki_trace_cache`. At 320x512x61 it also cuts the warm-up forward of every launch from 36.5 s (`tp4_teacache2/gen1`, `d6ffb9a`, same root as `tp4_devvae`) to 13.5 s (`B/hunyuan_video/tp4_tracecache/`), output bit-identical |
| 3 | TP4, 720x1280x129 | after fix 2, the compile failed on every rank: `flash_attention_bounded(FakeTensor(size=(6, 119168, 128)))`: `BIR emission failed ... colz: body blob exceeds uint32 offset range` (`B/hunyuan_video/tp4_720p.blob_overflow/compile/run.log`) | one bounded-kernel call takes all of a rank's heads; its NKI body grows with heads x seqlen_q x seqlen_k and overflows a 32-bit offset (8.5e10 here; nkilib tests up to 4.35e10). At 3 heads (4.26e10) it still overflowed (`15b503f`, `tp4_720p.blob_overflow_3heads/`) | `15b503f` then `ce9d212`: calls above 16 x 36864^2 = 2.17e10 are split into equal head chunks (one head at 720p); unit tests in `test_neuron_attention_c3.py`. On the device the kernel then compiles, and the next limit is host memory (neuronx-cc OOM-killed at 109.8 GiB resident, `tp4_720p.host_oom_1head/`): **open** |
| 4 (not a bug) | TP4, 320x512x61 | after `ce9d212`, a generate on a root compiled at `f4d9ac1` ran two neuronx-cc compiles (the dual and single block graphs; warm-up forward 75 s instead of 13.5 s), output bit-identical (`B/hunyuan_video/tp4_tracecache/gen2_ce9d212/`) | first suspected: the head-split loop slicing the inputs even for one call. `195fc8c` restored the unsliced single call, and the next generate (`gen3_head`, at `195fc8c`) **still** compiled both block graphs (69.4 s warm-up), with NEFFs of a third size (7,785,177 / 6,994,062 bytes against 7,801,464 / 6,996,905 at `f4d9ac1` and 7,802,415 / 6,991,577 at `ce9d212`). The block NEFFs carry `source_line` debug metadata for the ops from `ops_impl/attention.py` (`debug_info/debug_info_hlo.dbg_*`), so *inference:* any edit that moves lines in that file changes the compiled graph's key, and every cache compiled before the edit compiles its blocks once more. That is torch-neuronx's cache keying, not a Difflet bug; `195fc8c`'s stated cause is refuted, and its code (the single call as it was before the split) is kept | none needed; one recompile per cache root after `15b503f`..`195fc8c` (about 60 s at TP4, 320x512x61) |
| 5 | one core, VAE decode, 320x512x61 latents | the device's eager bf16 decode was at 20.8 dB against the fp32 host decode (rel-L1 0.226, colour banding) while bf16 on the host was at 55.9 dB (`B/hunyuan_video/vae/neuron_bf16_*`) | the device's bf16 `F.group_norm` (torch-neuronx 2.14.3.0.1 / neuronx-cc 2.28) loses precision as the elements per group grow: rel-L1 6e-3-2e-2 at 81,920, 0.20-0.38 at 1.2M-2.4M, 1.35-2.05 at 4.5M-8.9M (host bf16 2e-3-4e-3); every other decoder op matches the host (`B26/f2/bisect/`, `B26/f2/gn_micro.log`) | `2e04372`: `install_group_norm_fp32` computes bf16 GroupNorm on neuron tensors in fp32; 3 CPU tests. Device bf16 decode 55.9 dB, fp32 bit-identical (`B26/f2/`) |
| 6 (toolchain, worked around) | one core, compiled VAE decoder at the 5x32x32 latent tile | neuronx-cc 2.28: `NCC_EVRF009` (the whole decoder as one graph exceeds the 24 GB HBM limit), `NCC_ILSA902` (internal error on `torch.where` selects), `NCC_ITRF901` (internal error on the frame-doubling upsamplers), and a GroupNorm fused with a 3D conv making the conv 8x slower (0.68 vs 0.08 s per resnet) (`B26/f1a/`) | compiler limits, not Difflet bugs | `1c10dfc`: the decoder is compiled in 26 regions, masks multiplied in, the upsampler conv in its own region, resnets split into norm / conv regions; tests in `tests/unit/models/hunyuan_video/test_hunyuan_video_vae_tiling.py`, `tests/unit/backends/test_neuron_hv1_vae.py` |

## How to inspect manually

* **Pass/fail per case:** the last `PASS|FAIL <name>` lines and the `[rank N] PASS|FAIL ...` summary
  in `J/<name>.log`, which then ends in `EXIT 0` or `EXIT 1`. The exact torchrun command is the
  `launch:` line.
* **Metrics:** `J/<name>.json` holds rank 0's results, and `J/<name>.rankN.json` the others':
  `parity.<t>.vs_fp32.rel_l1`, `parity.<t>.control.rel_l1`, `ratio_vs_control`,
  `device_evidence.counters` (compilations and cache hits), `peak_device_mem_gb`,
  `benchmark.median_ms` and `rank_output_hashes`, which must be equal on every rank.
* **The full-depth failure, per block:** `J/analyze_blocks2.txt` lists accumulated error by
  channel group, with columns `c_big` and `c_rest`. `J/drift.txt` gives the signed drift per
  massive channel.
* **The deep dive:** `J/s3/q1.log` .. `q5.log` give one line per case (name, environment, ratio,
  exit code). To re-run one case, use `python J/s3/runq.py <queue> "<name>|<ENV=v;...>|<check
  args>"`, for example `"brk_p3|HV1_BRK=p3|"` for a graph break at p3 with the default args, or
  `"x|NEURON_CC_FLAGS=--disable-mixed-precision-accumulation|--oracle J/hv1_dit_full_v20.pt
  --tp-degree 2 --exec-mode compile --iters 0"`. The host emulation is
  `PYTHONPATH=. python J/s3/cpu_acc.py -,A <oracle.pt>...` from the worktree.
* **Re-run one cell:** `bash J/run.sh <name> --oracle J/hv1_dit_full_v20.pt --tp-degree 2
  --exec-mode compile --out-json J/<name>.json` from the worktree. Check first that no
  `torch.distributed.run` process is alive.
* **A CLI run:** `B/hunyuan_video/<cell>/<run>/run.log` (the exact command on its `cmd:` line,
  HEAD and dirty flag, the CLI's `load+warmup` lines, `/usr/bin/time -v`), `summary.json`
  (`rc`, `wall_s`, `mp4` frame check, `sha256_mp4`, `sha256_latents`, per-process probe spans and
  step deltas, `nmon_peak_gib`), `mid.png` (open it) and `out.mp4`.
  `B/scripts/breakdown.py B/hunyuan_video/<cell>/<run>` prints the e2e breakdown and
  `B/scripts/hvinfo.py <run dirs>` the load phases, neuronx-cc compiles and HBM.
* **The F1/F2 runs:** a CLI run under `B26/hunyuan_video/<cell>/<run>/` has the same files as above;
  `python B26/s4/scripts/breakdown_s4.py B26/hunyuan_video/s4_tp4/gen1` prints the breakdown with
  the decode on rank 0 (`B/scripts/breakdown.py` labels that decode as a vae launch), and
  `breakdown_s4.py --cell B26/hunyuan_video/s4_tp4 gen1,gen2,gen3` the cell statistics. The decode
  accuracy: `B26/s4/frames_vs_fp32.log` (`B26/s4/scripts/frames_vs_fp32.py`); a VAE decode run:
  `B26/f2/<run>/` and `B26/f1a/<run>/` (`run.log` ends in `RESULT {...}` with times and
  `vs_ref`).
* **A 720p attempt:** `grep -v 'not tested' B/hunyuan_video/tp4_720p.<attempt>/compile/run.log |
  grep -E 'BIR emission|F137|RuntimeError'`; the OOM kill is in
  `B/hunyuan_video/tp4_720p.host_oom_1head/dmesg_oom.txt`.

## Not covered

* **NOT MEASURED** in Task 8: the CLI runs of TP2xCP2 ulysses and ring, and TP4+SP DiT parity.
* **Full unit suite** (`pytest tests/unit` in the native venv at `195fc8c`, serial, 23.5 min;
  `J/pytest_unit_full_195fc8c.log`): 3477 passed, 65 skipped, 138 failed, 27 errors. 25 errors
  are collection errors of modules that import `neuronx_distributed` (the AoT stack, not
  installed in the native venv); the other 2 are setup errors of
  `test_neuron_flux_pipeline.py::test_matches_diffusers_flux_pipeline` with the same
  order-dependent `ValueError` as the 6 failures below (that file passes alone at `195fc8c`,
  4 passed). 132 of the 138 failures fail
  the same way on an archive of `4c38a58` (`J/pytest_baseline_4c38a58_on_failed.log`), so they
  predate this stage. The other 6 (`test_neuron_flux_dit.py::test_tp1_matches_diffusers`, one in
  `test_neuron_flux_pipeline.py`, four in `test_neuron_flux_teacache.py`) pass when run alone at
  `195fc8c` and fail only in full-suite order (`proj_out_attn` returns one value instead of two:
  state left by an earlier test; not traced. The failing line is the Flux single block's row-parallel
  `proj_out_attn`, which this stage did not change).
  The tests this stage added or touched pass: `tests/unit/cli -k 'hunyuan or neuron or vae'` and
  `test_neuron_cache.py` (409 passed after `f4d9ac1`), `test_neuron_attention_c3.py` (42 passed
  at `195fc8c`). The two failures in `test_neuron_compile_c9.py` are among the 132: they expect
  the HLO directory without the per-stack-version suffix that `059bff4` added.
* A cold compile time in the DiT checks, because the NEFF cache was warm (the CLI cells measure
  cold compiles, `benchmark/trn2/hunyuan_video_neuron.md`).
* Full-depth parity at 61 frames (only timing ran at that shape). TP4+SP was not run.
