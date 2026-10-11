# Benchmark — Wan2.1 T2V 14B on the neuron backend (TorchNeuron, TP4, 50 steps)

**Status:** g=5.0: ok, g=1.0: ok  
**Backend:** neuron (torch-neuronx, per-block `torch.compile`)  
**Device:** trn2.3xlarge / 4 NeuronCores (logical config 2) / 96 GiB/device  
**Timestamp (last launch):** g=5.0: 2026-10-11 10:44 UTC, g=1.0: 2026-10-11 11:29 UTC  
**Commit:** `3763193977f4a508f72f713444d100f65fa88b08` in `/home/ubuntu/Difflet/.claude/worktrees/neuron-phase1`

> Configuration: tp=4, bf16, TorchNeuron per-block torch.compile, host fp32 umT5 on rank 0, host VAE; 480x832x9 (latent 1x16x3x60x104, 4,680 tokens, 512 text tokens), bf16, seed 42, prompt "a cinematic shot of a red fox running through a snowy forest".

## Headline

| guidance | compile (s) | e2e cold (s) | e2e warm (s, n=3 mean) | DiT step median (ms) | DiT forward median (ms) | peak HBM (GiB/rank) |
|---|---:|---:|---:|---:|---:|---:|
| 5.0 | 57.6 | 206.5 | 141.2 | 1067.2 | 531.5 | 8.13 |
| 1.0 | 60.9 | 186.3 | 114.1 | 534.4 | 530.9 | 8.13 |

e2e = the benchmark script's `prepare_runtime` → mp4 saved on rank 0 (NEFF cache empty for cold, populated for warm; OS page cache hot in both). The user-facing `difflet generate` walls, the page-cache-cold row and the resident (serving) rows are in the tables below. DiT step = one denoise iteration of the real loop (at g=5.0 two forwards + the host UniPC step).

Run order: guidance 5, 1; per guidance script-cold, script-warm1, script-warm2, script-warm3, cli-w4check, cli-warm1, cli-warm2, cli-warm3, cli-truecold, cli-cold. Cold compile walls: g=5.0 57.6 s, g=1.0 60.9 s (same block graph and shape; one cold compile per guidance value, each on its own fresh cache): they differ by 5.8%.

## Guidance 5.0 (headline: real video, CFG = two batch-1 forwards per step)

| metric | definition | cold launch | warm launch (n=3) | resident (2nd/3rd generate) | trn2 AoT baseline (20 steps, g=1.0) |
|---|---|---:|---:|---:|---:|
| compile wall (s) | rank-max `warmup forward` cold − warm (first warm launch) | **57.6** (= 70.5 − 12.9) | n/a | n/a | 7879 (incl. VAE 6098; transformer 255) |
| NEFF compile (s) / NEFFs | compiler-only `CompilationTime` sum on the compiling rank (block + eager-op; a sum over NEFFs compiled concurrently, not a wall clock); Dynamo frame compile (trace + lowering); new `.neff` files in the row's cache | 201.2 (rank 0: block 51.6 + eager-op 149.6; Dynamo 12.0, lowering 0.19) / 47 new: 4 block (one per rank, 4 files >= 1 MiB) + 43 eager-op | 0.0 / 0 new (w1: 0.0 s, 0 new, w2: 0.0 s, 0 new, w3: 0.0 s, 0 new) | 0 compilations in iterations 2-3 (True) | transformer 255 s (AoT build) |
| counters per rank | whole process: TotalCompilations / PersistentHits / InMemoryHits | r0 38/1/22271 r1 2/42/22286 r2 5/39/22286 r3 2/42/22286 | w1: r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286<br>w2: r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286<br>w3: r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286 | w1 it2-3 generate deltas: r0 0/0/7406 r1 0/0/7410 r2 0/0/7410 r3 0/0/7410; r0 0/0/7406 r1 0/0/7410 r2 0/0/7410 r3 0/0/7410 | — |
| load (s) | rank-max `load checkpoint` (each rank's bf16 slices of the fp32 shards) | 3.7 | 3.6 (median 3.6; 3.5-3.6, n=3) | n/a | 343 cold / 28.4 warm |
| prepare_runtime / load() total (s) | rank-max: runtime + process group bring-up; the whole `load()` (runtime init, build on meta, load checkpoint, per-block compile wrap, warm-up forward, text encoder load) | 18.4 / 83.4 | 18.1 (median 17.4; 17.1-19.7, n=3) / 18.2 (median 18.2; 18.2-18.3, n=3) | n/a | — |
| text encoder load (s) | rank 0 `text encoder load` phase (umT5 fp32 `from_pretrained`, lazily mmap'd: the read lands in the first encode) | 0.9 | 0.7 (median 0.7; 0.7-0.7, n=3) | n/a | 94.4 cold / 8.3 warm (umT5 NEFF load) |
| e2e (s), script | rank 0: `prepare_runtime` start → first mp4 saved (load, warm-up, encode, 50 steps, host VAE decode at 12 threads, mp4) | **206.5** | **141.2 (median 140.5; 140.1-142.9, n=3)** | generate only: 103.6 (median 103.5; 103.3-104.1, n=6) | 394 (page cache dropped) / 56.4 (n=3) |
| e2e (s), script, latent output | `prepare_runtime` start → final latents (no decode) | 157.3 | 91.8 (median 91.2; 90.7-93.4, n=3) | generate only: 54.5 (median 54.5; 54.4-54.5, n=6) | — |
| launch wall (s), script | driver: torchrun spawn → exit (3 generates + decodes); spawn + imports → `prepare_runtime` | 421.3; spawn 4.5 | 355.5 (median 354.8; 354.3-357.5, n=3); spawn 4.4 (median 4.3; 4.2-4.5, n=3) | n/a | — |
| e2e (s), CLI | `difflet generate --backend neuron` process wall (torchrun ranks, then the driver's host VAE decode at OMP_NUM_THREADS=3) | 266.7 (own fresh cache; 47 new .neff) | 198.4 (median 198.9; 197.0-199.2, n=3) | n/a | 394 (page cache dropped) / 56.4 (n=3) |
| e2e (s), CLI, true cold | page cache dropped (`sudo -n`), NEFF cache warm: the trn2 definition of a cold start | 780.9 (Cached 84.4 → 0.4 GiB at the drop; NEFF warm) | n/a | n/a | 394 (page cache dropped) |
| CLI rank load+warm-up / stage (s) | the `[wan]` rank lines, rank-max (load+warm-up includes `prepare_runtime`) | 104.3 / 160.2 | 36.3 / 91.6 ; 34.2 / 89.6 ; 36.0 / 91.3 | n/a | — |
| CLI start → latents / → mp4 (s) | file mtimes: process start → latents.pt saved by rank 0; latents.pt → mp4 written (rank teardown + the driver's host VAE load, decode at 3 threads, export) | 164.6 / 101.3 | 95.9 / 102.5 ; 93.8 / 102.4 ; 95.6 / 102.4 | true cold: 678.0 / 102.0 | — |
| text encode / denoise / VAE decode (s) | rank-0 stage clocks: prompt encode (the negative prompt is encoded lazily inside the first denoise iteration), `_denoise`, host decode incl. VAE load and mp4 write | 2.01 (+ negative 0.26) / 53.6 / 49.1 (VAE 48.7) | 2.02 (+ negative 0.26) / 53.7 / 49.0 (VAE 48.7) | 0.80 (+ negative 0.25) / 53.6 / 48.8 (VAE 48.6) | — |
| DiT step median (ms) | inter-step deltas of the real loop, `torch.neuron.synchronize` both sides, step 0 excluded, n=49, rank 0 | evidence only: 1066.2 | **1067.2** (w2 1067.1, w3 1066.8) | 1067.1 (pooled, n=294) | 554.8 (isolated timer) |
| DiT step rank-max median (ms) | per step, the max over the 4 ranks (what the request waits for) | 1068.8 | 1070.9 | 1069.9 | — |
| DiT step mean / p90 / min / max (ms) | same series | 1067.0 / 1071.4 / 1060.6 / 1075.4 | 1068.1 / 1072.4 / 1061.8 / 1078.2 | 1067.7 / 1072.1 / 1061.2 / 1088.2 | — |
| DiT forward (ms) | per forward (host→device inputs, forward, sync, copy back), n=100; median (mean / p90) | 531.0 | 531.5 (531.6 / 534.1) | 531.2 (531.2 / 533.4, n=600) | — |
| throughput (steps/s) | 1 / mean step | evidence only: 0.937 | 0.936 | 0.937 | 1.80 |
| peak device mem (GiB/rank) | `torch_neuronx.max_memory_allocated()` per rank (the process's peak so far: after the launch, after generates 2-3) | 8.07 8.03 8.13 8.13 | 8.03 8.03 8.03 8.03 | 8.03 8.03 8.03 8.03 | null |
| fallbacks / unwarmed shapes | `track_fallbacks()` (load and every generate) / `app.unwarmed_shapes`, every rank | [] / [] | [] / [] ; [] / [] ; [] / [] | [] / [] | — |
| throttle | sysfs `ham_throttle` `entries`/`max` per core before vs right after every generate (semantics undocumented: a flag, not a verdict); step series p90/median (<= 1.10) and first-10 vs last-10 medians (<= 5 %) | sysfs entries/max changed across 3/3 generates (max seen ['128/255', '255/255']); p90/median max 1.005, first-10 vs last-10 max 0.08%; within limits | sysfs entries/max changed across 6/9 generates (max seen ['128/255', '255/255']); p90/median max 1.006, first-10 vs last-10 max 0.22%; within limits | step medians of the 3 generates per launch: spread 0.08% (max over the warm launches; flag above 2 %) | — |
| output | decoded frames (first generate of the launch; resident: generate 2 of the first warm launch): shape, finite, min/max/mean/std; mp4 bytes, sha256 | [1, 3, 9, 480, 832] finite=True [-1.000, 1.000] mean -0.122 std 0.532; 101970 B mp4 a9f9004fc1343eab… | [1, 3, 9, 480, 832] finite=True [-1.000, 1.000] mean -0.122 std 0.532; 101970 B mp4 a9f9004fc1343eab… | [1, 3, 9, 480, 832] finite=True [-1.000, 1.000] mean -0.122 std 0.532; 101970 B mp4 a9f9004fc1343eab… | (1,3,9,480,832) |
| final latents | sha256 of the fp32 latents, equal on the 4 ranks of every generate (else the launch failed) | 1 distinct: 8e5eaffed4cf6f3a… | 1 distinct: 8e5eaffed4cf6f3a… | 1 distinct: 8e5eaffed4cf6f3a… | — |
| W4 reproduction (D80) | one extra warm CLI run with W4's parameters: latents and mp4 sha256 equal to W4's | — | yes: latents 8e5eaffed4cf6f3a…, file e639f3146c361940…, mp4 a9f9004fc1343eab…; wall 200.8 s | — | — |

Checks per launch (every check's detail is in the JSON, `neuron.checks`):

- `script-cold`: 17/17 pass; counter_rule_cold: Total/PersistentHits/InMemoryHits r0 38/1/22271 r1 2/42/22286 r2 5/39/22286 r3 2/42/22286; new .neff 47
- `script-warm1`: 18/18 pass; counter_rule_warm: Total/PersistentHits/InMemoryHits r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286; new .neff 0
- `script-warm2`: 18/18 pass; counter_rule_warm: Total/PersistentHits/InMemoryHits r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286; new .neff 0
- `script-warm3`: 18/18 pass; counter_rule_warm: Total/PersistentHits/InMemoryHits r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286; new .neff 0
- `cli-w4check`: 11/11 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-warm1`: 8/8 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-warm2`: 8/8 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-warm3`: 8/8 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-truecold`: 8/8 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-cold`: 8/8 pass; file_rule_cold: new .neff 47

## Guidance 1.0 (one forward per step; the trn2-comparable row)

| metric | definition | cold launch | warm launch (n=3) | resident (2nd/3rd generate) | trn2 AoT baseline (20 steps, g=1.0) |
|---|---|---:|---:|---:|---:|
| compile wall (s) | rank-max `warmup forward` cold − warm (first warm launch) | **60.9** (= 73.9 − 13.0) | n/a | n/a | 7879 (incl. VAE 6098; transformer 255) |
| NEFF compile (s) / NEFFs | compiler-only `CompilationTime` sum on the compiling rank (block + eager-op; a sum over NEFFs compiled concurrently, not a wall clock); Dynamo frame compile (trace + lowering); new `.neff` files in the row's cache | 157.4 (rank 1: block 55.0 + eager-op 102.3; Dynamo 13.1, lowering 0.19) / 47 new: 4 block (one per rank, 4 files >= 1 MiB) + 43 eager-op | 0.0 / 0 new (w1: 0.0 s, 0 new, w2: 0.0 s, 0 new, w3: 0.0 s, 0 new) | 0 compilations in iterations 2-3 (True) | transformer 255 s (AoT build) |
| counters per rank | whole process: TotalCompilations / PersistentHits / InMemoryHits | r0 5/34/11162 r1 25/19/11171 r2 4/40/11171 r3 13/31/11171 | w1: r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171<br>w2: r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171<br>w3: r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171 | w1 it2-3 generate deltas: r0 0/0/3703 r1 0/0/3705 r2 0/0/3705 r3 0/0/3705; r0 0/0/3703 r1 0/0/3705 r2 0/0/3705 r3 0/0/3705 | — |
| load (s) | rank-max `load checkpoint` (each rank's bf16 slices of the fp32 shards) | 4.0 | 4.0 (median 4.0; 4.0-4.0, n=3) | n/a | 343 cold / 28.4 warm |
| prepare_runtime / load() total (s) | rank-max: runtime + process group bring-up; the whole `load()` (runtime init, build on meta, load checkpoint, per-block compile wrap, warm-up forward, text encoder load) | 20.3 / 87.3 | 17.7 (median 17.4; 17.2-18.6, n=3) / 18.7 (median 18.7; 18.6-18.7, n=3) | n/a | — |
| text encoder load (s) | rank 0 `text encoder load` phase (umT5 fp32 `from_pretrained`, lazily mmap'd: the read lands in the first encode) | 0.7 | 0.7 (median 0.7; 0.7-0.7, n=3) | n/a | 94.4 cold / 8.3 warm (umT5 NEFF load) |
| e2e (s), script | rank 0: `prepare_runtime` start → first mp4 saved (load, warm-up, encode, 50 steps, host VAE decode at 12 threads, mp4) | **186.3** | **114.1 (median 113.7; 113.6-115.2, n=3)** | generate only: 76.7 (median 76.6; 76.3-77.0, n=6) | 394 (page cache dropped) / 56.4 (n=3) |
| e2e (s), script, latent output | `prepare_runtime` start → final latents (no decode) | 136.4 | 65.0 (median 64.7; 64.3-66.0, n=3) | generate only: 27.6 (median 27.6; 27.5-27.6, n=6) | — |
| launch wall (s), script | driver: torchrun spawn → exit (3 generates + decodes); spawn + imports → `prepare_runtime` | 346.3; spawn 4.5 | 274.6 (median 274.2; 273.5-276.0, n=3); spawn 4.3 (median 4.3; 4.3-4.4, n=3) | n/a | — |
| e2e (s), CLI | `difflet generate --backend neuron` process wall (torchrun ranks, then the driver's host VAE decode at OMP_NUM_THREADS=3) | 239.9 (own fresh cache; 47 new .neff) | 173.0 (median 173.0; 171.8-174.3, n=3) | n/a | 394 (page cache dropped) / 56.4 (n=3) |
| e2e (s), CLI, true cold | page cache dropped (`sudo -n`), NEFF cache warm: the trn2 definition of a cold start | 747.1 (Cached 72.3 → 0.4 GiB at the drop; NEFF warm) | n/a | n/a | 394 (page cache dropped) |
| CLI rank load+warm-up / stage (s) | the `[wan]` rank lines, rank-max (load+warm-up includes `prepare_runtime`) | 105.1 / 133.9 | 36.5 / 65.0 ; 36.6 / 65.2 ; 39.1 / 67.6 | n/a | — |
| CLI start → latents / → mp4 (s) | file mtimes: process start → latents.pt saved by rank 0; latents.pt → mp4 written (rank teardown + the driver's host VAE load, decode at 3 threads, export) | 138.2 / 100.9 | 69.2 / 102.9 ; 69.3 / 101.6 ; 71.9 / 101.6 | true cold: 644.4 / 101.8 | — |
| text encode / denoise / VAE decode (s) | rank-0 stage clocks: prompt encode (the negative prompt is encoded lazily inside the first denoise iteration), `_denoise`, host decode incl. VAE load and mp4 write | 1.99 / 26.8 / 49.8 (VAE 49.5) | 2.07 / 26.7 / 48.5 (VAE 48.3) | 0.81 / 26.8 / 48.8 (VAE 48.5) | — |
| DiT step median (ms) | inter-step deltas of the real loop, `torch.neuron.synchronize` both sides, step 0 excluded, n=49, rank 0 | evidence only: 534.8 | **534.4** (w2 534.3, w3 533.9) | 534.6 (pooled, n=294) | 554.8 (isolated timer) |
| DiT step rank-max median (ms) | per step, the max over the 4 ranks (what the request waits for) | 536.0 | 535.5 | 536.3 | — |
| DiT step mean / p90 / min / max (ms) | same series | 535.3 / 536.9 / 529.5 / 546.4 | 534.8 / 537.8 / 529.5 / 541.8 | 535.0 / 539.0 / 528.8 / 548.1 | — |
| DiT forward (ms) | per forward (host→device inputs, forward, sync, copy back), n=50; median (mean / p90) | 531.2 | 530.9 (531.2 / 533.7) | 531.1 (531.2 / 533.8, n=300) | — |
| throughput (steps/s) | 1 / mean step | evidence only: 1.868 | 1.870 | 1.869 | 1.80 |
| peak device mem (GiB/rank) | `torch_neuronx.max_memory_allocated()` per rank (the process's peak so far: after the launch, after generates 2-3) | 8.03 8.03 8.13 8.13 | 8.03 8.03 8.03 8.03 | 8.03 8.03 8.03 8.03 | null |
| fallbacks / unwarmed shapes | `track_fallbacks()` (load and every generate) / `app.unwarmed_shapes`, every rank | [] / [] | [] / [] ; [] / [] ; [] / [] | [] / [] | — |
| throttle | sysfs `ham_throttle` `entries`/`max` per core before vs right after every generate (semantics undocumented: a flag, not a verdict); step series p90/median (<= 1.10) and first-10 vs last-10 medians (<= 5 %) | sysfs entries/max changed across 3/3 generates (max seen ['128/255', '255/255']); p90/median max 1.008, first-10 vs last-10 max 0.06%; within limits | sysfs entries/max changed across 7/9 generates (max seen ['128/255', '255/255']); p90/median max 1.011, first-10 vs last-10 max 0.20%; within limits | step medians of the 3 generates per launch: spread 0.13% (max over the warm launches; flag above 2 %) | — |
| output | decoded frames (first generate of the launch; resident: generate 2 of the first warm launch): shape, finite, min/max/mean/std; mp4 bytes, sha256 | [1, 3, 9, 480, 832] finite=True [-1.000, 1.000] mean -0.363 std 0.450; 88546 B mp4 e56c43137f8824ba… | [1, 3, 9, 480, 832] finite=True [-1.000, 1.000] mean -0.363 std 0.450; 88546 B mp4 e56c43137f8824ba… | [1, 3, 9, 480, 832] finite=True [-1.000, 1.000] mean -0.363 std 0.450; 88546 B mp4 e56c43137f8824ba… | (1,3,9,480,832) |
| final latents | sha256 of the fp32 latents, equal on the 4 ranks of every generate (else the launch failed) | 1 distinct: a703d237e9a9b537… | 1 distinct: a703d237e9a9b537… | 1 distinct: a703d237e9a9b537… | — |

Checks per launch (every check's detail is in the JSON, `neuron.checks`):

- `script-cold`: 15/15 pass; counter_rule_cold: Total/PersistentHits/InMemoryHits r0 5/34/11162 r1 25/19/11171 r2 4/40/11171 r3 13/31/11171; new .neff 47
- `script-warm1`: 16/16 pass; counter_rule_warm: Total/PersistentHits/InMemoryHits r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171; new .neff 0
- `script-warm2`: 16/16 pass; counter_rule_warm: Total/PersistentHits/InMemoryHits r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171; new .neff 0
- `script-warm3`: 16/16 pass; counter_rule_warm: Total/PersistentHits/InMemoryHits r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171; new .neff 0
- `cli-warm1`: 8/8 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-warm2`: 8/8 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-warm3`: 8/8 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-truecold`: 8/8 pass; file_rule_warm: new .neff 0, new HLO-cache files 0 (the CLI prints no counters; a warm launch adds nothing to the row's cache)
- `cli-cold`: 8/8 pass; file_rule_cold: new .neff 47

## Definitions

- **compile time**: rank-max `app.phase_seconds["warmup forward"]` on the launch with an empty cache (`DIFFLET_COMPILE_CACHE` fresh: `neuron/{neff,neff_local,hlo}` empty) minus the same on the next launch with the populated cache; the compiler-only `CompilationCache.CompilationTime` sum and the new `.neff` count beside it.
- **e2e cold / warm**: the bench script's rank-0 wall from `prepare_runtime` to the first mp4 saved; warm proven per launch by per-rank `TotalCompilations == 0 and PersistentHits >= 1` and no new `.neff`. The CLI rows are the `difflet generate` process wall (torchrun spawn and imports included; host VAE in the driver); the CLI prints no counters, so a warm CLI launch is proven by its cache gaining no file.
- **resident**: generates 2 and 3 in the same process (`InMemoryHits`), the serving steady state; e2e from the generate's start to its mp4.
- **DiT step**: `RealLoopStepTimer(sync=torch.neuron.synchronize)`, one stamp per denoise iteration (after the scheduler step), the device synchronized before the loop, step 0 excluded, n=49; median is the headline. The trn2 554.8 ms is an isolated forward timer at 20 steps, g=1.0 (not re-measured with this rule).

## Reproduction (neuron)

```bash
# one supervised job, guidance 5.0 then 1.0 (about 1.5-2 h of device time):
HF_HUB_OFFLINE=1 PYTHONSAFEPATH=1 PYTHONPATH=<worktree> \
    <venv>/bin/python -m benchmark.neuron_cold_warm --record-dir <records>
# re-render this file from the recorded launches (no device):
PYTHONSAFEPATH=1 PYTHONPATH=<worktree> <venv>/bin/python -m benchmark.neuron_cold_warm \
    --record-dir <records> --assemble-only
# the user-facing command each CLI row timed:
difflet generate --backend neuron --exec-mode compile --host-vae \
    --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 \
    --steps 50 --guidance-scale 5.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4
```

## Appendix: generic report, guidance 5.0 (`benchmark/report.py`)

Rendered by the shared renderer, headings demoted, its Trainium reproduction section dropped. Its fixed labels assume the AoT protocol: "compile (AOT, one-time)" is the warm-up compile above, and "cold start (page cache dropped)" is the NEFF-cold launch with a hot page cache (the true-cold CLI row is the page-cache-cold one), so its sentence that the cold/warm gap is the cold disk read does not apply: here the gap is the block compile and the cold eager-op NEFFs.

## Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** ok  
**Backend:** neuron  
**Device:** trn2.3xlarge / 4 NeuronCores (logical config 2) / 96 GiB/device  
**Timestamp:** 2026-10-11 10:44 UTC

> Best-performing configuration: tp=4, bf16, TorchNeuron per-block torch.compile, host fp32 umT5 on rank 0, host VAE

### Configuration

| key | value |
|---|---|
| model type | wan |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 832, 'num_frames': 9} |
| steps | 50 |

### End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 57.56 s |
| **e2e generate — cold start** (page cache dropped) | **3.4 min (206 s)** |
| **e2e generate — warm cache** | **2.4 min (141 s)** |
| peak device memory | 8.1 GB |

> Cold vs warm: **3.4 min (206 s) → 2.4 min (141 s)** (1.5× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

### Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 1.07 s | 1.07 s | 1.07 s | 1.06 s | 49 |
| end-to-end (warm) | 2.4 min (141 s) | 2.3 min (141 s) | 2.4 min (143 s) | 2.3 min (140 s) | 3 |

**Throughput:** 0.936 DiT steps/s

Per-step basis: **real loop, neuron synchronize both sides, step 0 excluded, n=49** — device-synced inter-step deltas of a real generate loop, step 0 excluded, the same rule the other device folders use (`benchmark/harness.py::RealLoopStepTimer`).

### Stage breakdown (one warm generate)

| stage | seconds |
|---|---|
| text_encoder_load_s | 0.71 |
| text_encode | 2.02 |
| negative_encode (inside denoise) | 0.26 |
| denoise | 53.66 |
| vae_decode (host, 12 threads) | 48.99 |

### Compile breakdown

| component | build time |
|---|---|
| warmup_forward_cold_s | 70.47 s |
| warmup_forward_warm_s | 12.90 s |
| neff_compile_s | 3.4 min (201 s) |
| block_neff_compile_s | 51.58 s |
| eager_op_neff_compile_s | 2.5 min (150 s) |
| dynamo_frame_compile_cold_s | 12.03 s |
| dynamo_frame_compile_warm_s | 10.07 s |
| dynamo_lower_s | 186.2 ms |
| load_and_warmup_cold_s | 83.42 s |
| load_and_warmup_warm_s | 18.27 s |

### Output validity

| field | value |
|---|---|
| shape | [1, 3, 9, 480, 832] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-1.0000, 1.0000] (mean -0.1217, std 0.5324) |
| note | decoded frames of the first warm generate; 101970 byte mp4 sha256 a9f9004fc1343eab167e70f4522f382243f40b7b83765b726e01d02cb01b7f34; latents (1,16,3,60,104) sha256 8e5eaffed4cf6f3a79a94dc11664c4f6b49767375fc6f5c451a39330bed990f6 |

### Toolchain

- `torch` = 2.14.0+cpu
- `torch-neuronx` = 2.14.3.0.0+unknown.dev
- `neuronx-cc` = 2.28.4568.0+0163fa9e
- `nki` = 0.7.0+36523665040.g7b8cdcef
- `diffusers` = 0.38.0
- `transformers` = 4.57.6
- `numpy` = 2.5.3

### Notes

- Commit 3763193977f4a508f72f713444d100f65fa88b08 in /home/ubuntu/Difflet/.claude/worktrees/neuron-phase1 for all 19 launches; no change under difflet/ at any launch; uncommitted files at the launches: benchmark/adapters/neuron.py, benchmark/bench.py, benchmark/models.py, benchmark/neuron_cold_warm.py, benchmark/neuron_wan_bench.py, benchmark/trn2/wan_2_1_neuron.json, benchmark/trn2/wan_2_1_neuron.md, tests/unit/benchmark/ (per-launch sha256 of the benchmark files in neuron.provenance). guidance_scale=5: two batch-1 DiT forwards per step (CFG).
- compile_seconds is the rank-max `warmup forward` of the cold script launch minus the first warm launch's (Dynamo trace + StableHLO lowering + neuronx-cc + NEFF load that the cache removes); nothing is built ahead of time, so there is no AoT compile phase.
- e2e_cold_seconds / e2e_warm are the script's `prepare_runtime` -> mp4 saved on rank 0 (NEFF cache empty / populated; the OS page cache was hot for both, Cached 84.3 GiB before the cold launch). The trn2 row's cold start dropped the page cache: see the true-cold CLI row.
- Counters (Total/PersistentHits/InMemoryHits, whole process): cold r0 38/1/22271 r1 2/42/22286 r2 5/39/22286 r3 2/42/22286; warm1 r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286; warm2 r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286; warm3 r0 0/39/22271 r1 0/44/22286 r2 0/44/22286 r3 0/44/22286
- Cold NEFFs: 47 new .neff files (4 >= 1 MiB block NEFFs); per rank compiled block/eager-op: r0 1/37, r1 1/1, r2 1/4, r3 1/1; dynamo unique graphs per rank [1, 1, 1, 1]
- Threads: ranks OMP_NUM_THREADS=3 (torch default 3); umT5 encode pinned at 12 (D70); the script's host VAE decode on rank 0 at 12; the CLI driver's host decode at its torch default, 3 under OMP_NUM_THREADS=3 (pre-flight: 97.4 s at 3 vs ~65 s at 6, same mp4).
- Peak device memory is torch_neuronx.max_memory_allocated() per rank, in GiB (2**30).
- The device-side script benchmark/neuron_wan_bench.py had 1 sha256 value(s) across the 19 launches (byte-identical).
- Re-judged after the run (the stored verdict was a driver false positive; the record as it ran is kept as record.original.json): g5.0/cli-warm1 (stored failed ['rank_lines', 'compile_clean']). The job stopped there and resumed with --skip-existing; no launch was repeated.
- Fallbacks and unwarmed shapes were [] on every rank of every launch and generate (track_fallbacks only sees eager ops; the compiled blocks are proven by the counters).

## Appendix: generic report, guidance 1.0 (`benchmark/report.py`)

Rendered by the shared renderer, headings demoted, its Trainium reproduction section dropped. Its fixed labels assume the AoT protocol: "compile (AOT, one-time)" is the warm-up compile above, and "cold start (page cache dropped)" is the NEFF-cold launch with a hot page cache (the true-cold CLI row is the page-cache-cold one), so its sentence that the cold/warm gap is the cold disk read does not apply: here the gap is the block compile and the cold eager-op NEFFs.

## Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** ok  
**Backend:** neuron  
**Device:** trn2.3xlarge / 4 NeuronCores (logical config 2) / 96 GiB/device  
**Timestamp:** 2026-10-11 11:29 UTC

> Best-performing configuration: tp=4, bf16, TorchNeuron per-block torch.compile, host fp32 umT5 on rank 0, host VAE

### Configuration

| key | value |
|---|---|
| model type | wan |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 832, 'num_frames': 9} |
| steps | 50 |

### End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 60.89 s |
| **e2e generate — cold start** (page cache dropped) | **3.1 min (186 s)** |
| **e2e generate — warm cache** | **114.15 s** |
| peak device memory | 8.1 GB |

> Cold vs warm: **3.1 min (186 s) → 114.15 s** (1.6× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

### Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 534.8 ms | 534.4 ms | 537.8 ms | 529.5 ms | 49 |
| end-to-end (warm) | 114.15 s | 113.67 s | 115.17 s | 113.61 s | 3 |

**Throughput:** 1.870 DiT steps/s

Per-step basis: **real loop, neuron synchronize both sides, step 0 excluded, n=49** — device-synced inter-step deltas of a real generate loop, step 0 excluded, the same rule the other device folders use (`benchmark/harness.py::RealLoopStepTimer`).

### Stage breakdown (one warm generate)

| stage | seconds |
|---|---|
| text_encoder_load_s | 0.71 |
| text_encode | 2.07 |
| negative_encode (inside denoise) | 0.00 |
| denoise | 26.74 |
| vae_decode (host, 12 threads) | 48.54 |

### Compile breakdown

| component | build time |
|---|---|
| warmup_forward_cold_s | 73.88 s |
| warmup_forward_warm_s | 12.99 s |
| neff_compile_s | 2.6 min (157 s) |
| block_neff_compile_s | 55.02 s |
| eager_op_neff_compile_s | 102.31 s |
| dynamo_frame_compile_cold_s | 13.10 s |
| dynamo_frame_compile_warm_s | 9.99 s |
| dynamo_lower_s | 188.0 ms |
| load_and_warmup_cold_s | 87.35 s |
| load_and_warmup_warm_s | 18.69 s |

### Output validity

| field | value |
|---|---|
| shape | [1, 3, 9, 480, 832] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-1.0000, 1.0000] (mean -0.3625, std 0.4496) |
| note | decoded frames of the first warm generate; 88546 byte mp4 sha256 e56c43137f8824ba4890a87e002327e039d9b23b40de7be2e7a8862e801b2210; latents (1,16,3,60,104) sha256 a703d237e9a9b537f7236e20b1cedf0ebbb3597111a8c6a3e7f05c6d5e8ea820 |

### Toolchain

- `torch` = 2.14.0+cpu
- `torch-neuronx` = 2.14.3.0.0+unknown.dev
- `neuronx-cc` = 2.28.4568.0+0163fa9e
- `nki` = 0.7.0+36523665040.g7b8cdcef
- `diffusers` = 0.38.0
- `transformers` = 4.57.6
- `numpy` = 2.5.3

### Notes

- Commit 3763193977f4a508f72f713444d100f65fa88b08 in /home/ubuntu/Difflet/.claude/worktrees/neuron-phase1 for all 19 launches; no change under difflet/ at any launch; uncommitted files at the launches: benchmark/adapters/neuron.py, benchmark/bench.py, benchmark/models.py, benchmark/neuron_cold_warm.py, benchmark/neuron_wan_bench.py, benchmark/trn2/wan_2_1_neuron.json, benchmark/trn2/wan_2_1_neuron.md, tests/unit/benchmark/ (per-launch sha256 of the benchmark files in neuron.provenance). guidance_scale=1: one forward per step.
- compile_seconds is the rank-max `warmup forward` of the cold script launch minus the first warm launch's (Dynamo trace + StableHLO lowering + neuronx-cc + NEFF load that the cache removes); nothing is built ahead of time, so there is no AoT compile phase.
- e2e_cold_seconds / e2e_warm are the script's `prepare_runtime` -> mp4 saved on rank 0 (NEFF cache empty / populated; the OS page cache was hot for both, Cached 72.2 GiB before the cold launch). The trn2 row's cold start dropped the page cache: see the true-cold CLI row.
- Counters (Total/PersistentHits/InMemoryHits, whole process): cold r0 5/34/11162 r1 25/19/11171 r2 4/40/11171 r3 13/31/11171; warm1 r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171; warm2 r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171; warm3 r0 0/39/11162 r1 0/44/11171 r2 0/44/11171 r3 0/44/11171
- Cold NEFFs: 47 new .neff files (4 >= 1 MiB block NEFFs); per rank compiled block/eager-op: r0 1/4, r1 1/24, r2 1/3, r3 1/12; dynamo unique graphs per rank [1, 1, 1, 1]
- Threads: ranks OMP_NUM_THREADS=3 (torch default 3); umT5 encode pinned at 12 (D70); the script's host VAE decode on rank 0 at 12; the CLI driver's host decode at its torch default, 3 under OMP_NUM_THREADS=3 (pre-flight: 97.4 s at 3 vs ~65 s at 6, same mp4).
- Peak device memory is torch_neuronx.max_memory_allocated() per rank, in GiB (2**30).
- The device-side script benchmark/neuron_wan_bench.py had 1 sha256 value(s) across the 19 launches (byte-identical).
- Re-judged after the run (the stored verdict was a driver false positive; the record as it ran is kept as record.original.json): g5.0/cli-warm1 (stored failed ['rank_lines', 'compile_clean']). The job stopped there and resumed with --skip-existing; no launch was repeated.
- Fallbacks and unwarmed shapes were [] on every rank of every launch and generate (track_fallbacks only sees eager ops; the compiled blocks are proven by the counters).
