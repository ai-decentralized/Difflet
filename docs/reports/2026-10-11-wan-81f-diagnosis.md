# Wan 2.1 at 81 frames on trn2: where the time goes (2026-10-11)

Wan 2.1 T2V-14B, 480x832x81 (32,760 visual tokens per step), 50 steps, CFG 5.0, tp4,
one trn2.3xlarge (4 logical cores, LNC2), neuronx-cc 2.26.6360, runtime 2.34.10.
Diagnosis only (profiles, logs, one timestamped run) up to the fix in 26c0ffd. Data and tools:
`artifacts/wan-diagnosis-20261011/`.

## 1. End-to-end split (single-core-attention graph, before 26c0ffd)

One `difflet generate` with per-line timestamps, NRT INFO logs and a 5 s page-cache sampler
(`e2e/generate_filtered.log`, `e2e/mem.log`, `e2e/steps.json`):

| stage | s | note |
|---|---:|---|
| process start, imports | ~17 | |
| text encoder load | 67 | 3.3 GB not in page cache; ~8 s when warm |
| transformer load | 252 | weights to device ~47 s (5.7 GB re-read from EBS), NEFF load cache ~30 s, main NEFF load ~167 s |
| text encode | ~1 | |
| denoise | 981.7 | 19.63 s/step = 2 CFG calls x 9.8 s |
| VAE stage | 126 | subprocess start ~23 s, VAE + 6 chunk NEFFs ~25 s, decode + mp4 ~79 s |
| total | 1447.6 | |

The paper's "Wan spends 325 s in the chunked VAE decode" does not reproduce on this host.
`benchmark/parse_generate.py` labels the last load line `vae_decoder` when there is no role
marker, so the 212 s in `benchmark/trn2-eval/wan_2_1.json` under `vae_decoder` is the
transformer load.

## 2. The NKI attention ran on one physical core

`ops_impl.attention` picks `attention_cte[2]` (an SPMD launch over the two physical cores of an
LNC2 core) only when `NEURON_RT_VIRTUAL_CORE_SIZE=2` at trace time. The Wan CLI orchestrator
left it unset except for ring CP, as did the resident serving worker; HunyuanVideo, Qwen-Image
and MiniMax-H3 set 2. A device profile of a 2-layer 81-frame block
(`profiles/wan81_l2/breakdown.txt`) puts every attention instruction on pcore 0: TensorE 380 ms,
ScalarE (exp) 187 ms, VectorE 164 ms, and nothing on pcore 1, whose engines wait on semaphores.
Attention instructions cover 203 of the block's 474 ms.

A/B on the same host (profiler captures, `tools/wan_block_neff.py` builds the reduced-depth NEFFs):

| 480x832x81 | single-core attention | VC2 | VC2 + `--vectorize-strided-dma` |
|---|---:|---:|---:|
| 2 layers | 473.8 ms | 313.5 ms (-33.8%) | 304.3 ms (-35.8%) |
| 40 layers (one DiT call) | 9.592 s | | 6.117 s (-36.2%) |
| TensorE active (2 layers) | 47% | 77% | 80% |
| avg throttle util limit, pcore 1 | 0.59 | 0.82 | |

The quantization branch fixed the CLI path on 2026-10-05 (38997f9, -15% at 9 frames, where
attention is a smaller share) but not serving. That gap is the paper's undiagnosed "resident
DiT step slower than standalone" (Wan 19.6 vs 17.0 s, LTX-2 459 vs 417 ms; be5dfe3 measured
LTX-2 459.7 -> 413.7 ms with the same change). 26c0ffd ports the fix to this branch and extends
it to serving and the step benchmark. Re-measured rows: section 6.

## 3. Slow transformer load: spill DMA rings

The 81-frame NEFF carries 5.5 GB of "dma rings spill" per logical core (9 frames: 146 MB). The
runtime builds these static descriptor rings on the host at load (process RSS rises to 34 GB,
one ring alone has 1.74 M descriptors), which is the ~167 s main-NEFF load. The 2-layer NEFF has
the same 5.5 GB because the compiler builds one layer body and loops it, so the cost does not
depend on depth. The same rings are the 86.7 MB allocation that failed when tp2cp2 loaded
("Failed to allocate 86.717MB (dma rings spill)").

`--enable-dge --internal-enable-dge-levels=spill_reload` reached the backend
(`--dge-levels spill_reload,transpose,io,...`) but changed nothing: identical rings
(5.475 GB), load time and 2-layer runtime (473.8 vs 474.0 ms).

## 4. Inefficiencies shared with the other DiTs

From the 9-frame profile (`profiles/wan9f_tp4/breakdown.txt`) and the 2-layer 81-frame one.
HunyuanVideo (320x512x61) shows the same pattern with lower MFU (19% vs Wan 9-frame 30%), so
these are not Wan-specific:

- All compute engines idle (waiting on DMA) for 35-40% of the time on the busier pcore.
- Compiler-inserted transposes: 26% of TensorE instruction time at 9 frames, plus DMA transposes
  (20 GB per forward at 9 frames; 44 GB per 2 layers at 81 frames on the spill queues).
- HAM activity throttling: a 4/8 utilisation limit for about 40% of the run.
- At 9 frames the 4,680-token sequence is not a multiple of 128; the 72-token tail runs as
  128x6 / 128x12 tiles that cost as much as full ones (47 ms of TensorE per forward). Not an
  issue at 81 frames (32,760 = 255x128 + 120).

## 5. Compile time

HLO generation takes 823 s at 81 frames for 40 layers and 830 s for 2 layers, against 23 s at
9 frames (158 s at 33 frames): a one-time trace of the attention kernel that grows about with
the square of the sequence, not a per-layer cost. neuronx-cc takes 1365 s for 2 layers and
1676 s for 40.

## 6. Re-measured rows after 26c0ffd

Pending: `benchmark/trn2/eval/queue_wan_vc2.sh` (results in `benchmark/trn2-eval-vc2/`, latent
agreement with the single-core graph in `benchmark/trn2/eval/vc2/`).
