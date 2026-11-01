# HunyuanVideo 1.5 on the AOT (trainium) backend: on-device evidence

Plan: `docs/superpowers/plans/2026-10-30-hv15-aot-parity.md`. Host: `trn2.3xlarge`, one Neuron
device, 4 logical NeuronCores (logical config 2), 96 GB HBM. The spike's logs and NEFFs live
outside the repo, in the spike script's work directory (`S` below). Campaign evidence is curated
under `E = artifacts/verification-2026-10-30/hv15-aot/`.

## Summary (Phase 6, 2026-11-01)

HunyuanVideo 1.5 (`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, rev `286be7ce`)
generates video end to end on the Trainium AOT backend with true CFG, TP, CFG-parallel, SP,
CP (ulysses, gather_kv), DP, JSONL batching and TeaCache, at the 320x512x61 verification shape
and at the default 480x848x121 shape. The VAE decodes on the host (LIMIT); the compiled device
VAE is BLOCKED on the toolchain. Ring CP is N/A (mask), as for HunyuanVideo 1.0. The sections
below this summary hold the per-cell evidence, commands and logs.

### Toolchain

| stack | versions |
|---|---|
| AOT (`DIFFLET_BACKEND=trainium`, venv `difflet-neuron`) | neuronx-cc 2.26.6360.0 · torch-neuronx 2.9.0.2.15.32035 · neuronx-distributed 0.19.28492 · NxDI 0.10.18399 · nki 0.5.0 · torch 2.9.1 · torch-xla 2.9.0 · diffusers 0.38.0 |
| native reference (`--backend neuron`, venv `native-312`) | neuronx-cc 2.28.7631 · torch-neuronx 2.14.3.0.1 · torch 2.14.0 · nki 0.7.0 · diffusers 0.38.0 |
| runtime (both) | NRT 2.35.30.0 · driver 2.31.7.0 · trn2.3xlarge, 4 logical cores (LNC 2), 96 GiB HBM, 124 GiB host RAM |

The two stacks draw different bf16 seed-42 noise (torch 2.9.1 vs 2.14), so AOT-vs-native parity
is gated by feeding the AOT DiT the native run's `embeds.pt` and noise (Phase 0, Phase 1 re-gate).

### Support matrix

PASS = verified on device · LIMIT = works with a stated cost or restriction · BLOCKED = does not
work for a reason outside Difflet (toolchain) · N/A = not applicable by design · NOT MEASURED =
not run in this campaign. Shape 320x512x61, 50 steps, seed 42 unless the row says otherwise.

| feature | native | AOT | evidence (this doc) |
|---|---|---|---|
| TP2 (CFG's two passes in sequence) | NOT MEASURED | **PASS** | Phase 1 re-gate tp2: 1 step cos 0.999160 vs native; 3 steps 0.91x native's distance to diffusers |
| TP2 x CFG2 (default layout) | **PASS** (1325 ms/step) | **PASS** (1573-1607 ms/step) | Phase 0 native; Phase 1 re-gate tp2cfg 0.999124; Phase 2 and re-run (50 steps) |
| TP4 | NOT MEASURED | **PASS** | Phase 1 re-gate tp4 0.999131; Phase 2 tp4 1731 ms/step |
| TP4 + SP | NOT MEASURED | **PASS** at 320x512x61; refused before compile at 480x848x121 (N % 4 = 2, 6.4 s) | Phase 1 re-gate tp4sp 0.999142; Phase 2 tp4sp 1601 ms/step; Phase 3 |
| TP2 x CP2 ulysses | NOT MEASURED | **PASS** | Phase 1 re-gate 0.999132, 3 steps 0.53x; Phase 2 1783 ms/step |
| TP2 x CP2 gather_kv | NOT MEASURED | **PASS**, watch item (3-step gate 2 at 1.25x, R24) | Phase 1 re-gate 0.999149, 3 steps 1.25x; Phase 2 1858 ms/step |
| CP ring | NOT MEASURED (registry keeps ring for native) | **N/A (mask)**: refused before compile | Phase 1 "ring": joint ring kernel has no text-mask path; refused in 6.7 s; refused before encode since `b7b939b` |
| DP (`--dp 2 --tp-degree 2`) | NOT MEASURED | **PASS** | Phase 2 dp2tp2 (2690 ms/step per replica); DP bit-identity 4/4 |
| JSONL batch (`--requests`) | NOT MEASURED | **PASS** (after `47b2f74`) | Phase 5: DiT loaded once (27.0 s); both requests bit-identical to their single runs; MP4s byte-identical |
| TeaCache fixed cadence | **PASS** (cadence 2: 20 skipped, cos 0.932074) | **PASS** (cadence 2: 20 skipped, -41% denoise, cos 0.938644) | Phase 4a; Phase 4 reference (R26); ruling R30 |
| TeaCache online-delta | NOT MEASURED | **PASS** (alpha 0.6: 17 skipped, -35%, cos 0.918296) | Phase 4b |
| TeaCache adaptive | NOT MEASURED | **LIMIT**: weak signal, not recommended (R31) | Phase 4c: cubic R² 0.406, Pearson −0.116; 18 skipped, cos 0.910548, visibly degraded; probe doubles HBM (17.59 vs 9.69 GiB/core) |
| host VAE (AOT default) | NOT MEASURED | **LIMIT**: correct, slow (409-431 s at 61 frames, 2699 s at 121 frames) | Phase 2 host decode; Phase 3 retry attempt 2 |
| device VAE | **PASS** (31.4 s at 61 frames, 89.0 s at 121 frames) | **BLOCKED**: `vae15` does not compile on neuronx-cc 2.26 (NCC_EVRF007 / NCC_IBIR229 / NCC_EXTP003) | Phase 1 "aot-vae"; user decision: host decode this round |
| default shape 480x848x121, TP2 x CFG2 | **PASS** (11178 ms/step, same host) | **PASS** (13072 ms/step; compile 3970 s); VAE LIMIT as above | Phase 3 native reference; Phase 3 retry attempt 2 (after `1a9625b`; set through `DIFFLET_HUNYUAN15_COMPILER_ARGS`, a string byte-identical to the final default for this shape — `d001314` pins it in `test_hv15_default_compiler_args_exact_strings` — so the result stands without a rerun) |

### Performance, AOT vs native (TP2 x CFG2, 50 steps)

| metric | shape | native | AOT | AOT / native |
|---|---|---|---|---|
| DiT step median | 320x512x61 | 1325 ms | 1573 ms (graph v1), 1577 ms (v2), 1607 ms (HEAD flags, Phase 4c) | 1.19-1.21x |
| DiT step median | 480x848x121 | 11178 ms | 13072 ms | 1.17x |
| compile wall | 320x512x61 | 923.8 s (DiT + device VAE) | 462.8 s (DiT 448.1 s, graph v2; no VAE to compile) | 0.50x |
| compile wall | 480x848x121 | 2807.1 s | 3970.0 s (host RSS peak 116.5 of 124 GiB) | 1.41x |
| VAE decode | 320x512x61 | 31.4 s (device) | 409-431 s (host, fp32, tiled) | ~13x |
| VAE decode | 480x848x121 | 89.0 s (device) | 2699.2 s (host; 28-35 GiB RSS) | 30x |
| encode + denoise | 480x848x121 | 763 s | 765 s | 1.00x |
| generate e2e | 480x848x121 | 960.7 s | 4132.8 s (host VAE 2699 s, cold DiT load 574 s) | 4.3x |

Other AOT layouts at 320x512x61 (ms/step): TP4+SP 1601 · TP4 1731 · ulysses 1783 · gather_kv
1858 · DP replica (TP2, serial CFG) 2690. Peak HBM per core: TP2xCFG2 9.69 GiB (23.58 GiB at
480x848x121), TP4 5.76, TP4+SP 5.25, CP2 8.69-8.89.

### Gates and why

- **Phase 1 correctness (R16).** Gate 1: at 1 step, cosine ≥ 0.999 against native TP2xCFG2 on
  the same inputs. All six layouts pass (0.999124-0.999160), and each is closer to diffusers'
  CPU reference per forward (0.999566-0.999609) than native is (0.999136). Gate 2: at 3 steps,
  distance to diffusers' CPU bf16 reference ≤ 1.15x native's own distance (native cos
  0.983536). The plan's 0.999 at 3 steps is unattainable for any backend: native itself sits at
  0.9835 against diffusers.
  **Exceedance, verbatim:** gather_kv is at **1.25x** (cos 0.979483 vs diffusers),
  deterministic. Ruled PASS with a watch item (R24): it has the best 1-step parity (0.999149),
  and the 3-step spread between AOT layouts on identical inputs is itself 0.9765-0.9870.
- **Phase 2 (R16).** A functional and performance matrix: each 50-step cell exits 0, has no
  NaN/Inf in its latents and decodes to a mid frame that shows the prompt. 50-step cosines
  across layouts (0.955-0.969) are informational.
- **TeaCache (R26, R30).** The plan's latents cosine ≥ 0.99 is unattainable on either stack.
  The cells are judged against native's own cadence 2 on the same settings.
  **Exceedance, verbatim:** TeaCache cos is **~0.91-0.94 on both stacks**: native cadence 2
  0.932074; AOT cadence 2 0.938644, online-delta 0.918296, adaptive 0.910548.
- **DP and JSONL.** Bit-identity of every request against a single-replica / single run.

### Bugs found and fixed

| commit | found | bug and fix |
|---|---|---|
| `1bfe577`, `86d4715` | device, Phase 1 | runner: partial multi-core AoT stages exported `NEURON_RT_NUM_CORES=<n>`, which NRT 2.35 refuses unless it is 1 or the whole device; such stages now name their cores with `NEURON_RT_VISIBLE_CORES`, and multiples of 8 keep `NUM_CORES` (R19) |
| `fd2a8aa` | device, Phase 1 | the t2v image-stream switch was a Python bool of a tensor, which segfaulted the CFG-parallel trace; it is now in-graph |
| `1a9625b` | device, Phase 3 | NCC_IRPX901 at 480x848x121 on the in-graph fp32 valid-text count; `text_valid` is now computed on the host and passed as DiT input #10 |
| `dca10f3`, `420d2fa`, `d001314` | device, Phase 3 retry and Phase 4 | NCC_IRPX901 at 480x848x121 with `--enable-ccop-compute-overlap`; the option is now kept only up to 16384 image tokens (dropping it costs 13.4% per step at 320x512x61), from one token-count helper shared with the cache key, which now carries the compiler args |
| `67d3d92` | device, Phase 4c | the TeaCache probe's `reset_state` used the non-jit NxDModel API; every adaptive request died with `AttributeError`; `prev_mod` is now zeroed through `.state` |
| `47b2f74` | device, Phase 5 | outside `--requests`, `--negative-prompt` reached no AOT stage; `_run_aot` forwards it and the legacy request carries it |
| `29661ce` | CPU parity tests, Task 3 | the host token refiner used the transformer's mlp_ratio; it now uses 4.0, as diffusers' context_embedder does |
| `376c628` | unit coverage test | the parallel15 RowParallelLinear site now states `reduce_dtype=torch.float32` (NxD's default; no behaviour change) |

Related hardening from device findings: `b7b939b` (host VAE by default; ring and multi-shape
refused before encode), `a41ecc9` (`graph_version` in the AOT stage cache key, after stale
artifacts were silently reused), `93dde78` (DiT shape rules checked before encode).

### Known limitations and follow-ups

- **Device VAE: BLOCKED.** `vae15` does not compile on neuronx-cc 2.26 at any tried shape. The
  host decode is the default (LIMIT: 13x native's device decode at 61 frames, 30x at 121
  frames). Follow-ups: a segmented decoder (the HunyuanVideo 1.0 pattern) or a toolchain bump.
- **Ring CP: N/A (mask).** The Trainium joint ring-attention kernel has no text-mask path, and
  HV1.5's text is always padded; the AOT path refuses `--cp-mode ring` before encode. Native
  keeps ring. Same as HunyuanVideo 1.0.
- **Adaptive TeaCache: not recommended.** The calibration signal is weak (cubic R² 0.406,
  Pearson −0.116) and the output is visibly degraded. The probe loads a second copy of the DiT
  weights (17.59 vs 9.69 GiB/core at 320x512x61), so at the default shape (23.58 GiB/core
  without it) it cannot fit in 24 GiB per core. Use `--teacache-cadence 2`. Follow-up: load
  only the probe's block-0 weights.
- **DP + adaptive TeaCache:** the lazy probe compile can race between replicas (as for
  HunyuanVideo 1.0). Not measured. TeaCache flags with `--requests` are refused by the CLI.
- **`--mode latency`** resolves hv15 to TP1 x CFG2, which was not verified; pass
  `--tp-degree 2 --cfg-parallel` (the default) or another layout from the matrix.
- **Negative prompt on native (R32).** Native hv15 still drops `--negative-prompt` outside
  `--requests`, and in batch mode neither backend applies a CLI `--negative-prompt` to request
  lines without one. The fix touches native-shared function bodies and is left for a later round.
- **TP4 + SP at 480x848x121** is refused before compile (N % 4 = 2); use TP2 x CFG2.
- **Host memory at the default shape:** the compile peaked at 116.5 of 124 GiB host RAM and the
  DiT load at 102.2 GiB RSS.
- **Default shape on HEAD:** the 480x848x121 run used the compiler flags (overlap off) through
  `DIFFLET_HUNYUAN15_COMPILER_ARGS`, before `dca10f3`/`420d2fa` made them the default for that
  shape; it was not re-run on the final code.

## Spikes

Task 0, 2026-10-30, branch `worktree-hv15-aot-parity-plan` at `57f7d74` plus the spike script.

**What was compiled.** Per controller ruling R2 (the hv15 processors do not exist yet), the spike
does not build a transformer block. It traces the raw bounded attention call,
`difflet.backends.trainium.ops_impl.attention.attention(q, k, v, bound_min=..., bound_max=...,
tp_q=True, tp_k=True, tp_out=False, scale=1/sqrt(128))`, on ONE logical NeuronCore with no
collectives. Under `NEURON_RT_VIRTUAL_CORE_SIZE=2` that call is nkilib `attention_cte[2]` (the LNC2
grid launch), which is the same kernel and call that HV1.0's `_joint_attention` uses. q, k and v are
`[heads=8, S, 128]` bf16 (16 heads / tp2, with B=1 folded into the heads, in the `tp_q=True` layout).
The bounds are built in-graph from a runtime `key_valid_len` input with expand and cat only. The
count passes through fp32 before int32, and `mask_to_contiguous_bounds` is not used. Every real
query row gets `[0, N + valid_text)` and every padded query row gets `[0, 1)`.

**Compile path.** `torch_neuronx.trace` (single core, no NxD ModelBuilder), with the HV backbone's
compiler flags plus `--lnc=2`:
`--model-type=transformer -O1 --tensorizer-options='--enable-ccop-compute-overlap' --auto-cast=none --lnc=2 --internal-hlo2tensorizer-options='--verify-hlo=true'`.
The traced HLO holds exactly one `AwsNeuronCustomNativeKernel` call (the NKI kernel), so the NEFF
really contains the kernel and is not a fallback. Env set inside the script:
`NEURON_RT_VISIBLE_CORES=0`, `NEURON_LOGICAL_NC_CONFIG=2`, `NEURON_RT_VIRTUAL_CORE_SIZE=2`.

**Reference.** A CPU fp32 softmax(q k^T / sqrt(128)) v over keys `[0, key_valid_len)`, compared on
the real query rows. Padded query rows are checked against `v[:, 0]`, since they attend key 0 only.
A negative control on head 0, rel-L1 against an *unmasked* reference over all keys, must come out
large, which proves that `bound_max` was honored.

**Toolchain.** neuronx-cc 2.26.6360.0+6f180f47 · torch-neuronx 2.9.0.2.15.32035 · neuronx-distributed
0.19.28492 · nki 0.5.0 · nkilib (venv-bundled) · torch 2.9.1 · torch-xla 2.9.0 · NRT 2.35.30.0 ·
driver 2.31.7.0.

**Command form.** Every case runs through the plan's `run_py.sh` wrapper (the venv `bin/` on PATH):
`bash .superpowers/sdd/2026-10-30-hv15-aot-parity/run_py.sh scripts/hv15_aot_attention_spike.py <args>`.
The script saves `traced.pt` and `compile.json` per case. A rerun reuses that NEFF and runs only
the forward, which is how a forward was obtained after the first attempts lost the device to an
unrelated process (see Notes).

### (a) bounded, 320x512x61: PASS

`--mode bounded --height 320 --width 512 --num-frames 61 --valid-text 37`

| field | value |
|---|---|
| N (image tokens) | 16·20·32 = 10240 |
| q / k / v | `[8, 12288, 128]` (12225 real rows, padded to a multiple of 128) |
| key_valid_len | 10277 (N + 37) |
| compile | **157.9 s** (an earlier identical compile took 271.7 s while the host CPUs were shared with another job's pytest run) |
| forward | runs; 0.018 s on device |
| NaN / Inf | none / none |
| rel-L1 vs fp32 ref (real rows) | **3.09e-3** |
| max-abs vs fp32 ref | 4.85e-4 |
| padded rows vs `v[:,0]` max-abs | 0.0 |

Log: `S/bounded_320_b.log`. The negative control was added after this run (see the gather_kv row,
same kernel and bounds).

### (b) gather_kv (sharded q vs full k), 320x512x61: PASS at kernel level (no compiler error)

`--mode gather_kv --height 320 --width 512 --num-frames 61 --valid-text 37`

| field | value |
|---|---|
| q | `[8, 7168, 128]` (N/2 + 1985 = 7105 real rows, padded to a multiple of 128) |
| k / v | `[8, 12288, 128]` |
| key_valid_len | 10277 |
| compile | **130.2 s**, no error. `NCC_INLA001` / `NCC_IBIR243` were **not** reproduced. |
| forward | runs; 0.014 s on device |
| NaN / Inf | none / none |
| rel-L1 vs fp32 ref (real rows) | **3.08e-3** |
| max-abs vs fp32 ref | 5.03e-4 |
| head-0 rel-L1 vs *unmasked* ref (negative control) | 0.441, so the bounds are honored |
| padded rows vs `v[:,0]` max-abs | 0.0 |

Logs: `S/gather_kv_320.log` (compile; its first forward failed to get a core, see Notes) and
`S/gather_kv_320_run.log` (forward on the saved NEFF).

So the claim in `ops_impl/attention.py` (the `joint_ulysses_attention` docstring, "the gather_kv
route's sharded-query-vs-full-key shape is what neuronx-cc rejects") does **not** hold for the
bounded kernel *in isolation* on 2.26.6360.0 when both lengths are multiples of 128 and
q_len < kv_len. The recorded failures came from other contexts:
- HV1.0 tp2 cp2 gather_kv hit `NCC_INLA001`, an SBUF allocation error on a
  `concatenate [140,1536,1]`, compiling the **whole CP-2 DiT graph** (DEVELOPER.md). On 2.26 it
  drifted to `NCC_IBIR243`, behind a separate seqlen_q % 128 blocker (10176 = 79.5×128). That is a
  graph-level failure, and this kernel-only spike cannot reproduce or rule it out.
- LTX-2's `NCC_IBIR243` came from the bound path on **cross**-attention (video q against 1024 text
  keys, so q_len > kv_len). That is not this shape.

### (c) bounded at the default 480x848x121: NOT COMPLETED (over the time budget, killed)

`--mode bounded --height 480 --width 848 --num-frames 121 --valid-text 37`
N = 31·30·53 = 49290; q / k / v `[8, 51328, 128]` (51275 real rows); key_valid_len 49327.

Started at 19:16 UTC and killed by hand at 20:03 UTC (46 min) with no compiler error and no NEFF:
- **~29 min was the in-process trace** (the Python frontend trace, before neuronx-cc launched).
  The NKI kernel is traced per tile, and at 51k tokens that is about 401 q-tiles by 401 kv-tiles
  per head. The process RSS grew to ~7 GB.
- **17+ min was the neuronx-cc backend** (`walrus_driver --optlevel 2`), still running at
  ~24 GB RSS when it was killed.

For scale: the 12288-token case (a) traced and compiled in 158 s in total. So whether the
51k-token bounded call compiles on one core is **unknown, not failed**. What the run does show is
that a single 51k bounded attention costs more than 45 min of compile on its own. Log:
`S/bounded_480.log`.

### Decision

- **bounded: OK.** The Task 3 pad design stands as planned: pad the joint sequence N + 1985 to a
  multiple of 128 in-graph, real rows get `[0, N + valid_text)`, padded rows get `[0, 1)`, and the
  bounds are built with expand/cat after an fp32 count. The kernel compiles and is accurate at the
  320x512x61 shape (rel-L1 3.1e-3 bf16 vs fp32), and padded query rows come out exactly `v[0]` with
  no NaN, so slicing them off is safe. Nothing in Task 3 must change.
- **gather_kv: PASS at the kernel level, not BLOCKED.** The sharded-q vs full-k bounded call
  compiles and is correct on one core. Task 6 should therefore **not** pre-emptively make gather_kv
  a compile-time BLOCKED fail-fast. It should keep a fail-fast for per-rank q_len % 128 != 0 (the
  HV1.0 blocker; for hv15 the q shard N/cp + 1985 must be padded to a multiple of 128 like the joint
  sequence) and then compile the real tp2 cp2 graph. Residual risk: the HV1.0 `NCC_INLA001` was a
  whole-graph SBUF failure with the all-gather and concatenate in the graph, and this spike does not
  exercise that. If Task 6's full gather_kv compile hits it, the plan's fallback applies: gather_kv
  is BLOCKED with a fail-fast, ulysses is the CP path, and the follow-up is a toolchain bump or
  masked SDPA at O(L²).
- **Default shape (480x848x121): open risk, not a decision.** Case (c) did not finish within its
  budget. The pad design itself is not in question: it is the same kernel and the same bounds, and
  only the length changes. The risk is compile time and compile memory at about 51k tokens.
  Task 3 and later tasks should first bring up the AOT DiT at 320x512x61 and treat the default
  shape's compile as a measured item, not an assumption. Two levers if needed: the per-core
  sequence under ulysses after its all-to-all is the same full 51k, but gather_kv's per-rank q
  shard halves the q-tiles.

### Notes

- The first forward attempts of (a) and (b) failed with
  `NRT:nrt_allocate_neuron_cores Logical Neuron Core(s) not available - Requested:lnc0-lnc0 Available:0 ... (cores busy, ret=-16)`.
  A `pytest tests/unit -n 6` run started by another session held all 4 cores at the time. Once the
  device freed up, the forwards ran on the saved NEFFs (for (b)) or after a recompile (for (a)).
  This was not a property of the kernel.
- `torch_neuronx.trace` was chosen because it is the simplest single-core path, and it compiles
  the same `attention()` op and the same NKI `attention_cte[2]` kernel that the NxD ModelBuilder
  path compiles. It runs no SPMD or collective lowering, which is intended here.

## Campaign (Task 10), Phase 0 and Phase 1: 2026-10-30

Branch `worktree-hv15-aot-parity-plan`, campaign base `376c628`. Curated evidence (logs with host
identifiers scrubbed, per-step `meta.json` with wall time and neuron-monitor per-core peak HBM,
mid-frame PNGs, sha256 lists of the work-dir tensors) lives under
`E = artifacts/verification-2026-10-30/hv15-aot/`. Raw run dirs, work-dir tensors and NEFFs stay
outside the repo. Every CLI call ran through the plan's `run_difflet.sh` wrapper, wrapped by a
campaign cell runner that samples `neuron-monitor` once per second (`memory_used`, per-core sum
of the runtime's usage breakdown). Common flags: `--model-id
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --height 320 --width 512
--num-frames 61 --seed 42 --prompt "a cinematic shot of a red fox running through a snowy forest"`,
`HF_HUB_OFFLINE=1`, snapshot rev `286be7ce`.

### Phase 0: host and native reference

**Host check** (`.claude/skills/difflet-device-verify/scripts/env_check.sh`): trn2.3xlarge, 4
logical cores (LNC 2), 96 GiB HBM, 0 busy processes at start; 1.2 TB free; HF token present;
HV1.5 snapshot 50 GB in the HF cache. `<repo>/.venv` is absent, as ruled (R5: no `.venv`).

**Environment finding: the native backend does not run in `/home/ubuntu/venvs/difflet-neuron`.**
That venv is the AOT stack (torch 2.9.1, torch-neuronx 2.9.0.2.15, neuronx-cc 2.26.6360,
NxD 0.19.28492, NxDI 0.10.18399, nki 0.5.0, diffusers 0.38.0). A native `difflet compile` there
fails in every rank at process-group init, before any model code:
`AssertionError: Unknown backend type neuron` (`difflet/backends/neuron/runtime.py:228`,
`dist.init_process_group("neuron")`). The `neuron` process-group backend comes from TorchNeuron,
which is installed in `/home/ubuntu/venvs/native-312` (torch 2.14.0, torch-neuronx 2.14.3.0.1,
neuronx-cc 2.28.7631, nki 0.7.0, diffusers 0.38.0). Not a Difflet bug; the native reference runs
use that venv (same wrapper, `PATH` overridden through `HV15_ENV_FILE`). AOT cells use
`difflet-neuron`.

**Cross-stack noise.** The two stacks draw different bf16 initial noise for the same seed:
`randn_tensor(..., dtype=bfloat16)` with seed 42 at the 320x512x61 latent shape `[1,32,16,20,32]`
hashes `0ed37154…` on torch 2.9.1 and `02b85e79…` on torch 2.14.0. Drawing fp32 and rounding to
bf16 on torch 2.9.1 gives `02b85e79…`, bit-identical to torch 2.14's bf16 draw (the behaviour
`docs/verification/neuron-backend-results.md` describes). The hv15 encode stage also runs on the
two different torch versions. So an AOT CLI run and a native CLI run do not start from the same
inputs, and the native-parity gate below feeds the AOT DiT the native run's `embeds.pt` and the
native noise (`scripts/hv15_aot_denoise_from_inputs.py --noise-seed-fp32-round 42`).

**Native reference, TP2 x CFG2** (`DIFFLET_BACKEND=neuron`, native-312 venv, `--tp-degree 2 --cfg-parallel`):

| step | outcome | wall | stage timings (from the log) | peak HBM per core (GiB) | evidence |
|---|---|---|---|---|---|
| `compile` | PASS | 923.8 s | transformer load 179.0 s (warmup forward 153.1 s); vae load 30.9 s | 9.72 / 9.62 / 10.20 / 9.62 | `E/phase0-native/compile.log` |
| `generate --steps 3` | PASS | 235.9 s | encode 101.2 s (first, cold page cache); denoise 30.1 s, median 3728 ms (first 25096 ms); decode 32.0 s | 9.58 / 9.66 / 9.92 / 9.66 | `E/phase0-native/gen3.log`, `gen3_mid.png` |
| `generate --steps 50` | PASS | 176.2 s | encode 6.8 s; denoise 79.4 s, median 1325 ms (first 14416 ms); decode 31.4 s | 9.66 / 9.66 / 9.90 / 9.66 | `E/phase0-native/gen50.log`, `gen50.mp4`, `gen50_mid.png` |

- `embeds.pt` sha256 `eba584fa…004b4f`, identical in both native runs.
- `latents.pt` sha256: 3 steps `4646fa9b…7d3cf6`, 50 steps `33baa167…a1897d` (full hashes in
  `E/phase0-native/sha256.txt`).
- The 50-step video: 61 frames at 320x512, minimum per-frame pixel std 79.0, mean frame-to-frame
  difference 36.5 grey levels; the mid frame shows a red fox running through snow.
- The 3-step median (3728 ms) is above the 50-step median (1325 ms): with 3 steps the median is
  the second step, which (inference) still carries post-warm-up cost. Use the 50-step figure as
  the native per-step time at this shape.

### Phase 1: correctness gates (3 steps, 320x512x61, seed 42)

**How the gates are evaluated here.** Each AOT cell runs `difflet compile` and then `difflet
generate --steps 3 --keep-work-dir` on the trainium backend; the AOT stack's own encode stage
writes `embeds.pt` and draws the noise. Same-stack cells compare their `latents.pt` with
aot-tp2's. The native comparison (aot-tp2 only) runs the compiled AOT DiT on the native run's
`embeds.pt` and native noise through `scripts/hv15_aot_denoise_from_inputs.py`, and a third
reference, diffusers' own pipeline on the CPU in bf16 from the same embeds and noise
(`scripts/hv15_diffusers_cpu_latents.py`), arbitrates. Metrics come from
`scripts/hv15_aot_compare_latents.py` (cosine, rel-L1 = sum|a-b| / sum|ref|, max-abs, NaN/Inf).

**The 0.999 cosine gate at 3 steps is below the bf16 noise floor for this model.** With
identical embeds and noise, three bf16 implementations of the same pipeline land this far
apart after 3 steps:

| pair (3 steps, identical inputs) | cosine | rel-L1 | max-abs |
|---|---|---|---|
| AOT TP2 vs native TP2xCFG2 | 0.982189 | 0.1712 | 3.82 |
| AOT TP2 vs diffusers CPU bf16 | 0.985041 | 0.1576 | 4.06 |
| native TP2xCFG2 vs diffusers CPU bf16 | 0.983536 | 0.1723 | 3.35 |

and after 1 step (a single DiT forward per branch, x0 = noise - v):

| pair (1 step, identical inputs) | cosine | rel-L1 | max-abs |
|---|---|---|---|
| AOT TP2 vs native TP2xCFG2 | 0.999160 | 0.0434 | 0.56 |
| AOT TP2 vs diffusers CPU bf16 | 0.999609 | 0.0291 | 0.38 |
| native TP2xCFG2 vs diffusers CPU bf16 | 0.999136 | 0.0442 | 0.56 |

Native is no closer to diffusers than AOT is; AOT is slightly closer at both depths. The 3-step
trajectory amplifies per-forward bf16 differences (three steps of size 1/3 from pure noise), so
no pair of bf16 implementations meets 0.999 there, native vs diffusers included. This campaign
therefore reports both depths: the per-forward gate (1 step, cosine >= 0.999 against native,
which AOT meets) and the 3-step distance against the noise floor in the first table. Gate
verdicts below say which one they use. Evidence: `E/phase1/aot-tp2/cmp_*.json`,
`native_inputs{1,3}_vs_native.json`, `E/phase1/diffusers-cpu/` (diffusers 1 step 161.8 s,
3 steps 616.5 s on the host CPU).

**Inputs.** The AOT stack's encode gives `embeds.pt` sha `3051691e…`, the native stack's
`eba584fa…`: the Qwen2.5-VL embeddings differ slightly (prompt cos 0.999560, negative cos
0.999399, bf16 max-abs 7 and 9), the ByT5 embeddings and all four masks are bit-identical. The
seed-42 noise differs as described in Phase 0. `native_noise.pt` (torch 2.9 fp32 draw rounded to
bf16) sha `4785aa5f…`.

#### aot-tp2 (TP2, CFG's two passes in sequence)

| step | outcome | wall | detail | peak HBM per core (GiB) |
|---|---|---|---|---|
| `compile --tp-degree 2` | DiT PASS, VAE FAIL (see aot-vae) | 623.9 s | DiT compiled in 566.8 s into `hunyuan_video_15_aot_dit/039f757f7229a3ac`; the aot_vae stage then failed `NCC_EVRF007` | n/a (compile) |
| `generate --steps 3`, attempt 1 | FAIL, **bug 1** | 124.3 s | encode 103.1 s, then the aot_transformer stage died in `nrt_init`: ``NEURON_RT_NUM_CORES` must request one core, or the whole device (multiple of 8)`` | — |
| `generate --steps 3`, after fix `1bfe577` | DiT PASS; VAE stage exits (no artifact) | 222.1 s | encode 51.4 s; aot transformer load 134.7 s; denoise 10.2 s, median 2691 ms (first 4874 ms) | 10.25 on each of the 2 cores |
| DiT on native inputs, 3 steps | see tables | 52.8 s | load 35.4 s; denoise 10.1 s, median 2717 ms | 10.25 / 10.25 |
| DiT on native inputs, 1 step | see tables | — | load 35.8 s | — |

- **Gate (native, per forward): PASS**, cosine 0.999160 at 1 step; 3 steps 0.982189, inside the
  bf16 floor (native vs diffusers 0.983536). No NaN / Inf in any AOT latents.
- The serial-CFG layout runs the host refiner twice per step (cond and uncond); 2691 ms per step
  on 2 cores against native TP2xCFG2's 1325 ms on 4 cores (50-step median) is expected, not
  comparable.
- The DiT load in the CLI (134.7 s) vs the helper (35.4 s) is page-cache state, an inference: the
  CLI run was the first to read this artifact after the compile.
- `latents.pt` sha `aeaab2fb…` (CLI); full list in `E/phase1/aot-tp2/sha256.txt`.
- The device VAE does not compile (aot-vae), so the MP4 is missing. For a visual check the CLI
  latents were decoded with diffusers' VAE on the CPU (bf16, tiled 256 px):
  `E/phase1/aot-tp2/gen3_cpudecode.mp4`, `gen3_cpudecode_mid.png`, which shows a red fox running
  through snow (3 steps, so soft and noisy, like native's `E/phase0-native/gen3_mid.png`).
- First attempt's NRT diagnostic (from the kernel log, the run's own log was overwritten by the
  rerun): `E/phase1/aot-tp2/gen3_attempt1_nrt_dmesg.log`, `gen3_attempt1.meta.json`.

**Bug 1 (`1bfe577`), `fix(cli/runner)`: partial multi-core AoT stages exported
`NEURON_RT_NUM_CORES=<n>`, which NRT 2.35 accepts only as 1 or a whole device.** Any AoT stage
on 1 < n < all cores (here TP2 serial CFG on a 4-core trn2.3xlarge) died at runtime init. The
runner now exports `NEURON_RT_VISIBLE_CORES` (the first n cores of the inherited or detected
allocation) for those stages; one-core stages keep `NEURON_RT_NUM_CORES=1`, and a shell-exported
`NEURON_RT_NUM_CORES` is still respected. Control: the same NEFFs launched by hand with
`NEURON_RT_VISIBLE_CORES=0,1` ran before the fix; after it the CLI cell runs.
Unit test `tests/unit/cli/test_cli_runner.py::test_partial_multi_core_stage_names_its_cores`.

#### aot-vae: BLOCKED (vae15 does not compile on neuronx-cc 2.26.6360)

The single-graph `vae15` decoder (diffusers' `HunyuanVideo15Decoder3D` with the channel-repeat
workaround, traced whole per tile with `torch_neuronx.trace`, `--model-type=unet-inference -O1
--auto-cast=none`) fails in neuronx-cc at every shape tried, so neither the plan's parity cell
nor any CLI generate can produce an MP4 on the AOT path:

| shape (latent tile) | flags | result | log |
|---|---|---|---|
| 61 frames, 256 px tile (16x16), the CLI's | default | `[NCC_EVRF007] Instructions generated by compiler 13,279,851 exceeds the typical limit of 10,000,000` | `E/phase1/aot-tp2/compile.log` |
| 61 frames, 128 px tile (8x8), native's tile | default | `[NCC_IBIR229] State buffer allocation failed` on `GenericCopy bf16<128 x 512> -> <128 x 416>` | `E/phase1/aot-vae-control/tile128_61f.log` |
| 9 frames, 256 px tile, the plan's aot-vae cell | default | `[NCC_IBIR229]` on `GenericCopy bf16<128 x 512> -> <128 x 64>` | `E/phase1/aot-vae/tile256_9f.log` |
| 9 frames, 256 px tile | `--model-type=transformer -O1 --auto-cast=none` | `[NCC_IBIR229]`, same op | `E/phase1/aot-vae-control/probe_transformer_9f.log` |
| 1 frame, 256 px tile | default | `[NCC_EXTP003] Instructions generated by compiler 2359296 exceeds the typical limit of 300000` (`_convolution.3952`) | `E/phase1/aot-vae-control/tile256_1f.log` |

Not a campaign artifact: the module is unchanged since the M6a import (`a4936a3`, May 2026,
older toolchain); HV1.0's AOT VAE needed a 16-NEFF segmented decoder to get past a neuronx-cc
lowering bug in the same position, and the native backend's device VAE decodes HV1.5 with 128 px
tiles through TorchNeuron. The fix is a design choice and is left to the controller (NEEDS_CONTEXT):
a segmented vae15 (HV1.0 pattern), a host decode on the AOT path (diffusers' CPU VAE, 254 s for
61 frames at 320x512 in bf16 here), or a toolchain bump. The parity script now supports the
diffusers reference (`--reference diffusers`, PSNR), but that path is not exercised until a
decoder compiles.

#### aot-tp2cfg (TP2 x CFG2, the default layout)

| step | outcome | wall | detail | peak HBM per core (GiB) |
|---|---|---|---|---|
| `compile --tp-degree 2 --cfg-parallel`, attempts 1-2 | FAIL, **bug 2** | 46.1 s, 45.2 s | SIGSEGV while ModelBuilder traced the batch-2 DiT; faulthandler: `backbone15.py:674 if is_t2v:` | — |
| attempt 3 (first fix, dimless `amax()`) | FAIL | 44.8 s | `RuntimeError: Shapes are not compatible for broadcasting: f32[1,729,2048] vs. f32[1,729,1152]` at the new flag: the torch-xla trace lowered `amax()` with no dims to no reduction | — |
| attempt 4 (fix `fd2a8aa`, full `max()`) | DiT PASS, VAE FAIL (aot-vae) | 733.8 s | DiT compiled in 619.5 s into `hunyuan_video_15_aot_dit/58d5f319c159f86a` | — |
| `generate --steps 3` | DiT PASS; VAE stage exits | 279.3 s | encode 109.4 s; load 126.0 s; denoise 10.9 s, median 1593 ms (first 7701 ms) | 9.69 on each of 4 cores |
| DiT 1 step, AOT embeds, seed 42 | — | — | load 35.3 s; 2746 ms | — |

- `embeds.pt` sha `3051691e…`, identical to aot-tp2's (the AOT stack's encode is deterministic).
- **Gate (vs aot-tp2): per forward PASS**, cosine 0.999820 / rel-L1 0.0207 / max-abs 0.34 at
  1 step; 3 steps 0.986882 / 0.1631 / 2.73, inside the bf16 floor. No NaN / Inf.
  `latents.pt` sha `c99ce43a…`.
- **Device check (2), int64 `text_order` and the masks scattered under CFG: PASS.** Every DiT
  input, `text_order` (int64) and both masks included, goes through
  `scatter_to_process_group_spmd` on the cfg group in this layout (`backbone15.py` forward), and
  the per-forward result matches the serial-CFG layout to cosine 0.99982.
- Evidence: `E/phase1/aot-tp2cfg/` (`compile_attempt{1,2,3}*` for the failures).

**Bug 2 (`fd2a8aa`), `fix(trainium/hv15)`: the hv15 trace forward took a Python bool of
`torch.all(image_embeds == 0)`** (diffusers' t2v branch, kept in the static-reorder fork). That
executes the partial graph mid-trace; with CFG parallelism `image_embeds` comes from the SPMD cfg
scatter and the trace segfaults. The switch is now in-graph: a float `max() > 0` flag multiplies
the image stream and forms its mask (identical to diffusers' two branches; the host already
refuses non-zero `image_embeds` on the AOT path). The graph changed, so aot-tp2 was recompiled
after the fix (below). Unit tests: `test_trace_forward_keeps_the_t2v_switch_in_graph`,
`test_trace_forward_reduces_the_t2v_flag_with_a_full_max`
(`tests/unit/backends/trainium/test_hv15_parallel.py`).

**aot-tp2 after bug 2.** `difflet compile --tp-degree 2` after `fd2a8aa` returned in 0.2 s
("compiled into …039f757f7229a3ac (0.2 s)"): the stage reuses an existing artifact, and its key
has no code version, so the pre-fix NEFF would have stayed in use. The stale dir was moved aside
(`039f757f7229a3ac.stale-pre-fd2a8aa`, not deleted) and the DiT recompiled in 462.2 s (483.9 s
wall, `E/phase1/aot-tp2/compile_postfix.log`). The post-fix 3-step generate (154.4 s wall; DiT
load 111.1 s; denoise 9.3 s, median 2719 ms) gives `latents.pt` **bit-identical** to the pre-fix
run (`aeaab2fb…`, `cmp_gen3_postfix.work_latents__vs__aot-tp2_gen3.work_latents.json`): the
in-graph switch changes nothing numerically, and the AOT DiT is deterministic across a recompile.
The native-input comparisons above used the pre-fix NEFF and therefore stand.

#### aot-tp4 (TP4, CFG in sequence)

| step | outcome | wall | detail | peak HBM per core (GiB) |
|---|---|---|---|---|
| `compile --tp-degree 4` | DiT PASS, VAE FAIL (aot-vae) | 596.6 s | DiT compiled in 572.3 s into `hunyuan_video_15_aot_dit/2eef1e7d012eea0a` | — |
| `generate --steps 3` | DiT PASS; VAE stage exits | 78.7 s | encode 9.6 s; DiT load 23.7 s; denoise 6.8 s, median 1749 ms (first 3336 ms) | 5.75 on each of 4 cores |

- `embeds.pt` sha `3051691e…` (same as aot-tp2); `latents.pt` sha `579a01b9…`.
- **Gate (vs aot-tp2): per forward PASS**, cosine 0.999720 / rel-L1 0.0262 / max-abs 0.38 at
  1 step; 3 steps 0.985482 / 0.1718 / 3.95, inside the bf16 floor. No NaN / Inf.
- **Device check (3), TP > 1 numerics: PASS.** TP2 and TP4 (different shardings and
  all-reduce trees of the same DiT) agree per forward to 0.99972, TP2 x CFG2 to 0.99982, and the
  TP2 DiT agrees with diffusers' CPU pipeline per forward to 0.99961 (closer than native TP2 x
  CFG2, 0.99914).
- Evidence: `E/phase1/aot-tp4/`.

#### aot-tp4sp (TP4 + Megatron SP)

| step | outcome | wall | detail | peak HBM per core (GiB) |
|---|---|---|---|---|
| `compile --tp-degree 4 --sp` | DiT PASS, VAE FAIL (aot-vae) | 355.2 s | DiT compiled in 327.3 s into `hunyuan_video_15_aot_dit/6e2e215d0c101025` | — |
| `generate --steps 3` | DiT PASS; VAE stage exits | 196.2 s | encode 123.8 s (inference: cold page cache for the text encoders); DiT load 32.7 s; denoise 6.1 s, median 1618 ms (first 2891 ms) | 5.25 on each of 4 cores |

- `embeds.pt` sha `3051691e…`; `latents.pt` sha `6d9ef64b…`.
- **Gate (vs aot-tp4): per forward PASS**, cosine 0.999701 / rel-L1 0.0274 / max-abs 0.38 at
  1 step; 3 steps 0.983685 / 0.1805 / 3.66, inside the bf16 floor. No NaN / Inf.
- SP lowers the per-core peak (5.25 vs 5.75 GiB) and the step median (1618 vs 1749 ms) against
  plain TP4 at this shape.
- Evidence: `E/phase1/aot-tp4sp/`.

#### aot-tp2cfgsp (TP2 x CFG2 + SP; extra cell for device check (1))

| step | outcome | wall | detail | peak HBM per core (GiB) |
|---|---|---|---|---|
| `compile --tp-degree 2 --cfg-parallel --sp` | DiT PASS, VAE FAIL (aot-vae) | 664.4 s | DiT compiled in 639.5 s into `hunyuan_video_15_aot_dit/7a2dd70cb206fe42` | — |
| `generate --steps 3` | DiT PASS; VAE stage exits | 165.6 s | encode 11.4 s; DiT load 123.7 s; denoise 5.9 s, median 1469 ms (first 2926 ms) | 9.78 on each of 4 cores |

- `embeds.pt` sha `3051691e…`; `latents.pt` sha `60dba0e6…`.
- vs aot-tp2cfg: 1 step cosine 0.999717 / rel-L1 0.0265 / max-abs 0.41; 3 steps 0.983093 /
  0.1834 / 3.15. No NaN / Inf.
- **Device check (1), SP + CFG scatters with TP-local ranks: PASS.** With CFG2 x TP2 the global
  ranks are 0-3 and each cfg group's TP ranks are 0-1; SP's sequence scatter must use the
  TP-local rank (`tp_rank_util`); with the global rank, ranks 2-3 would index past a TP2 group's
  two slices (inference from the code, not run). The per-forward result matches the no-SP
  TP2 x CFG2 layout to 0.99972, the same agreement as every other layout pair. (The tp4sp cell
  has no CFG, so it does not exercise this.)
- Evidence: `E/phase1/aot-tp2cfgsp/`.

#### aot-tp2cp2-ulysses (TP2 x CP2, `--cp-mode ulysses`)

| step | outcome | wall | detail | peak HBM per core (GiB) |
|---|---|---|---|---|
| `compile --tp-degree 2 --cp-degree 2 --cp-mode ulysses` | DiT PASS, VAE FAIL (aot-vae) | 731.2 s | DiT compiled in 615.9 s into `hunyuan_video_15_aot_dit/16020e50bb9f6f06` | — |
| `generate --steps 3` | DiT PASS; VAE stage exits | 315.2 s | encode 125.5 s; DiT load 133.6 s; denoise 6.8 s, median 1834 ms (first 3184 ms) | 8.74 on each of 4 cores |

- `embeds.pt` sha `3051691e…`; `latents.pt` sha `72274456…`.
- **Gate (vs aot-tp2): per forward PASS**, cosine 0.999727 / rel-L1 0.0260 / max-abs 0.38 at
  1 step; 3 steps 0.985001 / 0.1746 / 3.67, inside the bf16 floor. No NaN / Inf.
- Evidence: `E/phase1/aot-tp2cp2-ulysses/`.

#### aot-tp2cp2-gather_kv (TP2 x CP2, `--cp-mode gather_kv`)

The Task 0 spike passed at kernel level, so the cell ran (it is not BLOCKED). The open question
from the spike, whether the whole CP-2 DiT graph hits HV1.0's `NCC_INLA001`, is answered: it
does not on neuronx-cc 2.26.6360 at this shape.

| step | outcome | wall | detail | peak HBM per core (GiB) |
|---|---|---|---|---|
| `compile --tp-degree 2 --cp-degree 2 --cp-mode gather_kv` | DiT PASS, VAE FAIL (aot-vae) | 354.8 s | DiT compiled in 332.8 s into `hunyuan_video_15_aot_dit/3c8c02fac8aec15c`; no `NCC_INLA001` / `NCC_IBIR243` | — |
| `generate --steps 3` | DiT PASS; VAE stage exits | 62.5 s | encode 8.0 s; DiT load 20.6 s; denoise 7.0 s, median 1899 ms (first 3252 ms) | 8.94 on each of 4 cores |

- `embeds.pt` sha `3051691e…`; `latents.pt` sha `25f719c2…`.
- **Gate (vs aot-tp2): per forward PASS**, cosine 0.999730 / rel-L1 0.0260 / max-abs 0.37 at
  1 step; 3 steps 0.987910 / 0.1599 / 2.73, inside the bf16 floor. No NaN / Inf.
- Evidence: `E/phase1/aot-tp2cp2-gather_kv/`.

#### ring: N/A (mask), rejected before compile

Same as HV1.0: the registry keeps ring for the native backend, and only the AOT path refuses it.

| command | wall | result |
|---|---|---|
| `compile --tp-degree 2 --cp-degree 2 --cp-mode ring` | **6.7 s** | exit 1 at the start of the aot_transformer stage, before any trace: `NotImplementedError: HunyuanVideo 1.5 on the trainium backend has no ring context parallelism: the Trainium joint ring-attention kernel has no text-mask path for the padded text stream. Use --cp-mode ulysses (or gather_kv).` |
| `generate … --cp-mode ring` (no artifact) | 32.0 s | the host encode runs first (10.2 s), then `[hv15] no valid compiled transformer artifact at …/hunyuan_video_15_aot_dit/2b98ee6d55211302 (manifest missing or configuration changed); run \`difflet compile\` with the same flags first.` The ring message only appears from `compile`. |

The generate path's message points at `difflet compile`, which then names the real reason in
seconds; a pre-encode ring check in `_run_aot` would say it directly (the Task 8 review's
deferred minor). Evidence: `E/phase1/ring/`.

### Phase 0-1 summary

| cell | outcome | compile (DiT) | generate wall (3 steps) | step median | peak HBM / core | 1 step vs ref (cos / rel-L1 / max-abs) | 3 steps vs ref (cos) |
|---|---|---|---|---|---|---|---|
| native TP2xCFG2 (reference) | PASS | 923.8 s wall (full compile) | 235.9 s | 1325 ms (50 steps) | 9.92 | ref | ref |
| aot-tp2 | **PASS** (per forward); latents only | 566.8 s (462.2 s post-fix) | 154.4 s | 2719 ms | 10.25 (2 cores) | vs native 0.999160 / 0.0434 / 0.56 | vs native 0.982189 (floor: native vs diffusers 0.983536) |
| aot-tp2cfg | **PASS** (per forward) after bug 2 | 619.5 s | 279.3 s | 1593 ms | 9.69 | vs aot-tp2 0.999820 / 0.0207 / 0.34 | 0.986882 |
| aot-tp4 | **PASS** (per forward) | 572.3 s | 78.7 s | 1749 ms | 5.75 | vs aot-tp2 0.999720 / 0.0262 / 0.38 | 0.985482 |
| aot-tp4sp | **PASS** (per forward) | 327.3 s | 196.2 s | 1618 ms | 5.25 | vs aot-tp4 0.999701 / 0.0274 / 0.38 | 0.983685 |
| aot-tp2cfgsp (extra) | **PASS** (per forward) | 639.5 s | 165.6 s | 1469 ms | 9.78 | vs aot-tp2cfg 0.999717 / 0.0265 / 0.41 | 0.983093 |
| aot-tp2cp2-ulysses | **PASS** (per forward) | 615.9 s | 315.2 s | 1834 ms | 8.74 | vs aot-tp2 0.999727 / 0.0260 / 0.38 | 0.985001 |
| aot-tp2cp2-gather_kv | **PASS** (per forward) | 332.8 s | 62.5 s | 1899 ms | 8.94 | vs aot-tp2 0.999730 / 0.0260 / 0.37 | 0.987910 |
| ring | **N/A (mask)**, rejected in 6.7 s | — | — | — | — | — | — |
| aot-vae | **BLOCKED** (vae15 does not compile on neuronx-cc 2.26) | — | — | — | — | — | — |

- Generate wall times include the host encode (8-126 s depending on page cache) and the
  aot_vae stage's immediate exit; every AOT generate exits 1 because no VAE artifact exists.
  The step median is from the `[hv15] request 0: denoise …` line (`636edaf` added it).
- No AOT latents file in Phase 1 contains NaN or Inf.
- Not measured here: the decoded-frame (PSNR) distance to diffusers at 9 frames. It needs a
  device VAE; the latents-level comparison with diffusers' CPU pipeline at 61 frames stands in
  for it (AOT is closer to diffusers than native is, at 1 and 3 steps).

**Device checks from code review.**
1. SP + CFG uses TP-local ranks: **PASS** (aot-tp2cfgsp, 1 step 0.999717 vs TP2 x CFG2).
2. int64 `text_order` and masks scattered under CFG: **PASS** (aot-tp2cfg, 1 step 0.999820 vs
   serial CFG; every input goes through the cfg scatter).
3. TP > 1 numerics: **PASS** (TP2, TP4, CFG, SP and CP layouts agree per forward at
   0.99970-0.99982; TP2 vs diffusers CPU 0.999609).

**Bug ledger (Phase 0-1).**
- `1bfe577` fix(cli/runner): partial multi-core AoT stages exported `NEURON_RT_NUM_CORES=<n>`,
  which NRT 2.35 refuses unless it is 1 or the whole device; such stages now name their cores
  with `NEURON_RT_VISIBLE_CORES`.
- `fd2a8aa` fix(trainium/hv15): the trace forward took a Python bool of
  `torch.all(image_embeds == 0)`, which segfaults the CFG-parallel trace; the t2v switch is now
  in-graph (full `max()`; a dimless `amax()` traced to no reduction).
- Open, not fixed (needs a design decision): vae15 does not compile (aot-vae).
- Tooling: `636edaf` (AOT step-median log line), `fe4c51c` (comparison and reference scripts).

## Campaign (Task 10), Phase 2 and Phase 3: 2026-10-31

HEAD at start `b9d9301`. Since Phase 1, `b7b939b` made the AOT `aot_vae` stage decode on the
host by default (diffusers' `AutoencoderKLHunyuanVideo15`, fp32, tiled), so every AOT generate
now writes an MP4; `difflet compile` no longer compiles a VAE. The DiT graph is unchanged since
`fd2a8aa`, so `difflet compile` reuses the Phase 1 DiT artifacts (a manifest hit in under 7 s;
the Phase 1 compile time is quoted next to it). Same runner, wrapper, venvs and common flags as
Phase 0-1, plus `--steps 50`. The cell runner now also samples host RSS every 2 s per stage
process (`host_rss_peak_gib` in each `meta.json`). Cells were driven with direct CLI commands
(not `run_cells.sh`/`verify_cli.py`): the plan's prompt and `--cp-mode` are then explicit in
every command, and the latents are kept for the cosine check.

### Phase 2: parallelism matrix (50 steps, 320x512x61, seed 42)

**Gate (ruling R16).** Correctness was gated in Phase 1; Phase 2 is a performance and
functionality matrix. Each 50-step cell must exit 0 with no NaN/Inf in its latents and write
an MP4 whose mid frame shows the prompt (a red fox running through snow). The latents cosine
against the AOT tp2cfg 50-step run is reported for information only.

**Columns.** *compile*: wall of `difflet compile` (a DiT manifest hit here). *generate*: the
staged wall time of `difflet generate` (host encode + DiT load + denoise + host decode).
*load*: `aot transformer load` (weights onto the cores). *step median*: from `[hv15] request 0:
denoise …`. *HBM*: neuron-monitor per-core peak during generate. *decode*: `host decode` of
the `aot_vae` stage. *RSS*: peak host RSS of the `aot_transformer` / `aot_vae` stage process.

| cell | outcome | compile s (Phase 1 DiT compile) | generate s | load s | denoise s / step median ms | HBM GiB/core | host decode s | RSS GiB DiT / VAE | cos vs tp2cfg | evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| tp2cfg (`--tp-degree 2 --cfg-parallel`) | **PASS** | 6.6 (619.5) | 802.8 | 273.6 | 79.8 / 1573 | 9.69 x4 | 417.0 | 37.4 / 18.5 | ref (no NaN/Inf) | `E/phase2/tp2cfg/` |
| tp4 (`--tp-degree 4`) | **PASS** | 5.8 (572.3) | 687.0 | 144.8 | 87.7 / 1731 | 5.76 x4 | 422.7 | 19.3 / 18.4 | 0.968992 (no NaN/Inf) | `E/phase2/tp4/` |
| tp4sp (`--tp-degree 4 --sp`) | **PASS** | 6.1 (327.3) | 544.8 | 22.1 (warm) | 81.2 / 1601 | 5.25 x4 | 409.4 | 21.0 / 18.5 | 0.955830 (no NaN/Inf) | `E/phase2/tp4sp/` |
| tp2cp2-ulysses (`--tp-degree 2 --cp-degree 2 --cp-mode ulysses`) | **PASS** | 6.6 (615.9) | 1235.9 | 697.0 (cold) | 90.5 / 1783 | 8.69 x4 | 417.5 | 35.9 / 19.4 | 0.963593 (no NaN/Inf) | `E/phase2/tp2cp2-ulysses/` |
| tp2cp2-gather_kv (`--tp-degree 2 --cp-degree 2 --cp-mode gather_kv`) | **PASS** | 6.9 (332.8) | 606.8 | 67.2 (warm) | 94.1 / 1858 | 8.89 x4 | 414.1 | 35.9 / 17.2 | 0.955108 (no NaN/Inf) | `E/phase2/tp2cp2-gather_kv/` |
| dp2tp2 (`--tp-degree 2 --dp 2`, 1 request) | **PASS** | 6.1 (462.2, TP2) | 748.6 | 155.4 / 156.0 | 137.1 / 2690 | 9.98 / 9.98 / 10.25 / 10.25 | 415.6 | 37.5 (both workers) / 19.7 | 0.969352 (no NaN/Inf) | `E/phase2/dp2tp2/` |

**tp2cfg.** Encode 9.0 s. Mid frame: a red fox running left to right across snow, snowy
pines behind (`gen50_mid.png`); 61 frames, minimum per-frame std 67.3, mean frame-to-frame
difference 39.5 grey levels. Against the native TP2xCFG2 50-step latents (Phase 0) the cosine
is 0.584: the two stacks start from different seed-42 noise and slightly different embeds
(Phase 0), so this cross-stack number is not a parity measure. The native run's per-step median
at this shape was 1325 ms (Phase 0); AOT tp2cfg is 1573 ms (1.19x). The host decode (417 s) is
now the largest part of the generate wall; native decodes on a core in 31.4 s.

**tp4.** Encode 9.9 s. Mid frame: the same fox and setting, a near-identical composition to
tp2cfg's (`gen50_mid.png`); minimum per-frame std 68.3, mean frame-to-frame difference 41.0.
Latents cosine 0.969 vs tp2cfg after 50 steps (Phase 1's 3-step distance between these layouts
was 0.985; the 50-step trajectory accumulates more bf16 difference, informational). TP4 runs
CFG's two passes in sequence, so its step (1731 ms) is slower than tp2cfg's (1573 ms) with the
same four cores, but each core holds half the weights (5.76 vs 9.69 GiB) and the DiT loads in
144.8 s instead of 273.6 s.

**tp4sp.** Encode 9.2 s. Mid frame: the same fox and snow scene (`gen50_mid.png`); minimum
per-frame std 67.6, mean frame-to-frame difference 41.0. Cosine 0.956 vs tp2cfg (informational).
SP trims the per-core HBM from 5.76 to 5.25 GiB against plain TP4, and the step from 1731 to
1601 ms. The DiT load (22.1 s) is warm-page-cache: a first tp4sp generate had loaded the same
shards minutes earlier and was killed during its host decode when the harness's 30-minute
monitor that supervised the job expired (the whole process tree went with it, `setsid`
notwithstanding). The killed attempt is not a Difflet failure; the job was relaunched as a
plain background task, and its partial log is kept outside the repo.

**tp2cp2-ulysses.** Encode 8.6 s. Mid frame: the same fox and snow scene (`gen50_mid.png`);
minimum per-frame std 68.6, mean frame-to-frame difference 40.8. Cosine 0.964 vs tp2cfg
(informational). The DiT load took 697.0 s, against 133.6 s in Phase 1 for the same artifact;
this artifact's shards are 32 GB on disk and (inference, not measured) were no longer in the
page cache after the three cells before it, so the 1235.9 s generate wall is a cold-read figure.
The step (1783 ms) is 13% slower than tp2cfg's on the same four cores: CP2 pays two
all-to-alls per attention, CFG-parallel none.

**tp2cp2-gather_kv.** Encode 9.9 s. Mid frame: the same fox and snow scene
(`gen50_mid.png`); minimum per-frame std 68.7, mean frame-to-frame difference 41.6. Cosine 0.955
vs tp2cfg (informational). Step 1858 ms, 4% slower than ulysses (1783 ms): every CP rank gathers
the full K/V. Its weight shards are hardlinks of the ulysses artifact's (same inode, link count
3: `tp0_sharded_checkpoint.safetensors`, 8.5 GB), which the ulysses cell had just read, so its
67.2 s load is a warm-page-cache figure.

**dp2tp2.** One request, as `verify_cli`'s dp2tp2 cell sends: the CLI warns `--dp > 1 with a
single request leaves replicas idle`, the router pins `worker 0: cores 0-1` and `worker 1: cores
2-3`, both workers encode-load and load the TP2 DiT (155.4 / 156.0 s), worker 0 denoises and
decodes, and the router ends `done=1 failed=0 unfinished=0`. A replica is TP2 with CFG's two
passes in sequence, so its step (2690 ms) matches Phase 1's aot-tp2 (2719 ms at 3 steps). Each
2-core replica fits: 9.98 / 10.25 GiB per core at peak, no HBM limit. Mid frame: the same fox and
snow scene (`gen50_mid.png`); minimum per-frame std 67.7. Cosine 0.969 vs tp2cfg
(informational). Bit-identity across replicas is the separate DP check below.

#### DP bit-identity (`scripts/verify_dp_correctness.py`): PASS, 4/4 bit-identical

Command (through the plan's `run_py.sh`, `DIFFLET_BACKEND=trainium HF_HUB_OFFLINE=1`,
`TMPDIR` in the cell dir): `python scripts/verify_dp_correctness.py --model-id
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --dp 2 --tp-degree 2 --steps 3 --
--height 320 --width 512 --num-frames 61`. The script sends four requests (prompts "a red fox
in snow", "a sailboat at dusk", "a neon city street", "a bowl of ramen"; seeds 1000-1003) once
with `--dp 1` and once with `--dp 2`, then compares each request's `latents_req000N.pt`.

| run | placement | denoise step median (3 steps) | host decode per request | result |
|---|---|---|---|---|
| `--dp 1` | `[dp-router] worker 0: cores 0-1`, requests 0-3 | 2686-2696 ms | 409.8-414.7 s (one at a time) | `done=4 failed=0` |
| `--dp 2` | `worker 0: cores 0-1` (requests 0, 2), `worker 1: cores 2-3` (requests 1, 3) | 2944-3315 ms | 734.9-744.4 s (two at once on the 6-core host) | `done=4 failed=0` |

`PASS bit-identical: latents_req0000.pt` … `latents_req0003.pt`, `[verify_dp_correctness] 4/4
identical`, exit 0, 4270.7 s wall. The sha256 of each request's latents is the same in both
runs, including requests 1 and 3, which ran on the other replica's cores (cores 2-3) in the
DP run: req0 `fcf19655…`, req1 `a4b0b76e…`, req2 `ba9673c7…`, req3 `23c73e85…` (full list in
`E/phase2/dpcheck/sha256.txt`). Per-core HBM peak 10.25 GiB on all four cores; no 2-core
replica ran out of HBM, so this is not a LIMIT cell. Peak host RSS: 41.7 GiB for the two
concurrent `aot_vae` processes.

Timing notes. The first request's encode took 902.0 s: the Qwen2.5-VL weights had left the page
cache and the root EBS volume was reading at about 15-17 MB/s at the time (measured from
`/proc/diskstats` while the encode process sat in uninterruptible disk sleep), against the
seconds it takes warm (7.4 s for requests 1-3). That is a host I/O condition, not a Difflet
issue. In the DP run the two replicas' steps are 10-20% slower than one replica alone (2944-3315
vs 2686-2696 ms); with 3 steps per request this is a handful of samples, and both replicas'
host-side work (scheduler, the per-step host refiner calls of serial CFG) shares 6 physical
cores (inference, not profiled). The decodes of the two replicas overlap on the same CPU, so
DP does not shorten the host-decode part of the wall time.

#### Phase 2 summary

- All six layouts and the DP check **PASS** at 50 steps: every generate exits 0, no latents file
  has NaN or Inf, and every mid frame shows the fox running through snow. The latents cosines
  against tp2cfg after 50 steps are 0.955-0.969 (informational; the per-forward gate was Phase 1's).
- Step median (ms): tp2cfg 1573 · tp4sp 1601 · tp4 1731 · ulysses 1783 · gather_kv 1858 ·
  dp2tp2 replica 2690 (serial CFG on 2 cores). Native TP2xCFG2 at the same shape: 1325 ms
  (Phase 0, other venv and toolchain); the fastest AOT layout is 1.19x slower per step.
- The host decode costs 409-423 s for 61 frames at 320x512 in every cell (fp32, tiled, 6
  physical cores at ~580% CPU, 17-19 GiB RSS). It is now the largest part of each generate wall
  (native decodes on a core in 31.4 s). At the default 480x848x121 the decode has about 5x the
  pixels and frames (Phase 3 below).
- Load times swing with the page cache and with the root volume's read rate (cold ulysses 697 s,
  warm tp4sp 22 s); the generate walls above carry that noise. During the DP check the root EBS
  volume was serving reads at about 15-17 MB/s.
- Process note: the first tp4sp attempt was killed when the harness monitor that supervised the
  job expired; the remaining cells ran as a plain background task.

### Phase 3: default shape 480x848x121, 50 steps

#### tp4 + SP at the default shape: rejected before compile in 6.4 s

`difflet compile --height 480 --width 848 --num-frames 121 --tp-degree 4 --sp` exits 1 after
6.4 s wall, at the configuration check of the aot_transformer stage, before any HLO is generated:

```
ValueError: HunyuanVideo 1.5 sequence parallelism splits the N=49290 image tokens of 480x848x121 over tp_degree=4, but N % 4 = 2; use num_frames in [117, 125] at this resolution, or tp_degree in [2].
```

N = 31 latent frames x 30 x 53 = 49290, and 49290 mod 4 = 2, as the message says.
`difflet generate` with the same flags behaves as the ring case did in Phase 1: the host encode
runs first (10.0 s here), and the aot_transformer stage then stops with `[hv15] no valid compiled
transformer artifact at …/hunyuan_video_15_aot_dit/275d3f909e43f791 … run \`difflet compile\`
with the same flags first.` (32.5 s wall). `difflet compile` then names the constraint in seconds.
`b7b939b` moved the ring and multi-shape refusals in front of the encode; this SP shape check
lives in the DiT config and still runs inside the transformer stage. Not counted as a bug (generate
needs a compiled artifact, and compile fails fast with guidance). The follow-up option is a
pre-encode check in `_run_aot` for generate: that the DiT manifest exists, or the token
divisibility itself. Either one avoids an encode that can take minutes when the page cache is
cold. Evidence: `E/phase3/tp4sp-reject/`.

#### AOT TP2 x CFG2 at 480x848x121: BLOCKED (neuronx-cc internal error NCC_IRPX901)

`difflet compile --height 480 --width 848 --num-frames 121 --tp-degree 2 --cfg-parallel` exits
1 after **1807.7 s** wall. No NEFF is written and there is nothing to generate from.

| phase | time | observation |
|---|---|---|
| module load | 37.9 s | `Finished loading module HunyuanVideo15Transformer3DModel` |
| HLO generation (torch-xla trace of the whole DiT, one process at 100% CPU) | **1729.8 s** | input example `[2, 65, 31, 30, 53]`; stage RSS rose steadily to a 31.9 GiB peak |
| neuronx-cc 2.26.6360 (`-O1 --model-type=transformer`) | ~30 s | fails in the Tensorizer, pass `RelaxPredicates`, right after `LegalizeSundaAccess` (`instance_count_change=+136991`) |

```
…/HunyuanVideo15Transformer3DModel[.1]/aten.sum.dim_IntList/aten__sum_reduce.223 [INTERNAL_ERROR] [NCC_IRPX901] RelaxPredicates assertion error: inst should be valid after relaxing predicates - Please open a support ticket at https://github.com/aws-neuron/aws-neuron-sdk/issues/new.
```

**Which op.** Decoding the HLO module (`libneuronxla.proto.hlo_pb2`) shows that `reduce.223` is
an f32 sum over dim 1 of `convert.216`, an f32 `[1, 1985]` tensor, giving `[1]`. 1985 = 1000
(Qwen) + 256 (ByT5) + 729 (image) text slots. That is the fp32 valid-text count in
`backbone15.py:704`, `text_valid = encoder_attention_mask.float().sum(1).to(torch.int32)`, which
feeds the bounded attention kernel's bounds in every block. Phase 1 and Phase 2 compiled the
same reduction at 320x512x61 (10240 image tokens) in every layout. Only the surrounding graph
changes at 49290 image tokens, so the failure depends on context. The op's own shape is the
same at both sizes.

**Toolchain and flags (CPU-only re-runs of the saved HLO, stopped after the Tensorizer):**

| compiler | flags | result |
|---|---|---|
| neuronx-cc 2.26.6360 (AOT venv) | as Difflet (`-O1`) | NCC_IRPX901 on `aten__sum_reduce.223` |
| neuronx-cc 2.26.6360 | `-O2` | same NCC_IRPX901, same op |
| neuronx-cc 2.28.7631 (native-312 venv) | as Difflet (`-O1`) | same NCC_IRPX901, same op |

So a toolchain bump to 2.28 does not lift it, and neither does the optimisation level. This is
a vendor compiler internal error, not a Difflet logic bug. It would need a Difflet-side
workaround in the traced graph (see the follow-up options below).

**Follow-up options (not attempted; each changes the DiT graph at every shape):**
1. Feed `text_valid` as a precomputed int32 input. It is the number of valid entries across
   the three text masks; `text_order` only permutes them. It would be computed on the host (or
   in the encode stage), and the in-graph reduction disappears.
2. Compute the count without a reduce op, e.g. `mask.float() @ ones(1985, 1)`, keeping the
   fp32-then-int32 rule.

Both change the traced graph at every shape, so with the current cache key (no code version)
every Phase 1/2 artifact would have to be invalidated by hand and re-gated. Each default-shape
attempt costs at least 30 min of HLO generation before neuronx-cc starts. Past the Tensorizer,
the backend's time on 51k-token bounded attention is unknown: spike (c) ran a single such call
for more than 45 min without finishing. Evidence: `E/phase3/tp2cfg/compile.log`,
`ncc226_O1_log_tail.log`, `ncc226_O2_log_tail.log`, `ncc228_O1_log_tail.log`,
`ncc_command.log`.

#### Native TP2 x CFG2 at 480x848x121, same host (reference): PASS

Native backend, `/home/ubuntu/venvs/native-312` (torch 2.14, neuronx-cc 2.28), same wrapper and
flags as Phase 0, `--height 480 --width 848 --num-frames 121 --steps 50 --tp-degree 2
--cfg-parallel`, run on this host right after the AOT attempt.

| step | wall | stage timings (from the log) | HBM GiB/core | host RSS peak | evidence |
|---|---|---|---|---|---|
| `compile` (cold) | **2807.1 s** | transformer load 1732.5 s (load checkpoint 126.6 s, warmup forward 1591.4 s); vae load 37.5 s | 18.77 x4 | 78.3 GiB (4 transformer ranks) | `E/phase3/native-tp2cfg/compile.log` |
| `generate --steps 50` | **960.7 s** | encode 8.4 s; transformer load 43.6 s; denoise 754.9 s, **median 11178 ms** (first 195214 ms); device decode 89.0 s | 18.48-18.52 | 70.9 GiB (transformer ranks) | `gen50.log`, `gen50.mp4`, `gen50_mid.png` |

The 121-frame video's mid frame shows a red fox leaping through deep snow among snow-laden pines
(`gen50_mid.png`); minimum per-frame std 63.1, mean frame-to-frame difference 25.3. The
documented native figures (`neuron-backend-results.md`, other run: compile 2837 s, warm e2e
934.7 s, 12625 ms per step) agree within 1% for the compile and 3% for the e2e; the step
median here is 11% lower. The first step (195.2 s) carries per-process one-time work that the
compile run's warmup forward already did once (inference: the bounded kernel's first launch in
this process). The AOT backend has no 480x848x121 DiT to compare against (above).

#### AOT host VAE at 480x848x121 (measured on the native run's latents): 2518.7 s, 28.0 GiB RSS

The AOT DiT gives no default-shape latents, so the AOT `aot_vae` stage's host decode was measured
on the native run's 50-step latents (`[1, 32, 31, 30, 53]` latent grid, copied into a fresh work
dir). Invocation: `python -m difflet.cli.stage --orchestrator hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
--stage aot_vae --stage-mode generate --height 480 --width 848 --num-frames 121 --tp-degree 2
--cfg-parallel --work-dir <dir> --output decode.mp4` with `DIFFLET_BACKEND=trainium`. This is
the code path `difflet generate` runs after the DiT, on the host CPU only; no other job ran
meanwhile.

| | value |
|---|---|
| host decode (`[hv15] request 0: host decode …`) | **2518.7 s** (42 min), stage wall 2535.1 s |
| peak host RSS of the stage | **27.98 GiB** |
| output | 121 frames, 480x848, minimum per-frame std 63.0, mean frame-to-frame difference 25.3 (native's device decode of the same latents: 63.1 / 25.3) |
| mid frame | the same leaping fox as native's (`decode_mid.png`) |

Against native's on-device decode of the same latents (89.0 s), the host decode is 28x slower,
and at this shape it would cost about 3.3x the whole native denoise (754.9 s). It grows about
6x from 320x512x61 (409-423 s) to 480x848x121, in line with the about 4.9x larger pixel x frame
volume. Evidence: `E/phase3/hostvae/`.

#### Phase 3 summary

| item | outcome | numbers |
|---|---|---|
| AOT TP2xCFG2 compile at 480x848x121 | **BLOCKED** (neuronx-cc NCC_IRPX901 on the fp32 valid-text count reduce; same on 2.28 and at -O2) | 1807.7 s wall to failure; HLO generation 1729.8 s; peak RSS 31.9 GiB |
| AOT TP2xCFG2 generate at 480x848x121 | not run (no artifact) | — |
| AOT host VAE decode at 480x848x121 | measured on native latents | 2518.7 s, 27.98 GiB peak RSS |
| tp4 + SP at 480x848x121 | **PASS** (rejected before compile) | 6.4 s, `N % 4 = 2; use num_frames in [117, 125] … or tp_degree in [2]` |
| native TP2xCFG2 at 480x848x121 (same host) | PASS | compile 2807.1 s, generate 960.7 s, step median 11178 ms, decode 89.0 s |

## Campaign (Task 10), Phase 3 retry and re-gate on the 10-input DiT graph: 2026-10-31

HEAD at start `93dde78`. Three commits since Phase 3 change the AOT DiT:
- `1a9625b`: the valid-text count `text_valid` (int32 `[B]`) is computed on the host and passed
  as DiT graph input #10. No mask reduction is left in the graph.
- `a41ecc9`: the AOT stage cache key carries `graph_version` (now 2). `difflet compile`
  therefore writes new artifact dirs; the Phase 1-2 dirs (9-input graph) are left in place and
  are not reused.
- `93dde78`: `_run_aot` checks the SP/CP shape rules before the encode stage.

**Graph labels.** In this section, *old graph* means the 9-input graph with the in-graph
count (Phases 1-3 above). *New graph* means the 10-input graph (`graph_version` 2). Every
number in this section is from the new graph unless it is marked otherwise.

Same runner, wrappers, venvs and common flags as Phases 2-3. Curated evidence:
`E/phase3-retry/`, `E/phase1-regate/`, `E/phase2-regate/`.

### Phase 3 retry: AOT TP2 x CFG2 at 480x848x121

#### Attempt 1 (new graph, Difflet's default compiler flags): BLOCKED, NCC_IRPX901 again, on a different op

`difflet compile --height 480 --width 848 --num-frames 121 --tp-degree 2 --cfg-parallel` exits
1 after **1788.9 s** wall.

| phase | time | observation |
|---|---|---|
| module load | ~38 s | |
| HLO generation | **1707.1 s** (all HLOs 1745.1 s) | input example `[2, 65, 31, 30, 53]`; peak stage RSS 31.7 GiB (total 32.7 GiB) |
| neuronx-cc 2.26.6360 (`--model-type=transformer -O1 --tensorizer-options=--enable-ccop-compute-overlap --auto-cast=none --internal-hlo2tensorizer-options=--verify-hlo=true`) | Tensorizer 27.5 s | fails in pass `RelaxPredicates` on subgraph sg0000, exit code 70 |

```
…/HunyuanVideo15Transformer3DModel[.1]/HunyuanVideo15TransformerBlock[.1]/AdaLayerNormZero[.2]/LayerNorm[.5]/aten.view.default/aten__native_batch_norm_batch-norm-training.575 [INTERNAL_ERROR] [NCC_IRPX901] RelaxPredicates assertion error: inst should be valid after relaxing predicates
```

**Which op.** In the decoded HLO module, `batch-norm-training.575` takes `gather.574`, a bf16
`[1, 1985, 2048]` tensor, from `torch/nn/modules/normalization.py:229`. That is the text-stream
LayerNorm (`norm1_context`) of the first block, applied to the output of the static
`text_order` gather in `backbone15.py`. The old graph's failing op, the count reduce, also read
the gathered text mask. With the count gone, the compiler trips on the next reduction over the
gathered text stream. The op's shape is the same at 320x512x61, which compiles. So this is the
same vendor compiler internal error, in the same pass, and it depends on the surrounding graph.
It is not a new Difflet logic bug.

**Compiler-flag probes** (CPU only: the saved HLO re-run through neuronx-cc 2.26.6360 until the
Tensorizer finishes or fails, ~30 s each):

| flags | sg0000 RelaxPredicates | result |
|---|---|---|
| Difflet's flags, without the `--verify-hlo` internal option (control) | assertion | **same NCC_IRPX901, same op** |
| `--model-type=transformer -O1`, **no `--enable-ccop-compute-overlap`** | `Finished (changed=True)` | passes RelaxPredicates; still in BirCodeGenLoop when the probe stopped it |
| `--model-type=generic -O1 --tensorizer-options=--enable-ccop-compute-overlap` | `Finished (changed=True)` | `Tensorizer finished after 28.9 s`, no error |

So the assertion appears only with the transformer model type *and* compute/collective overlap
on. Difflet's hv15 backbone already reads the compiler flags from the
`DIFFLET_HUNYUAN15_COMPILER_ARGS` environment override (`backbone15.py`
`NeuronHunyuanVideo15BackboneApplication.get_compiler_args`). Attempt 2 below uses that
override with the overlap option removed, so no code changes. The override is not in the AOT
cache key; at this shape there was no prior artifact, so nothing stale can be reused.
Evidence: `E/phase3-retry/tp2cfg-attempt1/` (`compile.log`, `ncc_log_excerpt.log`,
`ncc_command.log`, `probe-*_log_excerpt.log`).

#### Attempt 2 (new graph, compute/collective overlap off): PASS, compiles and generates the video

Same command, plus `DIFFLET_HUNYUAN15_COMPILER_ARGS="--model-type=transformer -O1 --auto-cast=none
--internal-hlo2tensorizer-options='--verify-hlo=true'"`. These are Difflet's hv15 DiT flags with
only `--tensorizer-options=--enable-ccop-compute-overlap` removed. Then a 50-step `difflet
generate` without any override; the artifact is already compiled.

| step | outcome | wall | breakdown (from the log) | HBM GiB/core | host RSS peak |
|---|---|---|---|---|---|
| `compile` | **PASS** | **3970.0 s** | module load 37.9 s; **HLO generation 1706.3 s**; neuronx-cc (Tensorizer passes in 17.8 s) 2078.8 s; model built in 3952.2 s, into `hunyuan_video_15_aot_dit/5161ab1fe7b07b80` (32 GB) | n/a | 31.5 GiB stage process; **116.5 GiB** total with the neuronx-cc processes (host has 124 GiB) |
| `generate --steps 50` | **PASS** | **4132.8 s** e2e | encode load 47.7 s, encode 103.1 s; DiT load 574.4 s (cold); denoise 661.9 s, **median 13072 ms** (first 21178 ms); **host VAE decode 2699.2 s** | **23.58** x4 | aot_transformer stage 102.2 GiB; aot_vae 34.8 GiB; encode 16.5 GiB |

- Output: 121 frames at 480x848, minimum per-frame std 44.4, mean frame-to-frame difference
  24.6. The mid frame (`gen50_mid.png`) shows a red fox running through snow in a birch forest.
  `gen50.mp4` (0.8 MB) is in the evidence dir.
- No NaN or Inf in the latents (`latents.pt` sha `35c65be5…`). The cosine against the native
  50-step latents at this shape is 0.234. As at 320x512 (0.584), the two stacks start from
  different seed-42 noise and slightly different embeds, so this is not a parity measure. Parity
  is gated per forward at 320x512x61 (re-gate below).
- **Against native, same host, same layout (Phase 3 reference):**
  - Step median: AOT 13072 ms vs native 11178 ms, so the AOT step is **1.17x slower**.
  - Denoise total: AOT 661.9 s vs native 754.9 s. AOT is lower because native's first step
    takes 195.2 s (per-process one-time work) and AOT's takes 21.2 s.
  - Compile: AOT 3970 s vs native 2807 s.
  - e2e: AOT 4132.8 s vs native 960.7 s. The difference is the host VAE (2699.2 s vs native's
    device decode of 89.0 s) and the cold DiT load (574.4 s vs 43.6 s). The host decode agrees
    with the 2518.7 s measured on native's latents in Phase 3 (within 7%).
  - Without the VAE and load (encode + denoise): AOT 765 s, native 763 s.
- The VAE stays host-decoded (user decision). At this shape it is **LIMIT**: 2699 s, 65% of the
  e2e wall.
- **Core claim with the pre-encode shape check (`93dde78`): PASS.** `_run_aot` now imports
  torch_neuronx / neuronx_distributed in the CLI driver before it launches the all-core
  aot_transformer stage. This generate is the first one on the new code. The stage loaded on
  all four cores (23.58 GiB on each), and there was no "cores busy" or NRT allocation error
  anywhere in the log.
- **Host memory.** Peaks: the compile reached 116.5 GiB of the host's 124 GiB, and the DiT load
  peaked at 102.2 GiB RSS in the aot_transformer stage. Native's ranks peaked at 70.9 GiB.
  Neither run failed, but the default-shape compile has little headroom on this host. The RSS
  sampler also recorded small (about 2.3 GiB) processes keyed `hang`/`toy`/`transformer`/
  `generate`. These are not part of this run; they come from another session's processes whose
  command line contains `difflet`.
- **Follow-up (not a code change in this campaign).** Difflet's default hv15 DiT flags fail at
  480x848x121 (NCC_IRPX901, attempt 1); the same flags without
  `--enable-ccop-compute-overlap` compile and run. Making that the default at this shape (or
  for all shapes) needs a decision and a perf check at 320x512x61, where the overlap flag is
  what Phases 1-2 measured. The override env var is not in the AOT cache key either.
  Evidence: `E/phase3-retry/tp2cfg/` (`compile.log`, `gen50.log`, `*.meta.json`,
  `gen50.mp4`, `gen50_mid.png`, `cmp_gen50__vs__p3-native-tp2cfg.json`, `sha256.txt`).

### Phase 1 re-gate on the new graph (320x512x61, seed 42, default compiler flags)

**Method (ruling R16).**
- Each layout is compiled fresh with `difflet compile`. The new `graph_version` gives every
  layout a new artifact dir, and the default hv15 compiler flags apply, overlap included.
- The compiled DiT is then run through `scripts/hv15_aot_denoise_from_inputs.py` on the
  **native run's `embeds.pt` and native noise** (`native_noise.pt`, the same inputs as Phase 1's
  aot-tp2 native comparison), for 1 step and for 3 steps.
- **Gate 1:** at 1 step, cosine ≥ 0.999 against native TP2xCFG2's 1-step latents.
- **Gate 2:** at 3 steps, distance (1 − cosine) to diffusers' CPU bf16 3-step reference
  ≤ 1.15x native's own distance. Native's distance is 1 − 0.983536 = 0.016464, so the bound is
  0.018934, i.e. cosine ≥ 0.981066.
- In Phase 1 only aot-tp2 was compared with native. Here every layout is.

| cell (new artifact) | compile s (DiT / wall) | 1 step vs native: cos / rel-L1 / max-abs | 1 step vs diffusers cos | 3 steps vs diffusers: cos (distance ratio vs native) | 3 steps vs native cos | NaN/Inf | gate | evidence |
|---|---|---|---|---|---|---|---|---|
| tp2 (`78b5bd73…`) | 516.0 / 530.1 (HLO 103.3) | **0.999160** / 0.0434 / 0.56 | 0.999609 | **0.985041** (0.91x) | 0.982189 | none | **PASS** | `E/phase1-regate/tp2/` |
| tp2cfg (`d3d575ba…`) | 448.1 / 462.8 (HLO 101.8) | **0.999124** / 0.0448 / 0.56 | 0.999566 | **0.986107** (0.84x) | 0.978131 | none | **PASS** | `E/phase1-regate/tp2cfg/` |
| tp4 (`b2122b91…`) | 301.4 / 315.3 (HLO 50.2) | **0.999131** / 0.0445 / 0.53 | 0.999573 | **0.987437** (0.76x) | 0.984438 | none | **PASS** | `E/phase1-regate/tp4/` |
| tp4sp (`eaea3dc3…`) | 317.3 / 330.3 (HLO 55.5) | **0.999142** / 0.0439 / 0.53 | 0.999578 | **0.990780** (0.56x) | 0.982686 | none | **PASS** | `E/phase1-regate/tp4sp/` |
| tp2cp2-ulysses (`--cp-mode ulysses`, `2cb886e3…`) | 327.2 / 341.7 (HLO 60.9) | **0.999132** / 0.0443 / 0.53 | 0.999569 | **0.991254** (0.53x) | 0.981850 | none | **PASS** | `E/phase1-regate/tp2cp2-ulysses/` |
| tp2cp2-gather_kv (`--cp-mode gather_kv`, `698cde4d…`) | 337.6 / 351.4 (HLO 57.9) | **0.999149** / 0.0435 / 0.56 | 0.999602 | **0.979483 (1.25x)** | 0.984527 | none | gate 1 PASS, **gate 2 FAIL** (see below) | `E/phase1-regate/tp2cp2-gather_kv/` |

Reference rows (from Phase 1, same inputs): native TP2xCFG2 vs diffusers, 1 step 0.999136, 3 steps
0.983536 (distance 0.016464).

- **The graph change is numerically neutral.** On the same native inputs, the new-graph tp2
  latents are **bit-identical** to Phase 1's old-graph aot-tp2 latents at 1 step and at 3 steps
  (`cmp_native_inputs{1,3}__vs__aot-tp2_native_inputs{1,3}.json`). Phase 2 tp2cfg below shows
  the same at 50 steps.
- **Gate 1 (1 step, ≥ 0.999 vs native): all six layouts PASS**, at 0.999124-0.999160. Every
  layout is also closer to diffusers per forward (0.999566-0.999609) than native is (0.999136).
- **Gate 2 (3 steps, ≤ 1.15x native's distance to diffusers): five PASS, gather_kv misses.**
  - tp2 0.91x, tp2cfg 0.84x, tp4 0.76x, tp4sp 0.56x, ulysses 0.53x.
  - **gather_kv: 1.25x** (cos 0.979483 vs diffusers; rel-L1 0.1838 vs native's 0.1723,
    i.e. 1.07x).
  - The miss is deterministic. A second 3-step run of the same artifact is bit-identical
    (`native_inputs3b`, `cmp_native_inputs3b__vs__…native_inputs3.json`).
- **Context for the gather_kv miss** (no conclusion drawn here; the controller should rule):
  - Per forward, gather_kv is the layout closest to diffusers (0.999602).
  - At 3 steps it is closer to native (0.984527) than tp2 is (0.982189).
  - The 3-step spread between AOT layouts on identical inputs is itself 0.9765-0.9870:
    gather_kv vs tp2 0.983062, gather_kv vs ulysses 0.976528, ulysses vs tp2 0.985139, tp4 vs
    tp2 0.987026.
  - So the 3-step metric moves by about 1-2% of cosine between layouts that agree to
    0.9991+ per forward. gather_kv's distance sits at the edge of that spread, not outside it.
  - Phase 1 had no gather_kv-vs-diffusers measurement (it was compared with aot-tp2 on AOT
    inputs, 3 steps 0.98791), so there is no old-graph number to compare with.
  - Phase 2's 50-step gather_kv video (old graph) showed the prompt correctly.
- HBM and load times are as in Phase 1. The helper's 1-step loads after a fresh compile are
  page-cache cold (159-279 s); the 3-step loads right after are warm (25-38 s).

### Phase 2 re-run: tp2cfg, 50 steps, 320x512x61, new graph

| step | outcome | wall | detail | HBM GiB/core | host RSS |
|---|---|---|---|---|---|
| `compile --tp-degree 2 --cfg-parallel` | PASS (manifest hit on the re-gate's `d3d575ba…`) | 12.9 s | DiT 0.0 s | n/a | — |
| `generate --steps 50` | **PASS** | 902.0 s | encode 102.8 s (cold), DiT load 213.2 s, denoise 80.5 s, **median 1577 ms** (first 3076 ms), host decode 430.8 s | 9.69 x4 | DiT 33.3 / VAE 18.5 GiB |

- The 50-step latents are **bit-identical** to Phase 2's old-graph tp2cfg latents: same sha256
  `73685bc9…`, cosine 1.000000.
- The step median is 1577 ms against 1573 ms on the old graph, so moving the count to the
  host costs nothing measurable per step.
- The MP4 shows the same fox as Phase 2; the mid frame has min per-frame std 67.3 and mean
  frame-to-frame difference 39.5, identical to the old run.
- Evidence: `E/phase2-regate/tp2cfg/` (`gen50.mp4`, `gen50_mid.png`, logs, meta,
  `cmp_gen50__vs__p2-tp2cfg.json`).

### Summary of the retry and re-gate

| item | graph | outcome |
|---|---|---|
| AOT TP2xCFG2 compile at 480x848x121, default flags | new | **BLOCKED**: NCC_IRPX901 on the text-stream LayerNorm after the `text_order` gather (1788.9 s to failure) |
| AOT TP2xCFG2 at 480x848x121, `--enable-ccop-compute-overlap` removed via `DIFFLET_HUNYUAN15_COMPILER_ARGS` | new | **PASS**: compile 3970 s (HLO 1706 s, neuronx-cc 2079 s); generate e2e 4132.8 s; step median 13072 ms (native 11178 ms, 1.17x); host VAE 2699.2 s (LIMIT); MP4 shows the prompt |
| Phase 1 gate 1 (1 step vs native ≥ 0.999), six layouts | new | **PASS** x6 (0.999124-0.999160) |
| Phase 1 gate 2 (3 steps vs diffusers ≤ 1.15x native) | new | PASS x5 (0.53-0.91x); **gather_kv 1.25x, deterministic, needs a ruling** |
| Phase 2 tp2cfg 50 steps | new | **PASS**; bit-identical to the old graph; 1577 ms/step |
| Pre-encode shape check (`93dde78`) and the all-core stage | new | **PASS** (stage claimed 4 cores, no NRT errors) |

The Phase 1-2 numbers in the earlier sections stay as they are: they are from the old 9-input
graph. The old artifact dirs and the `039f757f…stale-pre-fd2a8aa` dir are still on disk.

## Campaign (Task 10), Phase 4 and Phase 5 on the no-ccop default: 2026-10-31

HEAD at start `dca10f3`. That commit makes the hv15 AOT DiT compile **without**
`--tensorizer-options=--enable-ccop-compute-overlap` by default (the option caused NCC_IRPX901 at
480x848x121) and puts the effective compiler args into the AOT stage cache key. Every artifact
in this section is therefore freshly compiled; the earlier dirs are not reused.
`DIFFLET_HUNYUAN15_COMPILER_ARGS` is not set anywhere in this section.

Same runner (`runcell.py`: plan wrapper, neuron-monitor 1 s, host-RSS sampler), wrappers, venv
(`difflet-neuron`) and common flags as Phases 2-3: 320x512x61, `--seed 42`, `--prompt "a
cinematic shot of a red fox running through a snowy forest"`, `HF_HUB_OFFLINE=1`, one device
job at a time. Curated evidence: `E/phase4/`, `E/phase5/`.

### No-ccop perf check (R23): TP2xCFG2, 50 steps: **13% slower per step than with the overlap option**

| step | outcome | wall | detail | HBM GiB/core | host RSS |
|---|---|---|---|---|---|
| `compile --tp-degree 2 --cfg-parallel` | PASS, new dir `f464ca8f…` (manifest `compiler_args`: `--model-type=transformer -O1 --auto-cast=none --internal-hlo2tensorizer-options='--verify-hlo=true'`) | 450.7 s | DiT 438.3 s (old graph-v2 tp2cfg with overlap: 448.1 s) | n/a | 32.8 GiB |
| `generate --steps 50` | PASS | 719.0 s | encode 103.2 s (cold), DiT load 31.9 s, denoise 125.3 s, **median 1778 ms** (first 38030 ms, cold), host decode 412.3 s | 9.75 x4 | DiT 38.1 / VAE 18.5 GiB |

- **Latents vs the previous tp2cfg 50-step run** (`p2r-tp2cfg`, same graph, overlap on, same
  `embeds.pt` sha `3051691e…`, same seed): **cos 0.954898**, rel-L1 0.278, max-abs 4.44, no
  NaN/Inf. Not bit-identical, as expected from a different compile; the value sits in the
  50-step cross-layout band measured in Phase 2 (0.955-0.969). The MP4 shows the fox running
  through snow (`E/phase4/tp2cfg/gen50_mid.png`).
- **Same-session A/B, to separate the flag from host drift.** The DiT alone
  (`scripts/hv15_aot_denoise_from_inputs.py`, same `embeds.pt`, seed 42, 50 steps), alternating
  the old overlap artifact `d3d575ba…` and the new `f464ca8f…`, each loaded warm:

  | run | artifact | denoise s | step median ms | first step ms | latents |
  |---|---|---|---|---|---|
  | ov1 | overlap on (`d3d575ba…`) | 81.4 | 1606 | 2720 | **bit-identical** to the Phase 2 re-run CLI latents |
  | no1 | overlap off (`f464ca8f…`) | 92.1 | 1817 | 2980 | **bit-identical** to this section's CLI `gen50` latents |
  | ov2 | overlap on | 80.9 | 1598 | 2522 | bit-identical to ov1 |
  | no2 | overlap off | 92.0 | 1816 | 2931 | bit-identical to no1 |

  **Dropping the overlap option costs +13.4% per step at 320x512x61** (1816.5 vs 1602 ms,
  medians of the two runs each; +13.5% on the whole denoise). The step times are stable to
  ±0.3% across runs. This is above the 5% bar of R23: **the flag should be made shape-aware**
  (keep the overlap where it compiles). Evidence: `E/phase4/noccop-ab/`.
- The CLI's 1778 ms vs the helper's 1817 ms: same artifact and latents; the CLI run is a single
  sample and the helper times include its own host loop. The A/B uses one harness for both arms.

### Phase 4a: TeaCache fixed cadence, TP2xCFG2, 50 steps

Two explicit `generate` commands on the `f464ca8f…` artifact, identical apart from
`--teacache-cadence 2`; the no-TeaCache run is the R23 run above.

| run | `[teacache] stats` | denoise s (first step) | denoise minus first step | step median ms | latents vs no-TeaCache | mid frame |
|---|---|---|---|---|---|---|
| no TeaCache (`gen50`) | — | 125.3 (38.0 cold) | 87.3 s | 1778 | ref | fox in snow |
| `--teacache-cadence 2` (`cad2`) | `full_steps 30, skipped_steps 20, probe_calls 0` | 54.6 (3.2) | **51.4 s (-41%)** | 1768 (full steps) | **cos 0.938644**, rel-L1 0.346, max-abs 5.32, no NaN/Inf | same fox, same pose/scene/framing; leg and snow-spray detail differ |

- **skipped_steps 20 of 50: PASS.** With CFG parallel both CFG halves live in one batched DiT
  call, so a skipped step skips both (one `_dit` call per step; the 30 full steps are 30 calls).
- **Denoise time saved: 41%** (warm-equivalent; the no-TeaCache run's first step was a cold
  38 s, so first steps are excluded on both sides; the helper's warm no-TeaCache denoise is
  92 s, against which cadence 2 saves 44%). End to end the gain is small because the host VAE
  (410 s) dominates (`generate` wall 719.0 → 529.5 s, but 94 s of that is the cold encode).
- **Latents cos ≥ 0.99: NOT MET (0.9386).** Triage, no Difflet bug:
  - The output is right: the MP4 shows the same video (`E/phase4/tp2cfg/{gen50,cad2}_mid.png`).
  - The 0.99 bar is above this model's 50-step floor. A **pure compiler-flag change** (above:
    no computational change, ULP-level differences per step) already moves the 50-step latents
    to **cos 0.9549**; the Phase 2 cross-layout spread was 0.955-0.969. Skipping 20 of 50 steps
    is a much larger perturbation than that, yet its distance (1 − cos = 0.061) is only 1.36x the
    recompile's (0.045). cad2 vs the old overlap run: 0.9353.
  - HV1.0's 0.99984 at cadence 2 (`benchmark/trn2/hunyuan_video_neuron.md`) is not comparable:
    HV1.0 is guidance-distilled (one DiT pass/step) and its trajectory is far flatter.
  - Ruling needed: accept on skipped steps + visual + time, or set an hv15 cadence gate from the
    floor (e.g. distance ≤ 2x the same-layout recompile distance).
- `teacache_cadence_ab.sh` (TP4 serial CFG) **not run**: it sources `<repo>/.venv`, which does
  not exist here (R5: AOT uses `difflet-neuron`), and its prompt differs from the campaign's. The
  explicit TP2xCFG2 pair above covers the same check. Optional per the brief.

### Phase 4b: TeaCache online-delta, TP2xCFG2, 50 steps

`--teacache-online-delta 0.6`. `difflet/pipeline/teacache.py` has no default alpha (the mode is
off at 0.0); 0.6 is the value the CLI spec and plans use (`docs/plans/2026-06-23-cli.md`).

| run | `[teacache] stats` | denoise s (first step) | denoise minus first step | latents vs no-TeaCache | mid frame |
|---|---|---|---|---|---|
| `--teacache-online-delta 0.6` (`od06`) | `full_steps 33, skipped_steps 17, probe_calls 0, last_delta_estimate 0.1426` | 60.3 (3.2) | **57.1 s (-35%)** | **cos 0.918296**, rel-L1 0.373, max-abs 6.44, no NaN/Inf | same fox and scene |

- 17 skipped. The mode never skips two steps in a row, so within the 40-step window between
  warmup and cooldown at most 20 can be skipped (cadence 2 skips exactly those 20). It skips
  fewer steps than cadence 2 but drifts more (0.918 vs 0.939); which steps it skipped is not
  logged, so the reason is not established here.
- Same floor argument as 4a; the video is right (`E/phase4/tp2cfg/od06_mid.png`).

Reproduce (each one command; `R` = the plan wrapper `run_difflet.sh trainium`):
```
R compile  --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --height 320 --width 512 --num-frames 61 --tp-degree 2 --cfg-parallel
R generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --height 320 --width 512 --num-frames 61 --tp-degree 2 --cfg-parallel \
  --steps 50 --seed 42 --prompt "a cinematic shot of a red fox running through a snowy forest" --output gen50.mp4 [--teacache-cadence 2 | --teacache-online-delta 0.6]
# A/B: scripts/hv15_aot_denoise_from_inputs.py --compiled-dir <dir> --embeds <gen50 embeds.pt> --tp-degree 2 --cfg-parallel \
#   --height 320 --width 512 --num-frames 61 --steps 50 --seed 42 --out x.pt   (NEURON_RT_VIRTUAL_CORE_SIZE=2)
```

### Phase 4c: adaptive TeaCache (calibration + `--teacache-speedup`)

**Code under test moved during this cell (recorded so the artifact ids make sense).**
- The calibration was collected on HEAD `67d3d92` (no-ccop default).
- Before the adaptive generate, the controller landed `420d2fa` + `d001314` (R25: the ccop
  overlap is kept for N ≤ 16384 image tokens, so 320x512x61 = 10240 tokens compiles **with** it
  again). So the adaptive generate, the reset check and Phase 5 run on a fresh overlap artifact,
  `69162f72…` (manifest `compiler_args` include `--enable-ccop-compute-overlap`).
- Its no-TeaCache 50-step DiT output (`base`, `hv15_aot_denoise_from_inputs.py`, same `embeds.pt`
  and seed) is **bit-identical** to the Phase 2 re-run's overlap tp2cfg latents (`p2r`). That
  makes it the reference for this cell. Step median 1607 ms, the same as the R23 A/B's overlap
  arm (1598-1606).

**Bug found and fixed first: `67d3d92`.** The first collection attempt died before step 0 in
`teacache_probe15.reset_state`:
`AttributeError: 'RecursiveScriptModule' object has no attribute 'states'`
(`E/phase4/calibration/collect_prefix-67d3d92_FAILED.log`). The CLI's adaptive path calls the same
reset once per request, so **every adaptive hv15 AOT request failed**. Root cause, fix and device check
are in the bug ledger below.

**Calibration (Task 7b collector).** TP4 serial CFG (the collector runs `cfg=1`, TP only), so a
TP4 DiT + probe was compiled first. It used `difflet compile --tp-degree 4 --teacache-speedup 1.5
--teacache-calibration <the output path>`, which builds the probe with the DiT (344.5 s,
`9667b250…`). The collection itself:
```
scripts/calibrate_teacache.py --model hunyuan_video_15 --allow-hardware --foreground-ack "I am running TeaCache T0 in the foreground" \
  --source-dir <HF snapshot root> --compiled-dir <tp4 dir> --embeds <this section's AOT embeds.pt> --tp-degree 4 \
  --num-inference-steps 50 --seed 42 --skip-warmup --target-speedup 1.5 --out calibration_hunyuan_video_15_320x512x61_50.json
```
- PASS, 142.2 s. The output's labels are model `hunyuan_video_15`, `shape_label 320x512x61`,
  `num_steps 50`, `mod_input_source block0_modulated_input`, 49 samples (one prompt, seed 42).
  The cubic fit has threshold 0.2218 at target 1.5x.
- **Signal fit: weak, and the wrong sign.**
  - Cubic **R² 0.406** (the collector evaluates on its training samples).
  - Linear **Pearson −0.116** over all 49 steps (R² 0.013); −0.475 over the skip window,
    steps 5-44 (R² 0.225).
  - The probe's block-0 modulated-input rel-L1 rises monotonically, from 0.006 (step 1) to
    0.108 (step 49).
  - The guided noise_pred rel-L1 falls from 0.70 to about 0.12 by step 33, then rises again
    to 0.46.
  - The probe therefore does not track the quantity TeaCache needs, which is what the plan
    expected for hv15. `E/phase4/calibration/pairs_stats.txt` has the per-step pairs.
- **Context for 4a/4b:** the guided noise_pred changes by **12-70% per step** on hv15 (median
  18%). Reusing the cached residual on a skipped step therefore costs more here than on HV1.0.
  This is consistent with the 0.92-0.94 cosines in 4a/4b.

**Adaptive generate** on `69162f72…`: `--teacache-speedup 1.5 --teacache-calibration <above>`,
TP2xCFG2. The first adaptive use compiled the probe into the DiT's dir, as designed ("compiling the
TeaCache probe into … the DiT artifact is reused").

| run | `[teacache] stats` | denoise s (first step) | vs no-TeaCache | latents vs no-TeaCache (`base`) | mid frame | HBM GiB/core |
|---|---|---|---|---|---|---|
| no TeaCache (`base`, DiT only) | — | 82.1 (2.7) | ref | ref | (Phase 2 re-run video) | 9.69 |
| `--teacache-speedup 1.5` (`adapt`) | `full_steps 32, skipped_steps 18, probe_calls 50, last_delta_estimate -0.984` | **51.2** (2.6) | **-38%** (with 50 probe calls) | **cos 0.910548**, rel-L1 0.478, max-abs 4.77, no NaN/Inf | same fox and scene, **visibly degraded**: higher contrast/saturation, smeared legs (min frame std 77.2 vs 68.4) | **17.59** |

- **Outcome: LIMIT (weak signal).**
  - The controller engages: 18 of 50 steps skipped, which matches the 1.5x target (32 full
    steps).
  - The skips come from a signal that does not predict output change, and the quality loss
    is visible.
  - `last_delta_estimate` is negative: the cubic's prediction at the runtime signal. A
    negative predicted delta always skips. *Inference:* the runtime probe values fall where
    the cubic, fitted on 49 points from one prompt, has no meaning.
  - Recommendation: keep adaptive TeaCache off for hv15. Cadence 2 skips more (20), costs
    less (cos 0.939) and needs no probe.
- **Generate wall 575.2 s** (encode 8.4 s, probe compile + DiT load 69.3 s, denoise 51.2 s,
  host decode 416.3 s).
- **HBM: the probe costs a full second copy of the DiT weights on device.**
  - Per core: 17.59 GiB with the probe vs 9.69 without, at TP2xCFG2. The reset check (load
    only, no compile) shows the same 17.59.
  - The calibration's TP4 run: 9.9 vs 5.76 GiB.
  - The probe's `weights/tp*_sharded_checkpoint.safetensors` are hard links to the DiT's
    shards (link count 7), so disk is shared, but NxD loads the whole shard for the probe
    model.
  - At the default shape the DiT alone already uses 23.58 GiB per core, so adaptive TeaCache
    would need about 31.5 GiB of the 24 GiB per core and cannot fit there.
  - Recorded as a follow-up: load only the probe's few block-0 weights, or share the device
    weights between the DiT and the probe.
- **Per-request reset: PASS.** `reset_check.py` builds one TP2xCFG2 adaptive denoiser, loads it
  once, and runs two identical requests:
  - Both print `full_steps 32, skipped_steps 18, probe_calls 50, last_delta_estimate -0.984`.
  - Their latents are **bit-identical** (sha `5c02d1e1…`), and also bit-identical to the CLI
    `adapt` run.
  - So the controller and the probe's on-device `prev_mod` both reset per request.
  - Before `67d3d92` this script failed in the same `reset_state`.
- The CLI still refuses TeaCache flags with `--requests` ("per-request controller reset is a
  follow-up"); the reset itself is verified above at the API level.

Evidence: `E/phase4/calibration/` (calibration JSON, pairs JSON, `pairs_stats.txt`, logs),
`E/phase4/tp4-probe-compile/`, `E/phase4/adaptive/` (`adapt.mp4`, `adapt_mid.png`, `base` and
`reset` logs, comparisons), `E/phase4/probe-reset-device/`. Helper scripts in `E/phase4/`:
`pairs_stats.py`, `reset_check.py`, `probe_reset_device.py`.

### Phase 4 reference (R26): native TeaCache cadence 2 on the same settings

Is the 0.92-0.94 drift above a property of skipping steps on HV1.5, or of the AOT TeaCache?
Native TP2xCFG2 (`native-312` venv, `--backend neuron`), 320x512x61, 50 steps, seed 42, the
campaign prompt, no negative prompt: the same command as Phase 0's `gen50`, plus
`--teacache-cadence 2`.
- **No-TeaCache reference:** Phase 0's native `gen50` (same command, flag absent).
- **Re-checked here:** a re-run (`gen50b`) is bit-identical to it (sha `33baa167…`). Native is
  deterministic across sessions, and no native code changed since Phase 0 (the branch's diff
  under `difflet/backends/neuron/` is empty).
- **Embeds:** the encode output is tensor-identical to Phase 0's.

| run | `[teacache] stats` | denoise s (first step) | denoise minus first step | step median ms | latents vs native no-TeaCache | mid frame |
|---|---|---|---|---|---|---|
| native no TeaCache (`gen50`, `gen50b`) | — | 79.4 / 79.0 (14.4 / 14.2) | 65.0 s | 1325 / 1324 | ref (`gen50b` bit-identical) | fox in snow |
| native `--teacache-cadence 2` (`cad2`) | `full_steps 30, skipped_steps 20, probe_calls 0` (each rank) | 52.8 (14.4) | **38.4 s (-41%)** | 1323 (full steps) | **cos 0.932074**, rel-L1 0.366, max-abs 8.94, no NaN/Inf | same fox, pose and scene; visibly softer, with a color fringe on the ear |

**Side by side** (each vs its own stack's no-TeaCache run, same settings):

| stack | mode | skipped | cos | rel-L1 | max-abs |
|---|---|---|---|---|---|
| AOT | cadence 2 | 20 | **0.938644** | 0.346 | 5.32 |
| AOT | online-delta 0.6 | 17 | 0.918296 | 0.373 | 6.44 |
| AOT | adaptive, speedup 1.5 | 18 | 0.910548 | 0.478 | 4.77 |
| **native** | **cadence 2** | **20** | **0.932074** | 0.366 | 8.94 |

**Interpretation.** At cadence 2, native drifts as far as AOT does (0.932 vs 0.939; AOT is
slightly closer to its own reference) and saves the same 41% of warm denoise time. So the drift
is a property of skipping steps on HV1.5 at CFG 6, not an AOT TeaCache defect. A cos ≥ 0.99
gate for hv15 TeaCache cannot be met on either stack. Evidence: `E/phase4/native-cad2/`.

### Phase 5: JSONL batch (`--requests`, 2 requests), TP2xCFG2, 50 steps: **PASS after one fix**

Run on the `69162f72…` overlap artifact (HEAD `d001314` for the batch and request 1, `47b2f74`
for request 0's single run; the fix touches only the CLI's argument plumbing, not the DiT).
`requests.jsonl` (`E/phase5/requests.jsonl`):
- request 0: the campaign prompt, `negative_prompt "blurry, low quality, distorted, watermark"`,
  no seed (so `--seed`'s default 42);
- request 1: `"a red panda eating bamboo in a misty bamboo forest, soft morning light"`, `seed 1234`.

Each single run repeats its request's prompt, negative prompt, seed, steps and layout
(`E/phase5/p5.sh`). Equality is checked on the tensors' bytes (`tsha.py`): `torch.save` writes the
file name into the archive, so the batch's `latents_req0000.pt` and a single run's `latents.pt`
never share a file sha256 even when the tensors are identical.

| run | wall s | encode s | DiT load s | denoise s (step median ms) | host decode s | HBM GiB/core | latents (tensor sha) | vs batch |
|---|---|---|---|---|---|---|---|---|
| batch, request 0 | 1275.8 (both) | 9.7 | **27.0, once for both** | 80.2 (1579) | 455.5 | 9.69 | `defdb414…` | — |
| batch, request 1 | (same) | 9.5 | (same load) | 78.9 (1578) | 586.7 | 9.69 | `5a31b833…` | — |
| single 0, before the fix | 568.8 | 11.9 | 29.1 | 81.1 (1586) | 416.5 | 9.69 | `15073ca1…` | **cos 0.615843**, rel-L1 0.852, max-abs 7.08 |
| single 0, after `47b2f74` | 549.0 | 8.0 | 27.0 | 79.4 (1559) | 407.0 | 9.69 | `defdb414…` | **bit-identical**; MP4 byte-identical |
| single 1 | 948.0 (cold encode 103 s, DiT load 275 s) | 103.2 | 275.1 | 84.5 (1562) | 413.5 | 9.69 | `5a31b833…` | **bit-identical**; MP4 byte-identical |

- **DiT loaded once: PASS.** The batch log has one `aot transformer load 27.0 s` line (one NxD load
  on 4 cores, `[dp-router] worker 0: cores 0-3`), then `request 0: denoise` and `request 1: denoise`,
  and ends `[dp-router] done=2 failed=0 unfinished=0`. The encoder (load 0.9 s) and the host VAE are
  also loaded once.
- **Per-request seed and negative prompt: PASS.** The claimed request files carry `seed 42` +
  the negative prompt (request 0) and `seed 1234` + none (request 1). Request 0's
  `negative_prompt_embeds` differ from the empty negative's; request 1's equal it.
- **Bit-identity: PASS for both requests** (after the fix). The MP4s are byte-identical too
  (`243455a9…` request 0, `329a6343…` request 1), so the host decode is deterministic as well.
- **The failure before the fix was a Difflet bug, not batch-vs-single noise.**
  - The pre-fix single 0 had the same prompt embeds as batch request 0 but different
    `negative_prompt_embeds` (cos 0.920) and mask (cos 0.655).
  - Its embeds file was byte-identical to the no-negative embeds of Phases 1-4 (`3051691e…`), and
    its latents were bit-identical to the Phase 2 re-run's no-negative latents (`p2r-tp2cfg`).
  - So the single run never saw `--negative-prompt`. Request 1, which has no negative prompt,
    matched bit for bit from the start, which also rules out a noise-path difference between
    batch and single mode.
  - Root cause, fix and test are in the Phase 4-5 bug ledger below.
- Mid frames (`E/phase5/*_mid.png`): request 0 shows a red fox walking through snow among
  snow-covered firs; request 1 a red panda holding a bamboo leaf in a bamboo grove. The
  pre-fix single 0 (`prefix_single0_mid.png`) is the no-negative video of Phase 2.
- `--requests` with TeaCache flags is still refused by the CLI (Phase 4c); this batch runs
  without TeaCache.

Evidence: `E/phase5/` (logs, meta, `batch_req{0,1}.mp4`, mid frames, comparisons,
`tensor_and_mp4_sha.txt`, `sha256.txt`; `prefix_single0.*` is the pre-fix run).

Reproduce (`R` = `run_difflet.sh trainium`):
```
R generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --height 320 --width 512 --num-frames 61 \
  --steps 50 --tp-degree 2 --cfg-parallel --requests requests.jsonl --keep-work-dir --work-dir batch.work
R generate ... --prompt "a cinematic shot of a red fox running through a snowy forest" \
  --negative-prompt "blurry, low quality, distorted, watermark" --seed 42 --keep-work-dir --work-dir single0.work --output single0.mp4
R generate ... --prompt "a red panda eating bamboo in a misty bamboo forest, soft morning light" --seed 1234 \
  --keep-work-dir --work-dir single1.work --output single1.mp4
```

### Bug ledger (Phase 4-5)

| commit | bug | how it showed | fix | test | device check |
|---|---|---|---|---|---|
| `67d3d92` | `teacache_probe15.reset_state` used the non-jit NxDModel API (`.states`, `write_to_neuron_buffer`); the application loads its model with `torch.jit.load` (`.state`, no `write_to_neuron_buffer`) | every adaptive hv15 AOT request died before step 0: `AttributeError: 'RecursiveScriptModule' object has no attribute 'states'` (calibration collector and `--teacache-speedup` alike) | zero each rank's `prev_mod` in place through `.state` | `tests/unit/models/hunyuan_video/test_aot15.py` mocks the jit-loaded shape | after reset every rank's `prev_mod` reads back all-zero, and `delta(x1)` equals the fresh-load value; two identical adaptive requests on one load are bit-identical (Phase 4c) |
| `47b2f74` | outside `--requests`, `--negative-prompt` reached no hv15 stage: the shared `_stage_cli_args` does not forward it, and `stage_loop`'s single (legacy) request carried no `negative_prompt`, so the encode stage encoded `""` | Phase 5: single run 0 vs batch request 0, cos 0.616; the single run's embeds were the no-negative embeds | `_run_aot` appends `--negative-prompt` to the stage args; the legacy request carries `args.negative_prompt` (a `--requests` line's still wins) | `test_legacy_request_carries_the_negative_prompt` (`tests/unit/cli/dp/test_stage_loop.py`), `test_aot_generate_forwards_the_negative_prompt` (driver → stage parser → encode stage); both fail on the parent commit | single 0 re-run: bit-identical to batch request 0, MP4 byte-identical |

Not fixed, and recorded:
- **Native has the same gap.** `_stage_cli_args` is shared with the native path, and the plan
  forbids changing its body (global constraint). So `--backend neuron` hv15 still ignores
  `--negative-prompt` outside `--requests`. The fix there is one tuple entry in
  `_stage_cli_args`; it needs a waiver of the constraint.
- **The CLI-wide default in batch mode.** The module docstring says a batch request without a
  negative prompt takes the command line's. The encode stage reads only `req.negative_prompt`,
  so a `--negative-prompt` given next to `--requests` is still ignored for such requests, on
  both backends. `stage_loop.effective(req, args, "negative_prompt", None)` in `_stage_encode`
  would fix it, but that is also a native-shared function body.
- `--negative-prompt`'s help text still calls it a Qwen-Image option.

### Phase 4-5 summary

| item | verdict | key numbers | evidence |
|---|---|---|---|
| R23 no-ccop perf check | done; the flag is now shape-aware (`420d2fa`, `d001314`) | no-ccop +13.4% per step at 320x512x61; overlap kept for N ≤ 16384 image tokens | `E/phase4/noccop-ab/` |
| 4a cadence 2, TP2xCFG2 | **PASS** on skips, time and video; cos gate **not met on either stack** | `skipped_steps 20` (both CFG halves skipped together), -41% warm denoise, cos 0.938644 | `E/phase4/tp2cfg/` |
| 4b online-delta 0.6 | PASS (engages), same drift class | 17 skipped, -35%, cos 0.918296 | `E/phase4/tp2cfg/` |
| 4c adaptive (calibration + speedup 1.5) | **LIMIT** (weak signal) after bug fix `67d3d92` | cubic R² 0.406, Pearson −0.116; 18 skipped, -38%, cos 0.910548, visibly degraded; probe doubles HBM (17.59 vs 9.69 GiB/core); per-request reset bit-identical | `E/phase4/{calibration,adaptive,probe-reset-device}/` |
| R26 native cadence 2 | reference | native cos 0.932074 vs AOT 0.938644 at 20 skips, both -41%; native deterministic (re-run bit-identical) | `E/phase4/native-cad2/` |
| 5 JSONL batch (2 requests) | **PASS** after bug fix `47b2f74` | DiT loaded once (27.0 s); both requests bit-identical to their single runs; MP4s byte-identical | `E/phase5/` |

Rulings still open from this section:
1. **The hv15 TeaCache cosine gate.** Native shows the same drift at cadence 2, so 0.99 is not
   reachable on HV1.5 at CFG 6. Proposal: gate hv15 TeaCache on `skipped_steps` plus the visual
   check, plus a distance no worse than native's at the same cadence (AOT 1 − cos = 0.061 vs
   native 0.068).
2. **Adaptive TeaCache for hv15:** keep it off. The signal is weak and the probe costs a second
   copy of the DiT weights in HBM, so it cannot fit at the default shape.
3. **Native `--negative-prompt` outside `--requests`, and the CLI default in batch mode:** both
   still dropped (bug ledger above). Fixing them needs a waiver of the "native function bodies
   unchanged" constraint.
