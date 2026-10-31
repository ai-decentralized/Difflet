# HunyuanVideo 1.5 on the AOT (trainium) backend: on-device evidence

Plan: `docs/superpowers/plans/2026-10-30-hv15-aot-parity.md`. Host: `trn2.3xlarge`, one Neuron
device, 4 logical NeuronCores (logical config 2), 96 GB HBM. Logs and NEFFs live outside the repo
under `S = /home/ubuntu/.claude/jobs/294a04fc/tmp/spike/`.

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
