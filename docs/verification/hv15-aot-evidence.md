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
