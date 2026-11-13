# DiT Block Profile (bf16 vs linear MXFP8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For seven DiTs on trn3, measure where one DiT block spends its time — per engine
(TensorE, VectorE, ScalarE, GpSimd, HBM DMA, collectives), per block component (q/k/v
projection, qk-norm, RoPE, attention core and its phases, output projection, MLP, modulation,
collectives) and in execution order — in bf16 and with linear MXFP8, reconcile it with the DiT
step, and publish one English artifact page with charts and a bottleneck verdict per model.

**Architecture:** A 1-step `generate` under `neuron-explorer inspect` dumps every rank's NEFFs
and a system profile, and exits by itself. The block NEFFs are the large NEFFs present on every
rank; each is confirmed by a per-model signature, replayed with its collectives by
`neuron-explorer capture` (one worker per rank of the world), and exported by `view` as
`summary-json` and `parquet` (one row per instruction, torch module path in `layer`). An analysis
package classifies instructions into components, separates HBM DMA instructions from compute on
the engine that issues them, and measures time as the union of busy intervals per physical core
(not summed durations). The system profile gives each block NEFF's executions per step and its
in-model time, so blocks are reconciled with the step. A collector merges everything with the
step times already in the evidence doc; the page is built from its `results.json`.

**Tech Stack:** bash, Python 3.12 (`~/venvs/native-fp8`), pandas + pyarrow, `neuron-explorer`
2.33 (`/opt/aws/neuron/bin`), pytest; Artifact page (HTML, charts).

**Spec:** the user's request of 2026-11-13 (conversation): "profile qwen / flux / ltx / hunyuan
1.0 / hunyuan 1.5 / wan 2.1 / wan 2.2 under PyTorch native and under linear quantization (HV1.5
native only); an English artifact recording each step's time, drawn as a human-readable chart —
in one DiT block, how long HBM, TensorE, VectorE, PSUM take." Reviewed by a Fable panel on
2026-11-13 (27 accepted changes; this is the revised plan).

## Global Constraints

- Device runs: trn3.48xlarge, LNC=2, `--backend neuron`, venv `~/venvs/native-fp8`,
  `export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH`,
  `PYTHONPATH=<worktree> python -m difflet.cli.main`.
- **Busy devices are never used; idle devices are used in parallel** (user, 2026-11-13).
  `require_idle <d>` (in `scripts/profile_dit/lib.sh`) runs before every command that touches a
  device (dump, each capture): it kills the lane's own leftover process group, then requires
  `fuser /dev/neuron<d>` to be empty and appends the `fuser` output to the lane log. A busy
  device, or a log line saying the cores are busy (exact string taken from the archived
  `failed_laneB` logs), writes `BUSY` and exits; `run_all.sh` rescans and requeues the config on
  another idle device — never a retry on the same device, never counted as a bug. The check
  narrows the race window, it does not close it. No private per-device lock. On 2026-11-13
  devices 0 and 6 are held by someone else's `neuronx-cc` processes.
- Lanes: one device each, `NEURON_RT_VISIBLE_CORES` = its cores (device d = 4d..4d+3), a 48-thread
  CPU set on that device's NUMA node (two lanes per node; devices 0-7: CPUs 48-95,144-191; 8-15:
  CPUs 0-47,96-143). Cold compiles (HV1 544x960x129, Wan 81f, HV1.5 121f) run at most 3-4 at once:
  the first one alone, with a 10 s sampler of the lane's process-tree RSS and `MemAvailable`;
  the concurrency follows that measurement. A `difflet compile` that returns in seconds is warm.
- Root-disk reads: before a lane starts, a serial prewarm `cat`s to `/dev/null` the root-disk
  files it will load (the LTX-2 and Wan 2.1 hub snapshots, each config's `weights-bf16/<rev>` and
  `quantized/<repo>__<rev>`), as `bench.sh`'s `warm_hub` does; the other models use
  `HF_HUB_CACHE=/mnt/nvme/difflet-quant-bench-hub` after checking that its revisions match the
  cache keys.
- Disk: every `generate` gets explicit `--output` and `--work-dir` under
  `/mnt/nvme/difflet-prof/<key>/<arm>/`; NTFF, parquet and NEFF stay there. `run_all.sh` checks
  `df --output=avail -BG /` before every lane; below 60 GB it writes `FAILED disk` and stops
  starting lanes. Never delete a cache without the user's approval.
- Compile caches: `/home/ubuntu/difflet-cache-fp8` (Qwen-Image), `/home/ubuntu/difflet-cache-video`
  (FLUX and the video models; ruling R7). Wan bf16 dumps set `DIFFLET_NKI_TRACE_CACHE=1` once Task 1 has confirmed it is
  not part of the compile-cache key; otherwise budget ~10 min per Wan bf16 dump.
- Supervision: `.claude/skills/difflet-device-verify/scripts/supervise.sh` in a Monitor (not the
  repo-root `scripts/supervise.sh`, which does not restart). Each lane ends in one terminal file
  `<out>/<key>/<arm>/STATUS` whose first line is `DONE`, `FAILED <reason>` or `BUSY`; a restarted
  lane skips phases already done (`dump exit 0` in `dump.log`, `blk<i>/summary.json` present).
- Commits: one per task, pushed to `origin/feat/mxfp8-sage-attention`; breakdowns committed with
  `git add -f docs/verification/assets/2026-11-13-dit-block-profile/<key>/*/blk*.json` (the path
  is git-ignored; never `git add -A`). The artifact page is updated after every config.
- Do not use `/home/ubuntu/fp8-bench/hv1_tp4/profsp_*/blk2` as a block reference: it is the VAE
  (6,431,167 bytes, 0 collectives, 205,399 of 205,734 instructions from `vae/modeling_vae.py`).

### The configurations

Flags are `scripts/quant_bench/bench.sh`'s `args_for <key>` verbatim, with `--steps 1`; the HV1
long shape takes its command line from evidence section 7f (no `--sp`). Capture workers = world
size = tp x cfg x cp = 4 for every key.

| key | model | shape | layout | arms | block NEFFs per arm |
|---|---|---|---|---|---|
| `qwen_image` | Qwen-Image | 1024x1024 | tp4 | bf16, mxfp8 | 1 |
| `flux` | FLUX.1-dev | 1024x1024 | tp4 | bf16, mxfp8 | 2 (dual, single) |
| `ltx_2` | LTX-2 | 512x768x121 | tp4 | bf16, mxfp8 | 1 |
| `wan21_official` | Wan 2.1 T2V 14B | 480x832x81, CFG 5.0 | tp4 | bf16, mxfp8 | 1 |
| `wan22_official` | Wan 2.2 T2V A14B, `--dual-expert` | 480x832x81, CFG 4.0/3.0 | tp4 | bf16, mxfp8 | 1 (`--steps 1` runs the high-noise expert only) |
| `hunyuan_video_L` | HunyuanVideo | 544x960x129 | tp4 | bf16, mxfp8 | 2 (dual, single) |
| `hv15_official` | HunyuanVideo-1.5 480p t2v | 480x848x121, CFG 6.0 | tp2 x CFG2 | bf16 | 1 |

17 block breakdowns in all (9 bf16, 8 mxfp8). DiT step medians for these shapes are in
`docs/verification/2026-11-07-mxfp8-video-models.md` sections 7f, 11, 12; the collector cites
section, host (section 12 ran on a second trn3) and step count, and reruns nothing.

### Time budget

| part | per config | all configs, one lane per idle device |
|---|---|---|
| compile, warm | seconds | seconds |
| compile, cold | 2-3 min (Qwen, FLUX, LTX-2), 11-16 min (Wan 81f), ~24 min (HV1.5 121f), up to ~42 min (HV1 544x960x129: 2,397 / 2,490 s) | ~45 min with 3-4 cold compiles at a time |
| dump (load + 1 step + exit) | 2-10 min | ~30-45 min with the prewarm |
| capture + summary + parquet per block NEFF | 3-15 min (measured in Task 1) | parallel unless Task 4 Step 2 finds interference |
| analysis, page | | ~3-4 h |

## Review Focus

1. A large NEFF that is not a block (VAE, text encoder, embedder) picked as one: Task 1 takes
   only NEFFs present on every rank and confirms each with a signature; tests pin a rank-0-only
   NEFF and a VAE-dominated one being rejected.
2. HBM transfers counted as ScalarE or Sync compute (DMA instructions are queued on those
   engines: in HV1's dual block 48.3 of ScalarE's 53.7 ms are `DMA_*`): Task 3 moves every
   `DMA_*` row to `hbm_dma`; a test pins it.
3. Summed durations exceeding the block (LNC=2 has two physical cores; pipelined instructions
   overlap): Task 3 measures interval unions per physical core and reports their mean; tests pin
   overlap counted once and components summing to no more than the block.
4. Rows with no module path (`layer == "Unknown"`, 20% of TensorE and 58% of GpSimd in HV1's dual
   block): reported as `unattributed`, never folded into `other`; the 10% `other` gate counts
   only attributed rows.
5. "PSUM time": PSUM is accumulator memory, not an engine; the page shows PSUM bytes and the PSUM
   read-out time split into pure copies (savable by fusion) and fused compute — both already
   inside the VectorE/ScalarE bars.

---

### Task 1: The capture script

**Files:**
- Create: `scripts/profile_dit/lib.sh` (`require_idle`, `write_status`), `scripts/profile_dit/capture.sh`,
  `scripts/profile_dit/signatures.json`, `scripts/profile_dit/README.md`
- Create: `scripts/profile_dit/system_profile.py` (copied from `/home/ubuntu/fp8-bench/wan21_tp4/analyze_profile.py`)
- Test: `tests/unit/scripts/test_profile_dit_capture.py`

**Interfaces:**
- Produces: `capture.sh --key <key> --arm <bf16|mxfp8> --device <d> --cpus <list> --out <root>`
  writing under `<root>/<key>/<arm>/`: `dump.log`, `system.json` (rank 0's system profile via
  `view --output-format json`) and `system_summary.json` (`system_profile.py`: per NEFF
  `{execs_per_step, in_model_median_ms}`, non-block NEFF total, host gaps, `cc_running`),
  `blk<i>/{block.neff, summary.json, pq/trace/*.parquet, capture.log}`, and `STATUS`.
- Produces: `signatures.json` = per key, per block kind, `{"require": [module roots], "attention_kernel": true}`;
  a candidate whose dominant `nki_source_location` is under `/vae/` or a text encoder is rejected.

- [ ] **Step 1: Failing tests** (drive `capture.sh` with a recording `PY`, a fake
  `neuron-explorer` and a fake `fuser`, as `tests/unit/scripts/test_quant_bench.py` drives
  `bench.sh`): each key's generate carries `--steps 1` and the table's height / width / frames;
  `--quant mxfp8` only on the mxfp8 arm; `NEURON_RT_VISIBLE_CORES=4d..4d+3` and
  `NEURON_RT_INSPECT_DEVICE_PROFILE=1`; capture uses `--collectives-worker-count=4
  --collectives-workers-per-node=4` for every key including `hv15_official`; a NEFF present only
  in rank 0's dump is not captured; a candidate whose rows come from `vae/modeling_vae.py` is
  rejected; `fuser` busy before the dump, or before the second capture, writes `BUSY` and exits
  0 without further device commands; `hv15_official mxfp8` exits 2 naming the pair.
- [ ] **Step 2: Run** — `DIFFLET_BACKEND=cpu PYTHONPATH=. ~/venvs/native-fp8/bin/python -m pytest tests/unit/scripts/test_profile_dit_capture.py -q -p no:cacheprovider`; expected FAIL.
- [ ] **Step 3: Implement.** Dump: `timeout -s INT -k 90 <limit> neuron-explorer inspect -o dump
  <PY> -m difflet.cli.main generate ...`, letting the 1-step run exit by itself (no polling, no
  kill); timeout -> `FAILED timeout dump`. Candidates: NEFFs over 1 MB whose sha256 appears in
  every rank's dump directory; their count must equal the table's, else `FAILED candidates`.
  Rank 0 vs rank 2 sha256 differ (HV1.5's CFG halves) -> profile both. Capture each with
  `timeout 1200 neuron-explorer capture ... --num-exec=8 --profile-nth-exec=4` (two profiled
  executions), then `timeout 900 view --output-format summary-json` and `--output-format parquet`;
  wait for `fuser` to clear between device commands. Confirm against `signatures.json`. The HV1
  dual-stream signature is written now (`HunyuanVideoAttention[attn]` and
  `HunyuanVideoFeedForward[ff]` roots, attention_cte rows); every other entry is filled from that
  key's first real parquet in Step 5 / Task 4, the HV1 single-stream one from the first capture of
  its 6,237,464-byte NEFF.
- [ ] **Step 4: Run the tests** — expected PASS.
- [ ] **Step 5: Device smokes**, each on an idle device: `qwen_image bf16`, `flux bf16` (two
  blocks) and `hv15_official bf16`. For each: `STATUS` = `DONE`, the expected block count kept,
  its signature entry written; for HV1.5 record CcOp.parquet's `replica_group` in `README.md` and
  stop to report unless it is `[[0,1],[2,3]]` or `[[0,1]]`. Capture `qwen_image`'s block once more
  with `--enable-dge-notifs`; keep the flag if it does not time out. Confirm whether
  `DIFFLET_NKI_TRACE_CACHE` enters the compile-cache key. Record each phase's wall time in
  `README.md` (the measured row of the time budget).
- [ ] **Step 6: Commit and push** — `scripts(profile): capture one DiT block's device profile per model and arm`.

### Task 2: Instruction classifier (offline, before any device run)

**Files:**
- Create: `scripts/profile_dit/components.py`
- Test: `tests/unit/scripts/test_profile_dit_components.py`

**Interfaces:**
- Produces: `COMPONENTS = ("modulation", "norm", "qkv_proj", "qk_norm", "rope", "attn_glue",
  "attention", "out_proj", "mlp_up", "mlp_act", "mlp_down", "collective", "other", "unattributed")`;
  `classify(layer: str, hlo_name: str, nki_source: str, hlo_attrs: str, model: str) -> str`;
  `attention_phase(nki_source: str) -> str` in `{"qk_matmul", "mask_softmax", "pv_matmul", "other"}`
  from a line-range table of nkilib 2.14's `attention_cte.py` (MM1 near :4128, mask/softmax near
  :3381, :3984, :4287, MM2 near :4472 — exact ranges read from the file);
  `attention_module(layer: str) -> str` (`attn`, `attn1`, `attn2`, `audio_attn1`, ...).

- [ ] **Step 1: Failing tests** with real strings from `/home/ubuntu/fp8-bench/hv1_tp4/profsp_V15/blk1/pq/trace/Instruction.parquet`
  (blk1 only), e.g.:

```python
assert classify("AdaLayerNormZero[norm1][0]/ColumnParallelLinear[linear][0]_dot", "%dot = dot(...)", "", "", "hunyuan_video") == "modulation"
assert classify("AdaLayerNormZero[norm1][0]/ColumnParallelLinear[linear][0]_all-gather.115", "%all-gather.115 = all-gather(%reshape.114)", "", "", "hunyuan_video") == "collective"
assert classify("HunyuanVideoAttention[attn][0]_add.330", "%add.330 = add(...)", "", "<hlo_attrs naming ops_impl/embeddings.py>", "hunyuan_video") == "rope"
assert classify("Unknown", "", "", "", "hunyuan_video") == "unattributed"
```

  plus the attention custom-call (-> `attention`), an attention-module `RowParallelLinear[0][0]_dot`
  (-> `out_proj`), and one MLP up / act / down case, all copied from that parquet.
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement** ordered rules: `layer == "Unknown"` or empty with no source ->
  `unattributed`; collectives by HLO opcode; bare `<AttentionClass>[...]_<op>` rows that are not
  the attention custom-call -> `rope` when `hlo_attrs` names the rotary source, else `attn_glue`
  (this rule runs before the attention rule); `attention` by `nki_source` naming an attention
  kernel; then module-path rules per model (q/k/v linears, `norm_q|norm_k|norm_added_*`, output
  projection = `RowParallelLinear[0][0]` under an attention module or `to_out|to_add_out|proj_out*`,
  MLP up/act/down per model naming, `norm1*|img_mod|txt_mod|AdaLayerNorm*` -> `modulation`, other
  norms -> `norm`); an MX linear's kernel rows (`matmul_mxfp8` source) inherit the linear's
  component from `layer`. Unmatched attributed rows -> `other`.
- [ ] **Step 4: Run** — expected PASS; then classify all of HV1 blk1: `other` below 10% of every
  engine's attributed busy time (extend rules with tests until it is).
- [ ] **Step 5: Commit and push** — `scripts(profile): classify a block's instructions into DiT components`.

### Task 3: Per-block analysis

**Files:**
- Create: `scripts/profile_dit/analyze.py`
- Test: `tests/unit/scripts/test_profile_dit_analyze.py`

**Interfaces:**
- Consumes: Task 2's functions; `blk<i>/summary.json`, `blk<i>/pq/trace/{Instruction,ActiveTime,CcOp,Warning}.parquet`.
- Produces: `analyze(block_dir: Path, model: str) -> dict`, CLI `analyze.py <block_dir> --model <key>`
  writing `<block_dir>/breakdown.json`:
  - `block_ms` = summary `total_time`; `replay_spread_ms` = the two profiled executions' difference;
    `block_neff_sha256`;
  - `engine_class` per row = `"hbm_dma"` if `opcode` starts with `DMA_` else the engine
    (`tensor`, `vector`, `scalar`, `gpsimd`, `sync`);
  - `engines_ms` {engine_class: ms} and `components` {component: {engine_class: ms}}: the union
    of `[start_ts, end_ts)` per (engine_class[, component], `pcore_idx`), `EVENT_SEMAPHORE*` rows
    excluded, mean over the two physical cores; `collective` component time from `CcOp.parquet`;
  - `engines_check` = summary `*_active_time` and `dma_active_time` with its static /
    hardware-dynamic / software-dynamic split (cross-check fields);
  - `attention_detail` {module: {phase: {engine_class: ms}}};
  - `memory_bytes` {hbm_read, hbm_write, sbuf_read, sbuf_write, psum_read, psum_write,
    psum_to_sbuf, spill_save, spill_reload}; `psum_readout_copy_ms` (COPY / CAST rows with
    `psum_read_bytes > 0`) and `psum_readout_fused_ms` (other such rows);
  - `mfu`, `mbu`; `throttle` {avg_util_limit, limited_time_share} from
    `throttle_avg_util_limit_nc\d+_percent` and `throttle_activity_1_active_time_nc\d+_percent` (means);
  - `other_share` and `unattributed_share` per engine_class; `warnings` (Warning.parquet messages);
  - `timeline`: pcore 0's merged intervals `{component, engine_class, start_us, end_us}` in
    execution order, gaps under 5 us merged (for the page's time-line chart).

- [ ] **Step 1: Failing tests** on synthetic frames: a Scalar `DMA_DIRECT2D` row lands in
  `hbm_dma`; two overlapping rows of one engine count once; a component sum never exceeds 1.10 x
  its engine's union nor `block_ms`; an `EVENT_SEMAPHORE` row adds nothing; a `RANGE_SELECT` row
  reading PSUM lands in `psum_readout_fused_ms`, a `COPY` row in `_copy_ms`; the throttle regexes
  read two cores' fields and average them; `analyze` raises `ValueError` naming the engine when
  `other` exceeds 10% of an engine's attributed time, or when unmapped attention lines exceed 10%
  of the attention time.
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement** with pandas; interval union by sorting starts and sweeping.
- [ ] **Step 4: Run** — expected PASS; then on HV1 blk1 and the Task 1 smokes: each engine's
  union reproduces `ActiveTime.parquet` within a tolerance fixed here and written into the test
  as a constant.
- [ ] **Step 5: Commit and push** — `scripts(profile): per-engine, per-component, per-phase breakdown of one block NEFF`.

### Task 4: Device runs, all configurations

**Files:**
- Create: `scripts/profile_dit/run_all.sh` (prewarm, lane scheduler, requeue, disk gate)
- Modify: `scripts/profile_dit/signatures.json`, `tests/unit/scripts/test_profile_dit_components.py`

- [ ] **Step 1:** Scan `fuser /dev/neuron<d>` for d in 0..15; check `df /` and `df /mnt/nvme`;
  run the prewarm.
- [ ] **Step 2: Capture interference check:** capture the `qwen_image` smoke NEFF once with no
  other lane running and once while the other lanes dump; if `total_time` or `avg_util_limit`
  moves by more than two lone captures differ, captures run one at a time (dumps and compiles
  stay parallel). Record the result in `README.md`.
- [ ] **Step 3:** One lane per (key, arm) on its own idle device, all at once if enough devices
  are idle, cold compiles capped as the Global Constraints say, `BUSY` configs requeued, lanes
  with a `STATUS` skipped. After each config: `analyze.py` on every kept block; a new block kind
  gets its signature entry and a classifier test case before its breakdown is committed.
- [ ] **Step 4:** After each config: `git add -f docs/verification/assets/2026-11-13-dit-block-profile/<key>/*/blk*.json`,
  commit, push, update the page.
- [ ] **Step 5:** End state: every lane `DONE`; 17 breakdowns; `wan22_official` vs
  `wan21_official` block sha256 compared and, when equal, noted on the page.

### Task 5: Collector, page and evidence doc

**Files:**
- Create: `scripts/profile_dit/collect.py`, `docs/verification/2026-11-13-dit-block-profile.md`
- Test: `tests/unit/scripts/test_profile_dit_collect.py`

**Interfaces:**
- Consumes: every `breakdown.json` and `system_summary.json`.
- Produces: `results.json` = `{"configs": [{key, arm, shape, layout, step_ms, step_source:
  {section, host, steps}, blocks: [{...breakdown, execs_per_step, in_model_median_ms}],
  non_block_ms, host_gap_ms, outside_blocks_ms, bottleneck: {engine_class, share, mfu},
  attention_share, addressable_attention_share}], "quality": [...]}` where
  - `outside_blocks_ms` = `step_ms` - sum(execs_per_step x in_model_median_ms);
  - `bottleneck` = the engine_class with the largest union time, its share of `block_ms`, and
    `mfu` beside it ("not TensorE-bound" when TensorE's share is low);
  - `attention_share` = attention union time on that engine_class / `block_ms`;
    `addressable_attention_share` = only the attention modules Plan B routes (self/joint, D = 128);
  - `quality` = the linear-MXFP8 rows of evidence sections 7f, 10, 11, 12 (vs device bf16 and vs
    CPU fp32), each with shape, section, and `profiled_shape: bool`.

- [ ] **Step 1: Failing tests** on synthetic breakdowns: `bottleneck` picks the largest
  engine_class after the DMA split; `attention_share` uses that engine_class; a config without an
  evidence step time gets `step_ms: null, step_source: "not measured"`; `outside_blocks_ms` is
  computed from executions x in-model medians; every quality row carries `shape`, `section` and
  `profiled_shape`.
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — expected PASS.
- [ ] **Step 5: The page** (load the `dataviz` skill and call the Artifact quickstart, intent
  `other`, before writing it). Sections: (1) overview — model, shape, layout, DiT step bf16 / MXFP8
  with source, blocks x executions vs step ("outside the blocks" remainder), bottleneck,
  attention share, addressable share, RoPE and cross-attention columns; (2) per model, a
  **time line of one block** in execution order, segments coloured by engine_class and labelled by
  component (the "how long does each step take" view); (3) engine union times bf16 vs MXFP8 as
  grouped bars with the block wall time as a reference line, throttle beside each block, and
  "no difference" where a bf16 / MXFP8 gap is below the replay spread; (4) per-component stacked
  bars per engine_class, with `unattributed` drawn on its own; (5) attention phases (QK matmul,
  mask/softmax, PV matmul) per module; (6) memory traffic and the PSUM note; (7) quality;
  (8) the bottleneck verdict per model in two sentences; (9) method: "time = busy time of one
  physical core (union of intervals, semaphore waits excluded); a logical core is two physical
  cores; engines run in parallel, so bars overlap; HBM time = time of DMA instructions; DMA
  values carry the profiler's own warning when present", the paths, what was not measured.
  Publish; record the URL in the evidence doc.
- [ ] **Step 6: Evidence doc** — the same tables in Markdown, the commands, the paths under
  `/mnt/nvme/difflet-prof/`, what was not measured.
- [ ] **Step 7: Commit and push** — `docs(profile): DiT block profile of seven models, bf16 vs linear MXFP8`.
