# DiT Block Profile (bf16 vs linear MXFP8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For seven DiTs on trn3, measure where one DiT block spends its time — per engine
(TensorE, VectorE, ScalarE, GpSimd, DMA/HBM, collectives) and per block component (q/k/v
projection, qk-norm, RoPE, attention core, output projection, MLP, modulation, collectives) —
in bf16 and with linear MXFP8, and publish the result as one English artifact page with charts
and a bottleneck verdict per model.

**Architecture:** Reuse the phase-H recipe from the MXFP8 campaign: a 1-step `generate` under
`neuron-explorer inspect` dumps the NEFFs; each block NEFF is replayed with its collectives by
`neuron-explorer capture` (one worker per TP rank); `neuron-explorer view` exports
`summary-json` (engine totals) and `parquet` (one row per instruction, with the torch module
path in the `layer` column). A small analysis package classifies every instruction into a block
component and sums engine time per component; a collector merges all configs with the DiT step
times already measured in the campaign's evidence doc. The page is built from the collector's
`results.json`.

**Tech Stack:** bash, Python 3.12 (`~/venvs/native-fp8`), pandas + pyarrow (parquet),
`/opt/aws/neuron/bin/neuron-explorer` 2.33, pytest; Artifact page (HTML, charts).

**Spec:** the user's request of 2026-11-13 (conversation): "profile qwen / flux / ltx / hunyuan
1.0 / hunyuan 1.5 / wan 2.1 / wan 2.2 under PyTorch native and under linear quantization (HV1.5
native only); an English artifact recording each step's time, drawn as a human-readable chart
— in one DiT block, how long HBM, TensorE, VectorE, PSUM take."

## Global Constraints

- Device: trn3.48xlarge, LNC=2, `--backend neuron`; device runs use the `native-fp8` venv with
  `export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH` and
  `PYTHONPATH=<worktree> python -m difflet.cli.main`.
- Never use a busy device; run in parallel on every idle one (user, 2026-11-13). Busy check
  before every launch: `fuser /dev/neuron<d>` non-empty = busy (`neuron-ls -a` lists only
  itself). On 2026-11-13 devices 0 and 6 are held by someone else's `neuronx-cc` processes.
  One lane per idle device; pin `NEURON_RT_VISIBLE_CORES` (device d = cores 4d..4d+3) and
  `taskset` a disjoint slice of that device's NUMA node's CPUs (devices 0-7: CPUs
  48-95,144-191; devices 8-15: CPUs 0-47,96-143). Lanes must not slow each other: at most two
  weight loads from the root EBS disk (~125 MB/s, shared) at a time — lanes take a load slot
  (`flock` on one of two lock files) for the dump's load phase; device-side `capture` replays
  are independent and run unthrottled.
- Disk: root has ~142 GB free. Profile outputs go to `/mnt/nvme/difflet-prof/` (NTFF and parquet
  are 0.1-1 GB per block). Stop and ask the user if `df /` falls below 60 GB; never delete a
  cache without the user's approval.
- Compile caches: reuse the campaign caches — `/home/ubuntu/difflet-cache-fp8` (Qwen-Image,
  FLUX) and `/home/ubuntu/difflet-cache-video` (the video models).
- Seed 42; prompts and guidance as `scripts/quant_bench/bench.sh` sets them for each key.
- MXFP8 arm = `--quant mxfp8` with the model's committed default set (`difflet/quant/targets.py`);
  HunyuanVideo-1.5 has no MXFP8 arm (user).
- Long jobs run under `scripts/supervise.sh`-style supervision in a Monitor (the
  difflet-device-verify skill); a job that dies without its done-marker is restarted, a job that
  writes a failure marker is reported, not retried.
- Commit after every task and push (`git push origin feat/mxfp8-sage-attention`); update the
  artifact page after every model.
- Never commit NTFF, parquet or NEFF files; commit scripts, `breakdown.json` / `results.json`
  and the evidence doc.

### The 13 configurations

| key | model id | shape | layout | arms | block NEFFs |
|---|---|---|---|---|---|
| `qwen_image` | Qwen/Qwen-Image | 1024x1024 | tp4 | bf16, mxfp8 | 1 |
| `flux` | black-forest-labs/FLUX.1-dev | 1024x1024 | tp4 | bf16, mxfp8 | 2 (dual, single) |
| `ltx_2` | Lightricks/LTX-2 | 512x768x121 | tp4 | bf16, mxfp8 | 1 |
| `wan21` | Wan-AI/Wan2.1-T2V-14B-Diffusers | 480x832x81, CFG 5.0 | tp4 | bf16, mxfp8 | 1 |
| `wan22` | Wan-AI/Wan2.2-T2V-A14B-Diffusers `--dual-expert` | 480x832x81, CFG 4.0/3.0 | tp4 | bf16, mxfp8 | 1 (both experts) |
| `hunyuan_video` | hunyuanvideo-community/HunyuanVideo | 544x960x129 | tp4 | bf16, mxfp8 | 2 (dual, single) |
| `hunyuan_video_15` | hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v | 480x848x121, CFG 6.0 | tp2 x CFG2 | bf16 | 1 |

The DiT step medians for these exact shapes are in
`docs/verification/2026-11-07-mxfp8-video-models.md` sections 11-12 and section 7f (HunyuanVideo
544x960x129: 18,494 / 17,968 ms); the collector cites them with their section, no step-time
reruns.

### Time budget (estimate, for the user)

| part | per config | all 13, one lane per idle device (14 idle today) |
|---|---|---|
| compile, cache warm | ~0 | ~0 |
| compile, cache cold | 2-3 min (Qwen, FLUX, LTX-2) to 11-30 min (Wan 81f, HV1 L, HV1.5 121f) | ~30 min (the longest compile; CPU-bound, all lanes at once) |
| dump (load + 1 step, stop after the NEFFs exist) | 2-8 min | ~30-45 min: weight loads share the root disk, two at a time |
| capture + summary per block NEFF | ~1 min | parallel |
| parquet export per block NEFF | 2-15 min (measured in Task 1) | parallel |
| device total, warm | 10-25 min | ~45-60 min |
| analysis, page | | ~2-3 h |

## Review Focus

1. A block NEFF picked by size that is not a block (text encoder, embedder, VAE): the
   classifier would report a meaningless breakdown — Task 1 confirms the block class from the
   parquet `layer` roots and refuses otherwise.
2. Instructions no rule matches: a large `other` share hides the answer — Task 3 fails the
   analysis when `other` exceeds 10% of any engine's busy time.
3. Engine times are busy times of parallel engines and overlap; a reader who adds them gets more
   than the block time — the page draws the block wall time beside them and says so in one line.
4. "PSUM time" does not exist (PSUM is an accumulator memory, not an engine) — the page shows
   PSUM bytes and the PSUM-to-SBUF eviction time (instructions with `psum_read_bytes > 0`).
5. Wan 2.2's block graph equals Wan 2.1's (same dims, same shape): if the block NEFF sha256 is
   identical, the collector reuses Wan 2.1's breakdown and labels it, instead of a second capture.

---

### Task 1: The capture script

**Files:**
- Create: `scripts/profile_dit/capture.sh`
- Create: `scripts/profile_dit/README.md`
- Test: `tests/unit/scripts/test_profile_dit_capture.py`

**Interfaces:**
- Produces: `capture.sh --key <key> --arm <bf16|mxfp8> --cores <list> --cpus <list> --out <dir>`
  writing `<out>/<key>/<arm>/blk<i>/{block.neff,summary.json,pq/trace/*.parquet,capture.log}`,
  `<out>/<key>/<arm>/dump.log`, and the marker `<out>/<key>/<arm>/DONE` (or `FAILED` with the
  reason on its first line).

- [ ] **Step 1: Write the failing test** — drive `capture.sh` with a recording `PY` and a fake
  `neuron-explorer` (as `tests/unit/scripts/test_quant_bench.py` drives `bench.sh`), asserting:
  the generate command carries `--steps 1`, the key's model flags from `bench.sh`, `--quant mxfp8`
  only for the mxfp8 arm, `NEURON_RT_VISIBLE_CORES` and `NEURON_RT_INSPECT_DEVICE_PROFILE=1`;
  `capture` runs once per candidate NEFF with `--collectives-worker-count=<tp>`; a key/arm pair
  outside the table (e.g. `hunyuan_video_15` + `mxfp8`) exits 2 naming the pair.
- [ ] **Step 2: Run it** — `DIFFLET_BACKEND=cpu PYTHONPATH=. ~/venvs/native-fp8/bin/python -m pytest tests/unit/scripts/test_profile_dit_capture.py -q -p no:cacheprovider`; expected FAIL (no script).
- [ ] **Step 3: Implement `capture.sh`.** Model flags per key are sourced from one table in the
  script (copied from `bench.sh`'s keys at the shapes above). Dump: `neuron-explorer inspect -o
  <arm>/dump <PY> -m difflet.cli.main generate ... --steps 1`; poll the rank-0 dump dir (lowest
  pid) and stop the run once the candidate NEFFs have not grown for 60 s after the first DiT
  step's log line, so the VAE decode never runs. Candidates: rank 0's NEFFs larger than 1 MB,
  largest first, at most 4. Capture each with the phase-H flags (`--num-exec=4
  --profile-nth-exec=4`, collectives workers = tp, start id 0), then `view --output-format
  summary-json` and `view --output-format parquet --output-file pq`. Keep the candidate only if
  its `Instruction.parquet` `layer` roots contain the key's block class (`QwenImageTransformerBlock`,
  `FluxTransformerBlock` / `FluxSingleTransformerBlock`, `LTX2VideoTransformerBlock`,
  `WanTransformerBlock`, `HunyuanVideoTransformerBlock` / `HunyuanVideoSingleTransformerBlock`,
  the HV1.5 block class) — the exact class names are read from the first real parquet and pinned
  in the script; fewer kept blocks than the table's count writes `FAILED`.
- [ ] **Step 4: Run the test** — expected PASS.
- [ ] **Step 5: Device smoke on `qwen_image bf16`** (one idle device, checked with `fuser`):
  `DONE` exists, one `blk1` kept, `summary.json` has `total_time`; record the wall time of each
  phase (dump, capture, summary, parquet) in `README.md` — this is the measured row of the time
  budget.
- [ ] **Step 6: Commit and push** — `scripts(profile): capture one DiT block's device profile per model and arm`.

### Task 2: Instruction-to-component classifier

**Files:**
- Create: `scripts/profile_dit/components.py`
- Test: `tests/unit/scripts/test_profile_dit_components.py`

**Interfaces:**
- Produces: `COMPONENTS: tuple[str, ...] = ("modulation", "norm", "qkv_proj", "qk_norm", "rope",
  "attention", "out_proj", "mlp_up", "mlp_act", "mlp_down", "collective", "other")` and
  `classify(layer: str, hlo_name: str, nki_source: str, model: str) -> str` (one of COMPONENTS).

- [ ] **Step 1: Write the failing tests** with real strings from the Task 1 smoke and the
  campaign's HV1 parquet (`/home/ubuntu/fp8-bench/hv1_tp4/profsp_V15/blk1/pq`), e.g.:

```python
assert classify("AdaLayerNormZero[norm1][0]/ColumnParallelLinear[linear][0]_dot", "%dot = dot(...)", "", "hunyuan_video") == "modulation"
assert classify("HunyuanVideoAttention[attn][0]_custom-call.411", "%custom-call.411 = custom-call(...)", ".../attention_cte.py:812", "hunyuan_video") == "attention"
assert classify("", "%all-gather.115 = all-gather(%reshape.114)", "", "hunyuan_video") == "collective"
assert classify("X[y][0]_add.9", "%add.9 = add(...)", "", "qwen_image") == "other"
```

  plus one case per component per model family, written from the first real parquet of each
  key (Task 4 adds them as each key lands).
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement** ordered rules: collectives by HLO opcode (`all-gather`,
  `reduce-scatter`, `all-reduce`, `all-to-all`, `collective-permute`); `attention` by
  `nki_source` naming an attention kernel (`attention_cte`, `scaled_dot_product_attention`,
  `flash`); then the module path in `layer` against a per-model table of `(regex, component)`
  (`to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj` -> `qkv_proj`, `norm_q|norm_k|norm_added_q|
  norm_added_k` -> `qk_norm`, rotary sources `ops_impl/embeddings.py` -> `rope`, `to_out|
  to_add_out|proj_out` -> `out_proj`, MLP up/act/down per model naming, `norm1|norm1_context|
  img_mod|txt_mod|scale_shift_table|AdaLayerNorm` -> `modulation`, other norms -> `norm`); an
  MX linear's kernel instructions (`nki_source` in `matmul_mxfp8`) inherit the linear's component
  from `layer`. Unmatched -> `other`.
- [ ] **Step 4: Run** — expected PASS.
- [ ] **Step 5: Commit and push** — `scripts(profile): classify a block's instructions into DiT components`.

### Task 3: Per-block analysis

**Files:**
- Create: `scripts/profile_dit/analyze.py`
- Test: `tests/unit/scripts/test_profile_dit_analyze.py`

**Interfaces:**
- Consumes: `classify` (Task 2); `blk<i>/summary.json` and `blk<i>/pq/trace/Instruction.parquet` (Task 1).
- Produces: `analyze(block_dir: Path, model: str) -> dict` and the CLI
  `analyze.py <block_dir> --model <key>` writing `<block_dir>/breakdown.json` with:
  `block_ms`; `engines_ms` {tensor, vector, scalar, gpsimd, sync, dma, collective} (summary.json
  `*_active_time`); `memory_bytes` {hbm_read, hbm_write, sbuf_read, sbuf_write, psum_read,
  psum_write, psum_to_sbuf, spill_save, spill_reload}; `mfu`, `mbu`, `throttle_pct`;
  `components` {component: {engine: ms, "hbm_bytes": int, "psum_evict_ms": float}};
  `other_share` {engine: fraction}.

- [ ] **Step 1: Write the failing tests** on a synthetic 6-row instruction DataFrame and a
  minimal summary dict: per-component engine sums equal the hand sums; the sum over components
  of each engine equals that engine's instruction total; `psum_evict_ms` counts only rows with
  `psum_read_bytes > 0`; `analyze` raises `ValueError` naming the engine when `other` exceeds
  10% of an engine's busy time.
- [ ] **Step 2: Run** — expected FAIL.
- [ ] **Step 3: Implement** with pandas `groupby(["component", "engine"])` over
  `duration_ns`; engine names normalised from the parquet's (`Tensor`, `Vector`, `Scalar`/`Act`,
  `GpSimd`, `Sync`) to the keys above.
- [ ] **Step 4: Run** — expected PASS; then run the CLI on the Task 1 smoke's `blk1`: it writes
  `breakdown.json` and `other_share` is below 0.10 for every engine (extend Task 2's rules if
  not, with their tests).
- [ ] **Step 5: Commit and push** — `scripts(profile): per-engine, per-component breakdown of one block NEFF`.

### Task 4: Device runs, all 13 configurations

**Files:**
- Create: `scripts/profile_dit/run_all.sh` (the lane assignment, sourced keys, supervision)
- Modify: `tests/unit/scripts/test_profile_dit_components.py` (one real-string case per new block class)

- [ ] **Step 1:** `fuser /dev/neuron<d>` for d in 0..15; the idle devices are the lanes; check
  `df /` and `df /mnt/nvme`.
- [ ] **Step 2:** One lane per (key, arm) on its own idle device, all 13 at once if 13 devices
  are idle (otherwise queue the rest behind the first lanes to finish), under supervision, with
  the two-slot load lock of the Global Constraints. After each config: `analyze.py` on every kept
  block; a new block class gets its classifier case (Task 2 file) before its breakdown is
  committed.
- [ ] **Step 3:** For `wan22`, compare the block NEFF's sha256 with `wan21`'s per arm; equal ->
  record `same_neff_as: wan21` in its `breakdown.json` and skip the capture.
- [ ] **Step 4:** After each key: commit the key's `breakdown.json` files under
  `docs/verification/assets/2026-11-13-dit-block-profile/<key>/<arm>/blk<i>.json`, push, update
  the page (Task 5's builder, once it exists; until then the status table only).
- [ ] **Step 5:** Expected end state: 13 configs `DONE` (or `same_neff_as`), 19 block
  breakdowns (FLUX and HV1 have two block NEFFs per arm).

### Task 5: Collector, page and evidence doc

**Files:**
- Create: `scripts/profile_dit/collect.py`
- Create: `docs/verification/2026-11-13-dit-block-profile.md`
- Test: `tests/unit/scripts/test_profile_dit_collect.py`

**Interfaces:**
- Consumes: every `breakdown.json` (Task 3/4).
- Produces: `results.json` = `{"configs": [{key, arm, shape, layout, step_ms, step_source,
  blocks_per_step, blocks: [breakdown...], bottleneck, attention_share}], "quality": [...]}` where
  `step_ms` / `step_source` are the evidence doc's numbers and section,
  `bottleneck` = the engine with the largest busy time and its share of `block_ms`, and
  `attention_share` = attention component busy time / block busy time on the bottleneck engine;
  `quality` = the linear-MXFP8 rows of evidence sections 10-12 (vs device bf16 and vs CPU fp32),
  each with its shape and section.

- [ ] **Step 1: Write the failing test** on two synthetic breakdowns: `bottleneck` picks the
  largest engine; `attention_share` is computed on that engine; a config whose step time has no
  evidence row gets `step_ms: null` and `step_source: "not measured"`.
- [ ] **Step 2: Run** — expected FAIL. **Step 3: Implement.** **Step 4: Run** — expected PASS.
- [ ] **Step 5: The page.** Load the `dataviz` skill and call the Artifact quickstart (intent
  `other`) before writing it. Sections: (1) overview table — model, shape, layout, DiT step bf16 /
  MXFP8 and source, block wall time, bottleneck engine, attention share; (2) per model, engine
  busy time bf16 vs MXFP8 as grouped horizontal bars with the block wall time as a reference
  line and the one-line overlap note; (3) per model, per-component stacked bars per engine; (4)
  memory traffic (HBM / SBUF / PSUM / PSUM-to-SBUF / spill bytes) and the PSUM note; (5) the
  quality table; (6) the bottleneck verdict per model in two sentences; (7) method and paths.
  Publish it; record the URL in the evidence doc.
- [ ] **Step 6: Evidence doc** — the same tables in Markdown, the commands, the paths under
  `/mnt/nvme/difflet-prof/`, what was not measured.
- [ ] **Step 7: Commit and push** — `docs(profile): DiT block profile of seven models, bf16 vs linear MXFP8`.
