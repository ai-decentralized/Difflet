# verify_cli parallelism matrix — design

**Date:** 2026-07-05
**Goal:** Rewrite `scripts/verify_cli.py` from a single-config-per-model smoke test into an
on-device parallelism-configuration matrix: 7 models × 4 parallel configs, each config sized
to exactly the 4 NeuronCores of a trn2.3xlarge, with per-cell auto-SKIP for unsupported
combinations, a timed `difflet generate` per supported cell, and a matrix summary.

**Target hardware (verified live):** trn2.3xlarge, instance `i-0c1041a9c3d677b30`, 1 Neuron
device, 4 NeuronCores (logical-neuroncore-config 2), 96 GB device memory, 521 GB free disk.
Fresh box: no `~/.cache/huggingface/hub`, no `~/.cache/difflet`. HF token present (needed for
gated `black-forest-labs/FLUX.1-dev`). `difflet` is an editable install of this repo inside
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` (torch 2.9.1, neuronx-cc 2.25, diffusers
0.38.0, transformers 4.57.6).

## 1. The matrix

### Models (7)

| key | HF model id | shape flags | staged | output artifact |
|---|---|---|---|---|
| `flux` | `black-forest-labs/FLUX.1-dev` | `--height 1024 --width 1024` | no | `flux.png` |
| `qwen_image` | `Qwen/Qwen-Image` | `--height 1024 --width 1024` | yes (text→dit→vae) | `qwen.png` (fallback `qwen.pt`) |
| `ltx_2` | `Lightricks/LTX-2` | `--height 256 --width 384 --num-frames 9` | no | `ltx2.pt` (orchestrator always writes `.pt`) |
| `wan` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `--height 480 --width 832 --num-frames 9` | yes (transformer→vae) | `wan.mp4` (fallback `wan.pt`) |
| `wan2_1` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `--height 480 --width 832 --num-frames 9` | yes (transformer→vae) | `wan21.mp4` (fallback `wan21.pt`) |
| `hunyuan_video` | `hunyuanvideo-community/HunyuanVideo` | `--height 320 --width 512 --num-frames 61` | yes (clip→llama→dit) | `hunyuan.mp4` (fallback `hunyuan.pt`) |
| `hunyuan_video_15` | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | none (fails at compile before shape matters) | scaffold | n/a |

Shapes are held constant per model across all configs (same values the previous verify_cli
used) so cells within a model differ only in parallelism. Steps / guidance-scale / seed are
left at each orchestrator's defaults (flux 28 steps, ltx_2 40, wan 2, hunyuan_video 4,
qwen_image 4; seed 42) — the matrix verifies parallel plumbing, not sample quality, and
within a model the step count is constant so generate timings stay comparable across configs.

Output-artifact notes (verified in orchestrator source):
- `ltx_2` generate always saves `output.with_suffix('.pt')` (`ltx_2.py:112`), so the expected
  artifact is `.pt` even though `--output` names `.mp4`.
- `wan` / `wan2_1` / `hunyuan_video` try MP4 via `diffusers.utils.export_to_video` and fall
  back to `.pt` on export failure; `qwen_image` tries PNG via torchvision and falls back to
  `.pt`. The artifact check accepts **either** suffix (exit 0 + tensor written = inference
  worked; the fallback is a host-side codec concern, not a parallelism failure).

### Parallel configs (4) — all sized to world_size 4 = (2 if cfg else 1)·cp·tp

| key | flags | world_size |
|---|---|---|
| `tp4` | `--tp-degree 4` | 4 |
| `tp2cp2` | `--tp-degree 2 --cp-degree 2` | 4 |
| `tp2cfg` | `--tp-degree 2 --cfg-parallel` | 2 × 2 (CFG doubles) = 4 |
| `tp4sp` | `--tp-degree 4 --sp` | 4 (SP reuses the TP group; no new axis) |

World-size math verified against `DiffletParallelConfig.mesh_spec` / `world_size`
(`difflet/pipeline/parallel_config.py:53-67`).

### Support rules → auto-SKIP

All rules verified in source:

1. **CFG-parallel rejected for guidance-distilled models** — `_DISTILLED_MODELS` in
   `difflet/cli/main.py:153-158`: flux, qwen_image, hunyuan_video, hunyuan_video_15.
   → `tp2cfg` runs only for ltx_2, wan, wan2_1.
2. **SP supported only for** flux, wan (2.2 and 2.1), hunyuan_video — `_SP_SUPPORTED_MODELS`
   in `difflet/cli/main.py:168-173`. → `tp4sp` SKIPs for qwen_image, ltx_2, hunyuan_video_15.
3. **CP unsupported for ltx_2** — `NotImplementedError` at `difflet/models/ltx_2/entry.py:24`
   ("deferred until the M4c transformer spike") — and **for hunyuan_video_15** —
   `difflet/models/hunyuan_video/entry.py` ("CP is deferred until the transformer port").
   → `tp2cp2` SKIPs for both. Note these two rules live in the model entries, **not** in
   `main.py`, so the script must encode them itself.
4. **hunyuan_video_15 is a scaffold** — `HunyuanVideo15Orchestrator.compile()` and
   `.generate()` raise `NotImplementedError` (`difflet/cli/orchestrators/hunyuan_video_15.py:30-42`).
   Its only non-skipped cell (`tp4`) is an **expected failure (XFAIL)**: documented known gap,
   does not fail the run. `download` is implemented and must PASS.

Resulting cells: **28 total = 18 expected PASS + 1 expected XFAIL (hunyuan_video_15/tp4) +
9 SKIP.**

|  | tp4 | tp2cp2 | tp2cfg | tp4sp |
|---|---|---|---|---|
| flux | run | run | SKIP (distilled) | run |
| qwen_image | run | run | SKIP (distilled) | SKIP (no SP) |
| ltx_2 | run | SKIP (no CP) | run | SKIP (no SP) |
| wan | run | run | run | run |
| wan2_1 | run | run | run | run |
| hunyuan_video | run | run | SKIP (distilled) | run |
| hunyuan_video_15 | run → XFAIL | SKIP (no CP) | SKIP (distilled) | SKIP (no SP) |

## 2. Per-model flow

```
difflet download --model-id <id>                      # once per model, untimed
  verify: exit 0 AND HF snapshot glob matches
          ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/*
for each supported config:
    difflet compile  --model-id <id> <config flags> <shape flags> [--cache-dir …]
      verify: exit 0            (duration recorded, informational)
    difflet generate --model-id <id> <config flags> <shape flags> [--cache-dir …]
                     --prompt <fixed prompt> --output <cell dir>/<artifact>
                     [--work-dir <cell dir>/work --keep-work-dir   # staged models]
      verify: exit 0 AND output artifact exists — TIMED (the headline number)
```

- Download failure ⇒ all of that model's cells SKIP with reason `download failed`.
- Compile failure ⇒ that cell's generate SKIPs; other cells of the model still run.
- Cells run sequentially (the 4 configs each need all 4 cores; nothing can share).
- Model order: flux, qwen_image, ltx_2, wan, wan2_1, hunyuan_video, hunyuan_video_15.
- Because generate always runs immediately after its own compile in the same cell, the
  compile cache is warm by construction: generate timing is pure inference (load + run),
  never compilation.

### wan / wan2_1 compiled-artifact collision

`WanOrchestrator._stage_compiled_dir` (`wan.py:210-223`) names stage dirs
`wan_transformer_tp{tp}cp{cp}{cfg}{sp}_h{h}w{w}f{f}` / `wan_vae_h{h}w{w}f{f}` — the model
version is **not** part of the name, so Wan 2.2 and Wan 2.1 at the same config would collide
in `~/.cache/difflet`. Mitigation: `wan2_1` passes `--cache-dir ~/.cache/difflet/wan2_1`
(supported by both compile and generate) so both models keep valid artifacts side by side and
each generate provably loads what its own compile produced. Recorded as a repo caveat worth a
follow-up fix in difflet itself (out of scope here).

## 3. Script architecture (approach)

Approaches considered:

- **A (chosen): declarative tables + generic cell runner.** `MODELS: dict[str, ModelSpec]`
  and `PARALLEL_CONFIGS: dict[str, ParallelConfig]` (dataclasses), skip rules as module-level
  frozensets (`DISTILLED`, `SP_SUPPORTED`, `CP_UNSUPPORTED`, `EXPECTED_FAIL_CELLS`) with
  comments pointing at the difflet source of truth, plus **drift-guard unit tests** that
  import `difflet.cli.main` and assert the script's sets match `_DISTILLED_MODELS` /
  `_SP_SUPPORTED_MODELS`. The script itself stays a pure subprocess driver (no difflet
  import at runtime).
- B: extend the old per-model `MODEL_CONFIGS` with 28 inlined entries — rejected: massive
  duplication, skip logic buried in data, unreadable summary code.
- C: import difflet inside the script to derive rules at runtime — rejected: couples the
  runner to the heavyweight import environment and hides rule provenance; the drift-guard
  tests give the same protection without the coupling.

### Components

- `ModelSpec` — model_id, hf snapshot glob, shape flags, prompt, output filename +
  accepted fallback suffixes, staged flag, optional cache_dir override.
- `ParallelConfig` — key, CLI flags, world_size (asserted == 4 at import time).
- `plan_cells()` — pure function `(models, configs) → list[Cell]` where each Cell is
  RUN / SKIP(reason) / RUN-XFAIL. Unit-testable without any subprocess.
- `run_cell()` — compile step then timed generate step via `subprocess.run`, per-cell log
  file, returns statuses + durations.
- `main()` — arg parsing (`--models`, `--configs`, `--step-timeout`), download loop, cell
  loop, summary rendering, `results.json`, exit code.

### Statuses

`PASS`, `FAIL`, `SKIP` (with reason), `XFAIL` (expected failure that failed — the
hunyuan_video_15 known gap), `XPASS` (expected failure that unexpectedly passed — flagged).
Exit code 0 iff no `FAIL` and no `XPASS`.

### Outputs

- Run root: `/tmp/logs/verify_matrix_<YYYYMMDD_HHMMSS>/`
  - `main.log` — banners, per-step one-liners, final summary.
  - `<model>/<config>/step_compile.log`, `step_generate.log` — full subprocess output
    (Neuron compile logs are enormous; one file per step keeps them navigable).
  - `<model>/<config>/<artifact>` and `work/` for staged models (`--keep-work-dir`).
  - `results.json` — machine-readable: per-model download status/duration, per-cell
    statuses, compile/generate durations, cmds, reasons.
- Console: progress lines plus a final matrix table:

```
Model              tp4              tp2cp2           tp2cfg           tp4sp
-----------------  ---------------  ---------------  ---------------  ---------------
flux               PASS 41.2s       PASS 44.0s       SKIP distilled   PASS 39.8s
...
hunyuan_video_15   XFAIL scaffold   SKIP no-CP       SKIP distilled   SKIP no-SP
```

plus a DOWNLOADS section, FAILED COMMANDS detail (cmd, reason, log path), and artifact
locations. Cell durations shown are the timed generate; compile durations live in
`results.json` and `main.log`.

### CLI

- `--models <keys…>` / `--configs <keys…>` — subset filters (default: all). Doubles as the
  manual resume mechanism if a multi-hour run dies partway (compile caches persist, so
  re-running a completed model is cheap for flux/ltx_2 and correct for all).
- `--step-timeout <seconds>` — per-step `subprocess.run` timeout, default 14400 (4 h);
  a hung compile must not stall the remaining cells. Timeout ⇒ FAIL with reason `timeout`.

## 4. Testing

Rewrite `tests/unit/cli/test_verify_cli.py` (imports of `MODEL_CONFIGS` etc. break) — all
pure-logic, no device, no subprocess:

- matrix shape: 7 models, 4 configs, every config world_size == 4;
- `plan_cells()` produces exactly the support table above (each SKIP with its reason,
  hunyuan_video_15/tp4 marked expected-fail);
- drift guards: script sets == `difflet.cli.main._DISTILLED_MODELS` / `_SP_SUPPORTED_MODELS`;
  script model ids ⊆ `VALID_MODELS`;
- command building: flags per config, `--cache-dir` for wan2_1, `--work-dir/--keep-work-dir`
  only for staged models, ltx_2 artifact expectation is `.pt`;
- `run_cell()` with mocked `subprocess.run`: PASS path, compile-fail ⇒ generate SKIP,
  generate timing captured, timeout ⇒ FAIL;
- summary/exit code: XFAIL doesn't fail the run, FAIL/XPASS do.

## 5. Live run plan

Environment: `PATH` includes `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin`.
Expected duration: many hours (≈300 GB of downloads + 19 cold Neuron compiles + 19
generates) — run in the background, monitor via `main.log` / `results.json`, report the final
matrix with timings. Disk (521 GB free) is expected to fit but is logged per model in
`main.log` (`shutil.disk_usage`). Known expected outcome: 18 PASS, 1 XFAIL
(hunyuan_video_15/tp4, NotImplementedError scaffold), 9 SKIP.

## 6. Out of scope

- Fixing the wan/wan2_1 compiled-dir collision inside difflet (documented above).
- Numerical parity between configs (covered by `scripts/*_parity_smoke.py`).
- hunyuan_video_15 stage implementation (tracked scaffold gap).
- Multi-cell concurrency (only 4 cores; cells are exclusive by definition).
