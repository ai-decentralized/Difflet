# CLI Verification Script Design

**Date:** 2026-06-23
**Status:** Approved

---

## Goal

After making changes to the Difflet codebase, run a single script that exercises every supported model's full `download → compile → generate` pipeline via the `difflet` CLI, reports pass/fail per step, verifies expected artifacts exist on disk, and writes a summary log you can inspect or share.

---

## Invocation

```bash
python scripts/verify_cli.py                        # all 5 models
python scripts/verify_cli.py --models flux wan      # subset
```

Valid `--models` values: `flux`, `ltx_2`, `wan`, `hunyuan_video`, `qwen_image`

---

## Output

Everything (command banners, stdout/stderr from each `difflet` call, artifact check results, summary table) is written to a single timestamped log file:

```
/tmp/logs/verify_cli_<YYYYMMDD_HHMMSS>.log
```

The terminal receives only:
- The log file path at start
- One progress line per step as it begins (`[flux] download ...`)
- The log file path again at the end

---

## Model Configs

Each model entry defines:
1. **Steps** — ordered list of `(step_name, difflet_args_list)` triples
2. **Artifact globs** — per step, a list of `pathlib` glob patterns; at least one match required per pattern
3. **Generate output path** — the `--output` value passed to `difflet generate`

### Step commands (exact CLI flags)

| Model | download | compile | generate |
|---|---|---|---|
| flux | `--model-id black-forest-labs/FLUX.1-dev` | `+ --tp-degree 2 --cp-degree 2 --height 1024 --width 1024` | `+ --prompt "a cat sitting on a bench" --output flux.png` |
| ltx_2 | `--model-id Lightricks/LTX-2` | `+ --tp-degree 4 --height 512 --width 768 --num-frames 121` | `+ --prompt "a cat walking through a garden" --output ltx2.mp4` |
| wan | `--model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `+ --tp-degree 2 --cp-degree 2 --height 480 --width 832 --num-frames 9` | `+ --steps 50 --guidance-scale 1.0 --seed 42 --prompt "a cat walking through a garden" --output wan.mp4 --keep-work-dir` |
| hunyuan_video | `--model-id hunyuanvideo-community/HunyuanVideo` | `+ --tp-degree 2 --cp-degree 2 --height 320 --width 512 --num-frames 61` | `+ --steps 50 --guidance-scale 6.0 --seed 42 --prompt "a cat sitting on a bench" --output hunyuan.mp4 --keep-work-dir` |
| qwen_image | `--model-id Qwen/Qwen-Image` | `+ --tp-degree 2 --cp-degree 2 --height 1024 --width 1024` | `+ --steps 50 --guidance-scale 7.5 --seed 42 --prompt "a cat sitting on a bench" --output qwen.png --keep-work-dir` |

### Artifact globs per step

**download** — same pattern for all models, derived from model ID:
```
~/.cache/huggingface/hub/models--<org>--<name>/snapshots/*/
```

**compile** artifacts:

| Model | Globs (all must match ≥1 file/dir) |
|---|---|
| flux | `~/.cache/difflet/flux/*/` |
| ltx_2 | `~/.cache/difflet/ltx_2/*/` |
| wan | `~/.cache/difflet/wan_transformer_tp2cp2_*/` and `~/.cache/difflet/wan_vae_*/` |
| hunyuan_video | `~/.cache/difflet/hunyuan_video_clip/` and `~/.cache/difflet/hunyuan_video_llama_*/` and `~/.cache/difflet/hunyuan_video_dit_*/` |
| qwen_image | `~/.cache/difflet/qwen_image_enc_*/` and `~/.cache/difflet/qwen_image_dit_*/` and `~/.cache/difflet/qwen_image_vae_*/` |

**generate** — check `--output` file exists (written into the model's work dir).

Inter-stage tensors (`--work-dir`) go to `/tmp/logs/verify_<model>/` with `--keep-work-dir` so they persist for inspection.

---

## Step Runner

`run_step(model, step, cmd, artifact_globs, log_fh)` → `StepResult`

1. Write banner to log: `=== [<model>] <step> === <timestamp>`
2. Write full command to log
3. Run via `subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)`
4. On non-zero exit → `FAIL(reason="exit code N")`
5. On exit 0 → check each glob; if any matches zero results → `FAIL(reason="artifact not found: <glob>")`
6. All globs satisfied → `PASS`

**Failure propagation:** If a step returns `FAIL`, all subsequent steps for that model are recorded as `SKIP` without running.

Step statuses: `PASS` | `FAIL` | `SKIP`

---

## Summary Table (end of log)

```
=== SUMMARY ===

Model              download   compile    generate
-----------------  ---------  ---------  ---------
flux               PASS       PASS       PASS
ltx_2              PASS       FAIL       SKIP
wan                PASS       PASS       PASS
hunyuan_video      PASS       PASS       FAIL
qwen_image         SKIP       SKIP       SKIP

FAILED COMMANDS:
  [ltx_2] compile
    cmd:  difflet compile --model-id Lightricks/LTX-2 --tp-degree 4 ...
    why:  exit code 1
    log:  /tmp/logs/verify_cli_20260623_143022.log

  [hunyuan_video] generate
    cmd:  difflet generate --model-id hunyuanvideo-community/HunyuanVideo ...
    why:  artifact not found: ~/.cache/difflet/hunyuan_video_dit_*/
    log:  /tmp/logs/verify_cli_20260623_143022.log

ARTIFACT LOCATIONS:
  flux           download   ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/
  flux           compile    ~/.cache/difflet/flux/*/
  flux           generate   /tmp/logs/verify_flux/flux.png
  ltx_2          download   ~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/*/
  ltx_2          compile    ~/.cache/difflet/ltx_2/*/
  ltx_2          generate   /tmp/logs/verify_ltx_2/ltx2.mp4
  wan            download   ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/*/
  wan            compile    ~/.cache/difflet/wan_transformer_tp2cp2_*/ | ~/.cache/difflet/wan_vae_*/
  wan            generate   /tmp/logs/verify_wan/wan.mp4
  hunyuan_video  download   ~/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/*/
  hunyuan_video  compile    ~/.cache/difflet/hunyuan_video_clip/ | hunyuan_video_llama_*/ | hunyuan_video_dit_*/
  hunyuan_video  generate   /tmp/logs/verify_hunyuan_video/hunyuan.mp4
  qwen_image     download   ~/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/*/
  qwen_image     compile    ~/.cache/difflet/qwen_image_enc_*/ | qwen_image_dit_*/ | qwen_image_vae_*/
  qwen_image     generate   /tmp/logs/verify_qwen_image/qwen.png
```

---

## File Layout

```
scripts/
└── verify_cli.py    # new file, ~200 lines
```

No changes to any existing source files.

---

## Non-Goals

- No timeout enforcement (steps run to completion)
- No parallel model execution (Neuron cores are shared; models run sequentially)
- No HunyuanVideo 1.5 (compile/generate not implemented)
- No dry-run mode
