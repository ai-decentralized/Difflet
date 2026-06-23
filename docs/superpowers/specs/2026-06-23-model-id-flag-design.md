# Design: Replace `--model` with `--model-id` (HuggingFace IDs)

**Date:** 2026-06-23
**Status:** Approved

---

## Goal

Replace the short model names accepted by `--model` (`flux`, `wan`, etc.) with the full
HuggingFace model IDs (`black-forest-labs/FLUX.1-dev`, etc.), and rename the flag from
`--model` to `--model-id`. Short names are removed entirely — no aliases.

---

## Valid `--model-id` Values

| `--model-id` | Orchestrator | CP support |
|---|---|---|
| `black-forest-labs/FLUX.1-dev` | FluxOrchestrator | yes (tp=2 cp=2) |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | WanOrchestrator | yes (tp=2 cp=2) |
| `hunyuanvideo-community/HunyuanVideo` | HunyuanVideoOrchestrator | yes (tp=2 cp=2) |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | HunyuanVideo15Orchestrator | no (tp=4) |
| `Qwen/Qwen-Image` | QwenImageOrchestrator | yes (tp=2 cp=2) |
| `Lightricks/LTX-2` | LTX2Orchestrator | no (tp=4) |

---

## Files Changed

### `difflet/cli/main.py`
- `VALID_MODELS` — replace short names with HF IDs
- `_MODEL_TYPE` — re-key by HF ID (values unchanged: `flux`, `wan`, etc.)
- `_add_model_flag()` — argument name becomes `--model-id`, dest `model_id`
- `_get_orchestrator()` — re-key mapping by HF ID
- `main()` — validation check uses `args.model_id` instead of `args.model`
- Error message updated to list HF IDs

### `difflet/cli/stage.py`
- `_ORCHESTRATOR_MAP` — re-keyed by HF ID
- `_build_stage_parser()` — `--model` argument renamed to `--model-id`

### `difflet/cli/orchestrators/wan.py`
- `_shared_cli_args()` — passes `"--model-id", _HF_MODEL_ID` instead of `"--model", "wan"`

### `difflet/cli/orchestrators/hunyuan_video.py`
- `_shared_cli_args()` — passes `"--model-id", _HF_MODEL_ID`

### `difflet/cli/orchestrators/qwen_image.py`
- `_shared_cli_args()` — passes `"--model-id", _HF_MODEL_ID`

### `docs/cli-staged-commands.md`
- All `--model <name>` occurrences updated to `--model-id <hf-id>`

---

## Error Handling

Unknown `--model-id` exits with code 1:
```
Error: Unknown model-id 'foo'. Valid model IDs:
  black-forest-labs/FLUX.1-dev
  Lightricks/LTX-2
  Qwen/Qwen-Image
  Wan-AI/Wan2.2-T2V-A14B-Diffusers
  hunyuanvideo-community/HunyuanVideo
  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
```

---

## What Does Not Change

- Orchestrator internals (`_HF_MODEL_ID`, `_MODEL_TYPE`, stage logic) — untouched
- `runner.py` — untouched
- Registry, compile cache, `DiffletPipeline` — untouched
- `_CLI_NAME` constants inside each orchestrator — kept for work-dir path naming only
