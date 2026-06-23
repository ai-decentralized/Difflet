# Difflet CLI — Staged Commands Reference

Use `download → compile → generate` separately to verify each step before proceeding.
All logs are saved to `/tmp/logs`.

```bash
mkdir -p /tmp/logs
```

CP support: Flux, Wan, HunyuanVideo, Qwen-Image use `tp=2 cp=2`.
LTX-2 and HunyuanVideo 1.5 do not support CP yet — use `tp=4`.

## Artifact locations

| Step | Where artifacts land |
|---|---|
| `download` | `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/` |
| `compile` (single-process: flux, ltx-2) | `~/.cache/difflet/<model_name>/<hash>/` |
| `compile` (staged models) | `~/.cache/difflet/<stage-specific-dir>/` — see per-model details below |
| `generate` inter-stage tensors | `--work-dir` path (default `~/.cache/difflet/work/<model>/`) |
| `generate` final output | `--output` path |

Override the compile cache root with `DIFFLET_COMPILE_CACHE=<path>` or `--cache-dir <path>`.

---

## Flux (single-process image model)

```bash
# Step 1: Download
# Artifacts: ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/<hash>/
difflet download --model-id black-forest-labs/FLUX.1-dev \
  2>&1 | tee /tmp/logs/flux-download.log

# Step 2: Compile
# Artifacts: ~/.cache/difflet/flux/<hash>/
difflet compile --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 2 --cp-degree 2 \
  --height 1024 --width 1024 \
  2>&1 | tee /tmp/logs/flux-compile.log

# Step 3: Generate
# Output: ~/outputs/flux.png
difflet generate --model-id black-forest-labs/FLUX.1-dev \
  --tp-degree 2 --cp-degree 2 \
  --height 1024 --width 1024 \
  --prompt "a cat sitting on a bench" \
  --output flux.png \
  2>&1 | tee /tmp/logs/flux-generate.log
```

---

## LTX-2 (single-process video model, CP not supported)

```bash
# Step 1: Download
# Artifacts: ~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/<hash>/
difflet download --model-id Lightricks/LTX-2 \
  2>&1 | tee /tmp/logs/ltx2-download.log

# Step 2: Compile
# Artifacts: ~/.cache/difflet/ltx_2/<hash>/
difflet compile --model-id Lightricks/LTX-2 \
  --tp-degree 4 \
  --height 512 --width 768 --num-frames 121 \
  2>&1 | tee /tmp/logs/ltx2-compile.log

# Step 3: Generate
# Output: ~/outputs/ltx2.pt  (video tensor)
difflet generate --model-id Lightricks/LTX-2 \
  --tp-degree 4 \
  --height 512 --width 768 --num-frames 121 \
  --prompt "a cat walking through a garden" \
  --output ltx2.mp4 \
  2>&1 | tee /tmp/logs/ltx2-generate.log
```

---

## Wan 2.2 (2-stage: transformer → vae)

`compile` spawns 2 subprocess stages (transformer @ tp×cp=4 cores, vae @ 1 core).
`generate` spawns the same stages in inference mode.

```bash
# Step 1: Download
# Artifacts: ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/<hash>/
difflet download --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  2>&1 | tee /tmp/logs/wan-download.log

# Step 2: Compile
# Artifacts:
#   ~/.cache/difflet/wan_transformer_tp2cp2_h480w832f9/   (transformer NEFF)
#   ~/.cache/difflet/wan_vae_h480w832f9/                  (VAE NEFF)
difflet compile --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --tp-degree 2 --cp-degree 2 \
  --height 480 --width 832 --num-frames 9 \
  2>&1 | tee /tmp/logs/wan-compile.log

# Step 3: Generate
# Inter-stage tensors (--keep-work-dir): /tmp/logs/wan-work/latents.pt
# Output: ~/outputs/wan.pt  (video tensor)
difflet generate --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --tp-degree 2 --cp-degree 2 \
  --height 480 --width 832 --num-frames 9 \
  --steps 50 --guidance-scale 1.0 --seed 42 \
  --prompt "a cat walking through a garden" \
  --output wan.mp4 \
  --work-dir /tmp/logs/wan-work \
  --keep-work-dir \
  2>&1 | tee /tmp/logs/wan-generate.log
```

---

## HunyuanVideo (3-stage: clip → llama → generate)

`compile` spawns: clip (1 core), llama (tp×cp=4 cores), generate (tp×cp=4 cores) — all with `NEURON_RT_VIRTUAL_CORE_SIZE=2`.

```bash
# Step 1: Download
# Artifacts: ~/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/<hash>/
difflet download --model-id hunyuanvideo-community/HunyuanVideo \
  2>&1 | tee /tmp/logs/hv-download.log

# Step 2: Compile
# Artifacts:
#   ~/.cache/difflet/hunyuan_video_clip/                      (CLIP NEFF, shared across shapes)
#   ~/.cache/difflet/hunyuan_video_llama_seq351/              (Llama NEFF, seq=256+95)
#   ~/.cache/difflet/hunyuan_video_dit_tp2cp2_h320w512f61/   (DiT+VAE NEFF)
difflet compile --model-id hunyuanvideo-community/HunyuanVideo \
  --tp-degree 2 --cp-degree 2 \
  --height 320 --width 512 --num-frames 61 \
  2>&1 | tee /tmp/logs/hv-compile.log

# Step 3: Generate
# Inter-stage tensors (--keep-work-dir):
#   /tmp/logs/hv-work/clip.pt   (CLIP pooled projections)
#   /tmp/logs/hv-work/llama.pt  (Llama encoder hidden states)
# Output: ~/outputs/hunyuan.pt  (video tensor)
difflet generate --model-id hunyuanvideo-community/HunyuanVideo \
  --tp-degree 2 --cp-degree 2 \
  --height 320 --width 512 --num-frames 61 \
  --steps 50 --guidance-scale 6.0 --seed 42 \
  --prompt "a cat sitting on a bench" \
  --output hunyuan.mp4 \
  --work-dir /tmp/logs/hv-work \
  --keep-work-dir \
  2>&1 | tee /tmp/logs/hv-generate.log
```

---

## Qwen-Image (3-stage: text → generate → vae)

`compile` spawns: text (tp×cp=4 cores), generate (tp×cp=4 cores), vae (1 core) — all with `NEURON_RT_VIRTUAL_CORE_SIZE=2`.

```bash
# Step 1: Download
# Artifacts: ~/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots/<hash>/
difflet download --model-id Qwen/Qwen-Image \
  2>&1 | tee /tmp/logs/qwen-download.log

# Step 2: Compile
# Artifacts:
#   ~/.cache/difflet/qwen_image_enc_tp2cp2_seq256/   (text encoder NEFF)
#   ~/.cache/difflet/qwen_image_dit_tp2cp2_h1024w1024/  (DiT NEFF)
#   ~/.cache/difflet/qwen_image_vae_h1024w1024/          (VAE NEFF)
difflet compile --model-id Qwen/Qwen-Image \
  --tp-degree 2 --cp-degree 2 \
  --height 1024 --width 1024 \
  2>&1 | tee /tmp/logs/qwen-compile.log

# Step 3: Generate
# Inter-stage tensors (--keep-work-dir):
#   /tmp/logs/qwen-work/text.pt     (text encoder hidden states)
#   /tmp/logs/qwen-work/latents.pt  (packed DiT latents)
# Output: ~/outputs/qwen.png
difflet generate --model-id Qwen/Qwen-Image \
  --tp-degree 2 --cp-degree 2 \
  --height 1024 --width 1024 \
  --steps 50 --guidance-scale 7.5 --seed 42 \
  --prompt "a cat sitting on a bench" \
  --output qwen.png \
  --work-dir /tmp/logs/qwen-work \
  --keep-work-dir \
  2>&1 | tee /tmp/logs/qwen-generate.log
```

---

## HunyuanVideo 1.5 (compile/generate not yet implemented)

`download` works; `compile` and `generate` raise `NotImplementedError` pending implementation
of the Qwen2.5-VL + ByT5 glyph + image-semantic text encoder pipeline.

```bash
# Step 1: Download (works today)
# Artifacts: ~/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v/snapshots/<hash>/
difflet download --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  2>&1 | tee /tmp/logs/hv15-download.log

# Steps 2 & 3: Not yet available
# difflet compile --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v ...   # raises NotImplementedError
# difflet generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v ...  # raises NotImplementedError
```

---

## Stage / Core Count Reference

| Model | CP support | tp | cp | Subprocess stages | `NEURON_RT_VIRTUAL_CORE_SIZE` | Work-dir tensors |
|---|---|---|---|---|---|---|
| black-forest-labs/FLUX.1-dev | yes | 2 | 2 | none (in-process) | unset | — |
| Lightricks/LTX-2 | no | 4 | 1 | none (in-process) | unset | — |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | yes | 2 | 2 | transformer (4 cores), vae (1 core) | unset | `latents.pt` |
| hunyuanvideo-community/HunyuanVideo | yes | 2 | 2 | clip (1), llama (4), generate (4) | 2 | `clip.pt`, `llama.pt` |
| hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v | no | 4 | 1 | TBD (compile/generate not implemented) | 2 | TBD |
| Qwen/Qwen-Image | yes | 2 | 2 | text (4), generate (4), vae (1) | 2 | `text.pt`, `latents.pt` |
