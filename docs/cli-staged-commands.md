# Difflet CLI — Staged Commands Reference

Use `download → compile → generate` separately to verify each step before proceeding.
All logs are saved to `/tmp/logs`.

```bash
mkdir -p /tmp/logs
```

CP support: Flux, Wan, HunyuanVideo, Qwen-Image use `tp=2 cp=2`.
LTX-2 does not support CP — use `tp=4`. HunyuanVideo 1.5 supports CP (ulysses, gather_kv) but
defaults to `tp=2 --cfg-parallel` (true CFG).

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

### Qwen-Image in FP8 (PyTorch-native `neuron` backend, Trainium3 only)

`--quant mxfp8` quantizes the DiT's attention q/k/v and MLP linears to OCP MXFP8 (the attention
output projections stay bf16: measured faster and closer to bf16 that way) (fp8 e4m3 weights
with one shared scale per 32 values, the activation quantized inside nkilib's `matmul_mxfp8`
kernel); no calibration file. `compile` writes the quantized checkpoint once under the cache
(`<cache-dir>/quantized/<model>/transformer/mxfp8-<id>/`, 27 GB for 40 GB of bf16) and
`generate` reads it. Measured on trn3 at tp4, 1024x1024, 50 steps: DiT step 239 ms against
299 ms in bf16 (0.80x); against the bf16 image, PSNR 26.8–39.2 dB over 4 prompts x 3 seeds
(mean 32.5), SSIM 0.955–0.990, LPIPS 0.014–0.092; against diffusers fp32 on the CPU (one
image) the mxfp8 image is 37.36 dB / SSIM 0.992, 1.02 dB closer than bf16's 36.34 dB / 0.990
(`docs/verification/2026-11-07-mxfp8-video-models.md` section 10). The
flag is refused on the AoT `trainium` backend, on trn2, under `--exec-mode eager`, with
`--teacache-speedup`, and for models other than Qwen-Image, FLUX.1-dev, LTX-2, Wan 2.1,
Wan 2.2-A14B, HunyuanVideo and HunyuanVideo-1.5 (HunyuanVideo-1.5 only with
`DIFFLET_QUANT_TARGETS`: it has no default set, below). **Do not combine `--quant mxfp8` with
`--sp`**: measured on HunyuanVideo (tp4, 320x512x61), 1263.8 ms per step against 659.4 ms for
bf16 with SP (1.92x); SP with MXFP8 has not been measured on the other models. Use the model's
default layout or CP2 instead, as measured: HunyuanVideo `--tp-degree 4 --cp-degree 2 --cp-mode
gather_kv`, HunyuanVideo-1.5 `--tp-degree 2 --cfg-parallel --cp-degree 2` (8 cores each; for
HunyuanVideo-1.5, CP2 with MXFP8 is itself 1.7-1.8x bf16's, below). `--quant-compute
dequant` stores the fp8 weights and computes in bf16 (memory only, for debugging). MXFP8 is the
only recipe: a per-tensor static fp8 variant was built and measured first (see the evidence
doc) and removed, because its kernel path is unusable in the compiled block graph on
neuronx-cc 2.28 and it needs calibration.

FLUX.1-dev accepts `--quant mxfp8` on the same path (`difflet compile/generate --model-id
black-forest-labs/FLUX.1-dev --backend neuron --tp-degree 4 --quant mxfp8`): the dual-stream
blocks' q/k/v and MLPs and the single-stream blocks' q/k/v, `proj_mlp` and `proj_out` (stored as
its attention and MLP halves, each quantized per rank K range) run in mxfp8; the dual blocks'
output projections stay bf16. Measured on trn3 (tp4, 1024x1024, 50 steps): DiT step 183 ms
against 222 ms in bf16 (0.82x), deterministic; at 28 steps the image is 34.7 dB / SSIM 0.978
from bf16; at 50 steps the sample drifts in composition (over 4 prompts x 3 seeds mxfp8 is
24.7 dB from bf16 on average, two bf16 runs at different TP degrees 29.3 dB); against diffusers
fp32 on the CPU the one reference image is 21.30 dB from mxfp8 and 23.36 dB from bf16, a 2.06 dB
gap that one image of this sampler cannot resolve. Evidence:
`docs/verification/2026-11-07-mxfp8-video-models.md` sections 6 and 10 (the earlier "0.99x"
predates the kernel's fast-DMA load path).

LTX-2 takes the flag on the same path (`difflet compile/generate --model-id Lightricks/LTX-2
--backend neuron --tp-degree 4 --height 512 --width 768 --num-frames 121 --quant mxfp8`): the
video stream's attention q/k/v (self and text cross-attention), the audio-to-video query, the
video-to-audio keys/values and the video MLP run in mxfp8; the audio stream (126 tokens at that
shape), every output projection, the modulation and the embedders stay bf16. Measured on trn3
(tp4, 512x768x121, 50 steps, seed 42): DiT step 966 ms against 1284 ms in bf16 (0.75x),
deterministic; the quantized DiT is 24 GB for 37.8 GB of bf16; against the bf16 clip PSNR
22.5 dB / SSIM 0.83 / LPIPS 0.078, where two bf16 runs at different TP degrees are 25.9 dB /
0.91 / 0.040 apart; against diffusers fp32 on the CPU both clips are far off at 121 frames
(mxfp8 16.85 dB, bf16 16.55 dB), mxfp8 no further than bf16. `DIFFLET_QUANT_TARGETS` can add the
audio linears (no step time to gain, measured). Evidence:
`docs/verification/2026-11-07-mxfp8-video-models.md` (sections 2-3 and 10).

HunyuanVideo takes it as well (`difflet compile/generate --model-id
hunyuanvideo-community/HunyuanVideo --backend neuron --tp-degree 4 --height 320 --width 512
--num-frames 61 --quant mxfp8`): the hub DiT is bf16, so `compile` quantizes straight from the
snapshot (18 GB for 24 GB of bf16; 4.85 GB of DiT weights per rank at tp4 instead of 6.54). The
dual-stream blocks' text q/k/v (`add_q/k/v_proj`) and both MLPs, and the single-stream blocks'
`proj_mlp` and `proj_out` (stored as its attention and MLP halves, each quantized per rank K
range) run in mxfp8; the image-token q/k/v (the dual blocks' `to_q/k/v` and the single blocks'
joint q/k/v), the dual blocks' output projections, the modulation, the token refiner and the
embedders stay bf16 (with the image-token q/k/v quantized the clip is 21.2 dB from bf16 instead
of 31.1, measured). Measured on trn3 (tp4, 50 steps, seed 42, guidance 6.0): 0.96x at 320x512x61
(635 against 660 ms per step) and 0.97x at 544x960x129 (17968 against 18494 ms), deterministic;
against the bf16 clip 31.1 dB / SSIM 0.938 / LPIPS 0.050 at 320x512x61, where two bf16 runs at
different TP degrees are 36.6 dB apart, and 26.5 dB / 0.907 / 0.128 at 544x960x129 (no floor
measured there); against diffusers fp32 on the CPU at 320x512x61 both clips are far off (mxfp8
20.59 dB, bf16 20.79 dB), mxfp8 0.20 dB further (544x960x129 not referenced; section 10). CP2 (`--cp-degree 2
--cp-mode gather_kv`, 403 ms per step on 8 cores) and `--dp 2` run as at tp4 (`--dp 2` when its
replicas may start at core 0: the router ignores an inherited `NEURON_RT_VISIBLE_CORES`, an open
item on `main`); with `--sp` the mxfp8 step is 1.92x the bf16 one under SP (1264 against 659 ms:
the dual block's rotary runs as element-wise DMAs, although in the default its q/k come from the
bf16 image projections), an open item whose cause is not known: do not combine them (above).
720x1280x129 does not load at tp4 or tp4 x CP2 on trn3 (HBM), and bf16 at tp2 does not fit in HBM at
544x960x129, so that shape has no bf16 floor. Same evidence document, section 7.

Wan 2.1 takes the flag too (`difflet compile/generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers
--backend neuron --tp-degree 4 --height 480 --width 832 --num-frames 81 --quant mxfp8`): the hub
DiT is fp32, so `compile` makes the bf16 copy first and quantizes from it. The text
cross-attention's q/k/v and the FFN run in mxfp8; the self-attention q/k/v, every output
projection, the modulation and the embedders stay bf16. Measured on trn3 (tp4, 50 steps,
seed 42): 0.89x at 480x832x9 (393.1 against 440.0 ms per step) and 0.98x at 480x832x81 (4956
against 5053 ms, measured before bf16 Wan's rotary went back to the bit-reproducible stride-2
law); per-rank DiT weights 5.36 GB instead of 7.50; against the bf16 clip 32.1 dB / SSIM 0.934 /
LPIPS 0.042 on average over 4 prompts x 3 seeds at 9 frames and 28.8 dB on the 81-frame clip,
where two bf16 runs at different TP degrees are 37 dB apart; against diffusers fp32 on the CPU at
9 frames the mxfp8 clip is 33.72 dB / SSIM 0.947 where bf16's is 36.08 / 0.952, 2.36 dB further
(81 frames not referenced; section 10). `--quant` is refused at `--tp-degree 8` (the FFN down projection's
1728-wide K per rank does not fit the MX layout).

A `DIFFLET_QUANT_TARGETS` override that quantizes the self-attention q/k too (for instance
`'blocks\.\d+\.attn1\.(to_q|to_k|to_v);blocks\.\d+\.attn2\.(to_q|to_k|to_v);blocks\.\d+\.ffn\.net\.0\.proj;blocks\.\d+\.ffn\.net\.2'`,
on both `compile` and `generate`) makes the application reorder every self-attention head's q/k
channels at load (weights, block scales, bias and the q/k norms; the log line
`[wan] self-attention q/k reordered for the half-split rotary (40 attentions)`), so that the
rotary turns contiguous halves: in the interleaved layout the pair swap moved the kernel's q/k
element by element and made the tp4 block 5x slower (measured). Only with that override: 0.77x at
480x832x9 and 0.94x at 480x832x81 for Wan 2.1, 4.60 GB per rank; but with both Wan 2.2 experts
it is 1.44 dB / SSIM 0.034 further from bf16 than the default over 12 clips (27.6 against
29.0 dB), past the default rule, which is why the default leaves the self-attention bf16. Same
evidence document, sections 4 and 8.

Wan 2.2-A14B the same way, with `--dual-expert`: `compile` makes both experts' bf16 copies and
one quantized directory per expert with one spec (two `[wan] MXFP8 transformer… (nki)` lines).
Measured with both experts (tp4, 50 steps): 0.89x at 480x832x9 (391.8 against 439.6 ms) and
0.98x at 480x832x81 (4985 against 5080 ms, before the rotary change), 29.0 dB / 0.895 / 0.084
from bf16 over the same 12 clips, both experts resident at 10.7 GB per rank instead of 15.0; at
81 frames the two experts' hand-off makes the clip very sensitive to any perturbation (two bf16
runs at different TP degrees are 28.7 dB apart; the mxfp8 clip is 20.0 dB from bf16 with the
same scene composed differently); against diffusers fp32 on the CPU at 9 frames the mxfp8 clip
is 33.46 dB / SSIM 0.949 where bf16's is 36.39 / 0.962, 2.93 dB further (81 frames not
referenced; section 10). With
the self-attention override above both experts reorder (the log line twice): 0.76x at 9 frames
and 0.93x at 81. Sections 5 and 8.

HunyuanVideo-1.5 has **no MXFP8 default** (the user's decision, 2026-11-09, on the numbers
below): on the native path `--quant mxfp8` is refused (exit 2, before anything is written)
unless `DIFFLET_QUANT_TARGETS` names the layers, from the four measured groups in
`difflet/quant/targets.py` (`HUNYUAN_VIDEO_15_GROUPS`). The hub DiT is fp32, so `compile` makes
the bf16 copy first and quantizes from it. The measured set, the former default, is the
dual-stream blocks' image q/k/v (`attn.to_q/k/v`) and image MLP (`ff`) in mxfp8, at the default
480x848x121 and tp2 x CFG2:

```bash
export DIFFLET_QUANT_TARGETS='transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v);transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)'
DIFFLET_BACKEND=neuron difflet compile --model-id \
  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --quant mxfp8
DIFFLET_BACKEND=neuron difflet generate --model-id \
  hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --quant mxfp8 \
  --steps 50 --seed 42 --prompt "a cat walking through a garden" --output hv15-mxfp8.mp4
```

In it the text stream's q/k/v (`attn.add_q/k/v_proj`) and MLP (`ff_context`), the
output projections, the modulation, the token refiner and the embedders stay bf16 (with the image
and text q/k/v both quantized the step is 2.7x bf16's at 480x848x9, and 2.10x at 121 frames
with every group, measured). Measured on trn3 (tp2 x CFG2,
480x848x121, 50 steps, seed 42, true CFG 6.0): DiT step 10032 against 9924 ms in bf16 (1.011x,
one generate, on the tree that reverted phase D's HunyuanVideo-1.5 rotary op D4; the set
was chosen with D4: 10314 against 10241 ms, 1.007x, deterministic); DiT weights per rank
6.97 GiB resident instead of 7.80, but no lower peak HBM (with D4 17.66 against 17.56 GiB: the
runtime's scratchpad grows by 0.94 GiB; without it the set peaks at 18.14 GiB, bf16 not
measured); with `--tp-degree 2 --cfg-parallel --cp-degree 2` (8 cores; `--cp-degree 2` alone
runs tp1 x CP2 on 2 cores) the step is 1.7-1.8x bf16's under CP2 (with D4: 10198 / 9539
against 5687 / 5682 ms). Against the bf16 clip 12.65 dB / SSIM 0.183 / LPIPS 0.504 (with D4
12.76 dB / SSIM 0.190 / LPIPS 0.510), where two bf16 runs at different TP degrees
are 15.90 dB / 0.312 / 0.363 apart (at 121 frames the clip with every group quantized differs
from bf16's in composition; the evidence's montage does not show this set's); against
diffusers fp32 on the CPU at 480x848x9 its clip is 17.41 dB / SSIM 0.446 where bf16's
is 24.89 / 0.846, 7.47 dB further (the image MLP alone: 20.20 dB, 4.69 dB further; the image
q/k/v alone: 18.10 dB; section 10). The set misses the device screen's SSIM threshold (0.212)
and at 121 frames gains neither speed nor peak HBM, hence no default
([issue #145](https://github.com/ai-decentralized/Difflet/issues/145)). Same evidence document,
section 9.

```bash
DIFFLET_BACKEND=neuron difflet compile --model-id Qwen/Qwen-Image \
  --tp-degree 4 --height 1024 --width 1024 --quant mxfp8 \
  2>&1 | tee /tmp/logs/qwen-compile-mxfp8.log
DIFFLET_BACKEND=neuron difflet generate --model-id Qwen/Qwen-Image \
  --tp-degree 4 --height 1024 --width 1024 --quant mxfp8 \
  --steps 50 --seed 42 --prompt "a cat sitting on a bench" --output qwen-mxfp8.png \
  2>&1 | tee /tmp/logs/qwen-generate-mxfp8.log
```

---

## HunyuanVideo 1.5 (3-stage: encode → aot_transformer → aot_vae)

On the Trainium (AOT) backend, `generate` spawns: `encode` (host, one process: Qwen2.5-VL + ByT5
prompt and negative-prompt embeddings), `aot_transformer` (one NxD process on all
`tp × cp (×2 with --cfg-parallel)` cores, `NEURON_RT_VIRTUAL_CORE_SIZE=2`) and `aot_vae`. The VAE
decodes **on the host by default** (diffusers' tiled decoder in fp32; about 7 min at 320×512×61,
about 45 min at 480×848×121), because the compiled `vae15` decoder does not build on neuronx-cc
2.26; `--device-vae` opts into it, and `compile` then compiles it too. `compile` builds only the
DiT. HunyuanVideo 1.5 runs true CFG (6.0), and `--tp-degree 2 --cfg-parallel` is its default
layout. `--cp-mode ring` and a multi-shape `--shapes` are refused before any stage launches.
Verified layouts and numbers: `docs/verification/hv15-aot-evidence.md`. On the PyTorch-native
`neuron` backend (Trainium3), `--quant mxfp8` runs part of the DiT in MXFP8: see the HunyuanVideo-1.5
paragraph under "Qwen-Image in FP8" above.

```bash
# Step 1: Download
# Artifacts: ~/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v/snapshots/<hash>/
difflet download --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  2>&1 | tee /tmp/logs/hv15-download.log

# Step 2: Compile (DiT only; ~8 min at 320x512x61, ~66 min at 480x848x121)
# Artifacts:
#   ~/.cache/difflet/hunyuan_video_15_aot_dit/<hash>/   (DiT NEFF + sharded weights; the
#                                                        TeaCache probe joins it on first adaptive use)
difflet compile --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --tp-degree 2 --cfg-parallel \
  --height 320 --width 512 --num-frames 61 \
  2>&1 | tee /tmp/logs/hv15-compile.log

# Step 3: Generate
# Inter-stage tensors (--keep-work-dir):
#   /tmp/logs/hv15-work/embeds.pt   (prompt + negative-prompt embeddings and masks)
#   /tmp/logs/hv15-work/latents.pt  (denoised latents)
# Output: hv15.mp4
difflet generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --tp-degree 2 --cfg-parallel \
  --height 320 --width 512 --num-frames 61 \
  --steps 50 --seed 42 \
  --prompt "a cinematic shot of a red fox running through a snowy forest" \
  --output hv15.mp4 \
  --work-dir /tmp/logs/hv15-work \
  --keep-work-dir \
  2>&1 | tee /tmp/logs/hv15-generate.log
```

Other verified layouts: `--tp-degree 4`, `--tp-degree 4 --sp` (not at 480×848×121),
`--tp-degree 2 --cp-degree 2 --cp-mode ulysses|gather_kv`, `--tp-degree 2 --dp 2`.
TeaCache: `--teacache-cadence 2` is the recommended mode (20 of 50 steps skipped);
`--teacache-online-delta` works; adaptive `--teacache-speedup` is available but not recommended
for HunyuanVideo 1.5 (weak calibration signal, visibly degraded output, and the probe's second
weight copy does not fit in HBM at the default shape).

---

## Stage / Core Count Reference

| Model | CP support | tp | cp | Subprocess stages | `NEURON_RT_VIRTUAL_CORE_SIZE` | Work-dir tensors |
|---|---|---|---|---|---|---|
| black-forest-labs/FLUX.1-dev | yes | 2 | 2 | none (in-process) | unset | — |
| Lightricks/LTX-2 | no | 4 | 1 | none (in-process) | unset | — |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | yes | 2 | 2 | transformer (4 cores), vae (1 core) | unset | `latents.pt` |
| hunyuanvideo-community/HunyuanVideo | yes | 2 | 2 | clip (1), llama (4), generate (4) | 2 | `clip.pt`, `llama.pt` |
| hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v | yes (ulysses, gather_kv) | 2 (+ `--cfg-parallel`) | 1 | encode (host), aot_transformer (4), aot_vae (host by default; 1 with `--device-vae`) | 2 | `embeds.pt`, `latents.pt` |
| Qwen/Qwen-Image | yes | 2 | 2 | text (4), generate (4), vae (1) | 2 | `text.pt`, `latents.pt` |
