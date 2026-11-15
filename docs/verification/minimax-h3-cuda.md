# MiniMax-H3 on CUDA

Difflet runs [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) on one
CUDA GPU and writes MP4 files with a stereo 32 kHz audio track. It supports:

- **T2VA**: text to video with audio.
- **I2VA**: an image as the first frame, plus text, to video with audio.

Both run through `difflet run`, `difflet serve` and the Python API, eager or with
`--torch-compile`.

The checkpoint is the released repository's root diffusers layout (`transformer/`,
`text_encoder/`, `vae/`, `audio_vae/`, `processor/`, `tokenizer/` and the two scheduler
configs, about 136 GB); `difflet` downloads only those. The commands below pin the checkpoint
revision they were checked with, `0919fb684102ef115126612f3fbb84d1027e7317`.
Reference-to-video (Ref2VA) is not supported.

## Requirements

- One NVIDIA GPU that holds the whole model: about 122 GB of weights plus activations, so a
  180 GB-class card such as a B200. Every component stays on the device.
- The GPU backend's setup: a CUDA 12.8+ driver and Python 3.11 or newer.
- About 140 GB of free disk for the checkpoint.

## Install

The README's GPU install, at the versions in `requirements-cuda.lock`:

```bash
git clone https://github.com/ai-decentralized/Difflet.git
cd Difflet
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
pip install -e ".[cuda]" -c requirements-cuda.lock
```

## Run

Every command needs `DIFFLET_BACKEND=cuda`. The first run downloads the checkpoint into the
Hugging Face cache.

Frame counts have the form 17n + 5, from 120 to 360: at 24 fps, 192 frames is 8 s and 243
frames is 10 s. Height and width are multiples of 32 and at least 256. The default is
1344x768 with 124 frames. `--steps` counts denoising steps (transformer evaluations), as the
reference does; the default is 50.

```bash
export DIFFLET_BACKEND=cuda

# T2VA, 10 s at 1344x768
difflet run --model-id MiniMaxAI/MiniMax-H3 --revision 0919fb684102ef115126612f3fbb84d1027e7317 \
  --height 768 --width 1344 --num-frames 243 \
  --steps 50 --seed 0 --prompt "A paper boat drifts down a rain-soaked street." \
  --output t2va.mp4

# I2VA, 8 s: the image becomes frame 0
difflet run --model-id MiniMaxAI/MiniMax-H3 --revision 0919fb684102ef115126612f3fbb84d1027e7317 \
  --height 768 --width 1344 --num-frames 192 \
  --steps 50 --seed 0 --first-frame first_frame.png \
  --prompt "The family keeps talking while steam rises from the ramen." \
  --output i2va.mp4
```

Add `--torch-compile` to compile the transformer blocks.

The image is converted to RGB and stretched to the output size with LANCZOS, without
cropping, as the neuron backend does when `--height`/`--width` are given. On CUDA the output
size never follows the image: without `--height`/`--width` it is the 1344x768 default,
whatever the image's aspect ratio. Prompts take up to 1024 tokens. `--last-frame` (a closing
keyframe) is refused on CUDA.

## Serve

```bash
DIFFLET_BACKEND=cuda difflet serve --model-id MiniMaxAI/MiniMax-H3 \
  --revision 0919fb684102ef115126612f3fbb84d1027e7317 \
  --height 768 --width 1344 --num-frames 192 --port 8000 --request-timeout 1800
```

A request takes about five minutes on a B200, longer than the default 300 s request timeout,
so the command raises it. Add `--torch-compile` for compiled serving. A server has one output
size, which its T2VA and I2VA requests share. Requests use the OpenAI-style video endpoint:
upload the first-frame image as `first_frame` for I2VA, and leave it out for T2VA.

```bash
curl -sf http://127.0.0.1:8000/v1/videos/sync \
  -F "prompt=The family keeps talking while steam rises from the ramen." \
  -F seed=0 -F num_inference_steps=50 \
  -F "first_frame=@first_frame.png;type=image/png" -o i2va.mp4
```

The upload is a PNG, JPEG or WebP file of at most 16 MiB and 4096x4096 pixels, at least 32
pixels a side, as for the neuron backend's keyframe servers
([Videos API](../serving/videos_api.md)). A `last_frame` upload is refused.

## Python

```python
from PIL import Image

from difflet import DiffletPipeline
from difflet.pipeline.parallel_config import DiffletParallelConfig

pipe = DiffletPipeline.from_pretrained(
    "MiniMaxAI/MiniMax-H3",
    revision="0919fb684102ef115126612f3fbb84d1027e7317",
    backend="cuda",
    parallel=DiffletParallelConfig(tp_degree=1),
    height=768,
    width=1344,
    num_frames=192,
)
result = pipe.app.generate(
    "The family keeps talking while steam rises from the ramen.",
    first_image=Image.open("first_frame.png"),  # omit for T2VA
    num_inference_steps=50,
    seed=0,
)
video = result.video  # CPU float tensor (1, 3, frames, height, width) in [0, 1]
audio = result.audio  # CPU float tensor (1, 2, samples) in [-1, 1]
rate = result.sample_rate  # 32000
```

Pass `parallel=DiffletParallelConfig(tp_degree=1)` explicitly: without it the registry's
default layout is the Trainium one. Pass `application_kwargs={"torch_compile": True}` for
compiled execution.

## How it runs

The code is in `difflet/backends/cuda/minimax_h3/`. The transformer, the two VAEs and their
tiling are CUDA's own modules, separate from the ones the neuron and Trainium backends run
(`difflet/models/minimax_h3/`), so a change on either side leaves the other's output alone.

- **Residency.** `load` puts every component on the GPU: the transformer, the Qwen3-VL text
  encoder with its vision tower, and the visual VAE's encoder and decoder and the audio
  VAE's decoder in FP32.
- **First-frame conditioning.** The canvas, the presentation to the text encoder (the image
  as image tokens in front of the prompt) and the packed layout are the shared FL2VA code in
  `difflet/models/minimax_h3/` that the neuron backend also runs; CUDA's encoder encodes the
  image over the reference's 256-pixel tiles. The image's latent frame is mixed with a little seeded noise and placed in the packed sequence
  as extra rows at frame 0's position. Those rows stay fixed during denoising; every output
  frame, frame 0 included, is generated.
- **Noise.** CUDA draws its noise the way MiniMax's published samples were made: the video
  and the audio noise each from a generator restarted at the request seed, and the keyframe's
  from frame 0 of a full-length draw at the seed. This is what reproduces the official
  `assets/i2va.mp4` below. The neuron backend follows Diffusers' modular pipeline instead,
  drawing keyframe, video and audio noise from one generator in turn, so the two backends'
  clips for one seed differ.
- **Attention.** The sequence is packed without padding rows and attends without a mask, so
  PyTorch SDPA can pick its fastest kernel.
- **Fused kernels.** Triton kernels do the per-row elementwise work in FP32 with the
  reference's BF16 roundings: RMS norm with AdaLN modulation, the gated residuals, Q/K
  norm with rotary embedding, and the SwiGLU gate. They and the remaining FP32 RMS norms
  are custom ops, which compiled blocks call as they are, and Inductor (default mode)
  rounds to BF16 where eager does. Compiled output therefore tracks eager; left to
  Inductor, those fused reductions moved the output from one compilation to the next.

## Results

Measured on one B200 (180 GB) with the install above, and compared with vLLM-Omni at commit
`74b022ec` on the same card in the same session. The acceptance check is vLLM-Omni's own
MiniMax-H3 I2VA test, `tests/e2e/accuracy/minimax_h3/test_minimax_h3_i2va_ref2va_similarity.py`.
It compares the output with MiniMax's official `assets/i2va.mp4` (1344x768, 192 frames,
24 fps, 50 steps, seed 0) and requires:

- matching video metadata;
- AAC 32 kHz stereo audio;
- ffmpeg SSIM ≥ 0.97 and PSNR ≥ 34 dB.

Times are for `difflet serve` and vLLM-Omni's server: one warm-up request, then the median
of three.

| I2VA, 1x B200 | SSIM | PSNR | Warm request | vs vLLM-Omni |
|---|---|---|---|---|
| Difflet eager | 0.9770 | 35.02 dB | 301.2 s | 0.984x |
| Difflet compiled | 0.9770 | 35.02 dB | 300.8 s | 0.985x |
| vLLM-Omni eager | 0.9756 | 36.03 dB | 306.0 s | |
| vLLM-Omni compiled | 0.9755 | 35.49 dB | 305.6 s | |

All nine Difflet I2VA clips pass: four from each server and one from `difflet run`. Compiled
output agrees with eager to within 2e-5 SSIM. An eager request spends 5.66 s per denoising
step, 9.0 s decoding video, and under a second on the prompt, keyframe and audio.

| | Ready | First request | Peak GPU memory |
|---|---|---|---|
| Difflet eager | 85 s | 307 s | 141.6 GiB |
| Difflet compiled | 95 s | 306 s | 141.7 GiB |
| vLLM-Omni eager | 95 s | 364 s | 130.6 GiB |
| vLLM-Omni compiled | 75 s | 313 s | 131.1 GiB |

Difflet's server loads every component and runs a short generation before it reports ready,
which also compiles its blocks. Peak memory is the device's used memory, sampled every second.
Difflet's peaks were sampled in an earlier session with the same checkpoint and precision;
vLLM-Omni's peaks were the same to the MiB in both sessions.

T2VA requests through both servers, and a compiled 10 s `difflet run`, returned H.264
1344x768 at 24 fps with AAC 32 kHz stereo. T2VA is not part of the acceptance test; for
information, the 10 s clip scores SSIM 0.866 against MiniMax's official `assets/t2va.mp4`.

## Limits

- One GPU per generation: tensor, sequence, context and CFG parallelism are refused.
- Ref2VA, a last keyframe, negative prompts and guidance scales are not supported. The
  checkpoint is guidance-distilled.
- BF16 only.
