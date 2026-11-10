# HunyuanVideo 1.5 on CUDA

The experimental single-GPU BF16 path targets
`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`.
HunyuanVideo 1.5 is separate from the original HunyuanVideo model.

```bash
difflet run \
  --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf \
  --height 480 --width 848 --num-frames 121 \
  --steps 50 --guidance-scale 6 --seed 0 \
  --prompt "a cat walking across a sunlit garden" --output video.mp4
```

The decoder uses Diffusers' native overlapping spatial tiles: 256×256 pixels
with 25% overlap. This keeps the requested output dimensions and frame count.
Tiled and untiled decoding can differ numerically; accuracy comparisons must
use the same decoder settings. Slicing is disabled.

`--torch-compile` compiles the repeated denoiser blocks. It does not compile
the VAE or text encoders. The blocks request deterministic Inductor reduction
selection so timing-based kernel choices do not vary the reduction policy.
Distributed inference is not part of this CUDA path.

A retained H200 run at source `fc402abf` completed the representative request
above through Python, CLI and resident HTTP in eager and compiled modes, using
Torch 2.11/cu128, Diffusers 0.38 and Transformers 5.20.1. The outputs contained
121 frames at 24 fps. The scoped public-path checks passed in both modes, and
separate CUDA traces verified execution of compiled denoiser blocks. These
traces are mode probes, not measurements of full-request latency.

Independent rescoring of the default-runtime captures reproduced all 510
NumPy metrics and their frozen acceptance bounds. It also retained 13 differences
between Torch and NumPy calculations, at most 2.78e-17, including two marginal
exceedances of strict Torch bounds. No bound was relaxed. The original results
and this rounding caveat must accompany a numerical acceptance claim.

The longer HTTP deadline used for these checks is available through the existing
configuration:

~~~bash
difflet serve \
  --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf \
  --height 480 --width 848 --num-frames 121 \
  --request-timeout 1800
~~~

The public long-prompt check used four steps and establishes request handling,
not full-length long-prompt numerical quality. Coverage remains limited to the
scoped checkpoint and requests; it does not establish broader model support.

The separate comparison against pinned Omni uses a different common runtime and
FP32 VAE policy. Default-runtime acceptance cannot qualify that comparison.
Controlled Qwen component captures isolated device-local rotary initialization
and the cuDNN SDPA policy during text encoding. Applying both candidate changes
matched all 33 captured tensors against Omni in each of two repeats, including
positive and negative conditioning and the first denoiser inputs. Either change
alone retained conditioning differences. These instrumented prefix captures stop
before the denoiser body; enabled SDPA flags do not identify the selected kernel.

The CUDA application now reconstructs the nonpersistent FP32 rotary frequencies
on the loaded encoder's device and scopes the SDPA policy to prompt encoding.
The caller's policy is restored before denoising and on encoding failure.

## Matched comparison with vLLM-Omni

In eager mode each denoiser block runs the joint attention the way vLLM-Omni
does:

- one packed QKV projection per stream;
- the text context trimmed to its valid tokens;
- Q/K RMSNorm and RoPE in one fused Triton op,
  `difflet_hv15::fused_joint_qk_norm_rope`
  (`difflet/backends/cuda/ops_impl/qk_norm_rope.py`), written for Difflet from
  the arithmetic it must reproduce;
- cuDNN SDPA.

Compiled blocks keep the masked valid-prefix path, so one graph serves every
prompt length.

### Setup

- **Code:** Difflet's HunyuanVideo 1.5 source at commit `579b0011`.
- **Reference:** vLLM-Omni `08177216` (vLLM 0.32.0), unmodified.
- **Runtime, shared by both:** Python 3.12, Torch 2.13.0+cu130, Triton 3.7.1,
  Diffusers 0.40.0 and Transformers 5.19.0.
- **Hardware:** one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB).
- **Procedure:** every engine and mode ran in a fresh process. The whole set
  ran twice, the second time in reverse order.

The two requests:

- **G**, vLLM-Omni's own test settings: 832×480, 33 frames, 50 steps, guidance
  6, flow shift 5, seed 42, prompt "A cat walks on the grass, realistic style."
- **P**, the request above: 848×480, 121 frames, 50 steps, guidance 6, seed 0.

Both engines start from bit-identical initial latents.

VAE tiling:

- Difflet always decodes with native tiles.
- vLLM-Omni decodes G untiled, its default, and P with its VAE tiling enabled,
  because an untiled 121-frame decode does not fit in 96 GB.

### Output agreement

Videos were written with Diffusers `export_to_video` at 24 fps and scored with
ffmpeg SSIM and PSNR through vLLM-Omni's own accuracy-test helper.

The acceptance bar is relative. On G, for each mode, Difflet against vLLM-Omni
must score at least what vLLM-Omni's own eager and compiled outputs score
against each other. P is reported but does not gate.

| Request | Mode | Difflet vs vLLM-Omni (SSIM / PSNR) | vLLM-Omni eager vs compiled | Gate |
|---|---|---|---|---|
| G | eager | 0.9711 / 39.95 dB | 0.8412 / 27.51 dB | pass |
| G | compiled | 0.9181 / 31.65 dB | 0.8412 / 27.51 dB | pass |
| P | eager | 0.9555 / 35.46 dB | 0.4999 / 18.04 dB | reported |
| P | compiled | 0.6465 / 21.16 dB | 0.4999 / 18.04 dB | reported |

**Eager.** The denoiser is bit-identical to vLLM-Omni's: for the first step,
in both guidance branches, all 54 blocks and the final output match exactly.
The whole remaining difference comes from the VAE decode. Decoding vLLM-Omni's
latents with Difflet's decoder reproduces Difflet's video exactly. Rounding
vLLM-Omni's FP32 latents to BF16 before its own decoder already gives SSIM
0.9747.

**Compiled.** The two engines compile differently, so their compiled
trajectories differ from block 0:

- vLLM-Omni compiles regions with dynamic shapes;
- Difflet compiles each block with static shapes and deterministic reductions.

Each engine's eager and compiled outputs differ from each other by a similar
amount: Difflet 0.8615 / 28.35 dB, vLLM-Omni 0.8412 / 27.51 dB.

**Repeatability.** Every engine and mode reproduced its first-pass denoiser
digests exactly in the second pass. The second pass also gave the same video
scores.

**Fused op.** The fused Q/K op was also compared directly with vLLM-Omni's
fused op on the GPU:

- inputs: G and P sequence lengths, 13 and 6 context tokens, three seeds each;
- result: Q, K and V bit-identical in 12 of 12 cases, at the same speed.

### Latency, start-up and memory

Warm latency is the median of the measured requests, CUDA-synchronized from
prompt to decoded frames. The speed bar is Difflet / vLLM-Omni ≤ 1.10.

| Mode | Request | Difflet (s), pass 1 / 2 | vLLM-Omni (s), pass 1 / 2 | Ratio, pass 1 / 2 |
|---|---|---|---|---|
| eager | G | 58.35 / 58.38 | 58.24 / 58.31 | 1.002 / 1.001 |
| eager | P | 408.7 / 408.4 | 414.9 / 415.1 | 0.985 / 0.984 |
| compiled | G | 58.18 / 58.14 | 53.74 / 53.75 | 1.082 / 1.082 |
| compiled | P | 405.1 / 405.0 | 393.9 / 393.9 | 1.028 / 1.028 |

Start-up and memory are recorded but do not gate:

- **Load time** (process start to loaded model): Difflet 14–24 s, vLLM-Omni
  24–26 s.
- **First G request** (includes compilation in compiled mode): Difflet
  66.3–66.8 s in both modes. vLLM-Omni 67.2–67.4 s eager, 67.0–72.2 s
  compiled.
- **Peak device memory:**

| Peak memory | Difflet | vLLM-Omni |
|---|---|---|
| Device (NVML) | 62.9 GiB | 92.9 GiB |
| Torch allocated | 45.3 GiB | 72.5 GiB |

These results cover this checkpoint, these two requests and this card. They
do not establish broader model support.

## Numerical harness

The numerical harness requires component admission before its `capture` and
`time` commands load a model:

```bash
python -m tests.numerical.gpu_hunyuan_video15_parity capture \
  --help
```

Supply `--component-admission` with a receipt inside the evidence root and
`--component-admission-sha256` with the digest established by independent review.
The receipt uses `hunyuanvideo15-components-v1`; its bundle contains raw
safetensors for stock eager, stock compiled, the FP32 contract control and the
candidate, plus frozen criteria, a checkpoint inventory, a parameter-name
schema and the pinned reference source. Bindings include the exact request,
cases, source revision, reference modules, execution mode and runtime policy.

The verifier checks weight fingerprints, positive and negative conditioning,
attention and positional encoding, denoiser intermediates and a scheduler
transition. It requires all seven denoiser inputs in both CFG branches, including
the packed video input, both encoder masks and image embeddings. Weight
fingerprints are SHA256 of contiguous BF16 tensor bytes in checkpoint parameter
name order: the FP32 contract control starts from those same BF16 values before
upcasting. Fingerprints are recomputed from checkpoint tensors, rather than
accepted solely because the four roles agree. The raw asset byte budget scales
with the admitted geometry and case inventory so full-resolution intermediates
fit; metadata keeps a separate fixed bound.

Inputs must agree exactly at a common boundary before intentional
precision conversions. Floating outputs must fit the unchanged envelope derived
from the two stock controls against the FP32 control; integer and complex
outputs cannot enter that metric calculation. Criteria must precede candidate
execution. Loaded scheduler configuration and effective runtime shift, decoder, precision, guider class/configuration,
cached effective guidance configuration and runtime policy are checked again
before full requests, and the mode probe precedes full-request warmup. The
request binding fixes the guidance scale that generation sets for each request.

A receipt and its own checksum do not establish independent provenance. The
reviewer must verify how the actual device tensors and frozen controls were
produced. CPU receipt tests establish rejection behavior only; neither those
tests nor admission alone qualify full generation, compiled execution, output
agreement or performance. Existing retained runs keep their original evidence
and acceptance boundaries.

The harness records `initial_compilation` observations for loading, initial
full warmups and the separate mode probe. CPU wall spans inside TorchDynamo
callbacks are reported independently of elapsed warmup time, with overlapping
spans counted once and trigger types kept separate. The profiled mode probe is
labelled accordingly. Missing APIs or incomplete callback spans remain missing
data. Later quality/timed requests are outside this observer's coverage; it
does not measure all backend/native-extension compilation or process-cold
turnaround, and its phase clock does not synchronize CUDA automatically.
Observation is process-wide: active-compilation entry and uncertain boundaries
remain missing. Model attribution requires exclusive compiler use and no
concurrent callback-registry mutation.
