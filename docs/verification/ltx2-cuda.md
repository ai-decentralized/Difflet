# Experimental LTX-2 on CUDA

Lightricks/LTX-2 runs on the existing single-device GPU application lifecycle.
Python can return joint audio/video or both latent streams. The generic CLI and
video-serving interfaces return video only; they do not mux the generated audio.
Those frontends have CPU regression coverage, not end-to-end GPU qualification.

Use BF16, one GPU and one resident shape. Height/width must be multiples of 32;
frame count must be 8k+1. TP/CP/SP/CFG parallelism, quantization, offload, TeaCache,
TAEF1 and bucketed shapes are outside this CUDA implementation. Other models'
GPU behavior and Trainium/Neuron/TPU implementations are unchanged.

## Installation and Python generation

The opt-in official-compatible profile uses a separate Linux x86_64/Python 3.12
runtime. Do not combine it with the older `cuda` extra or `requirements-cuda.lock`.
Start from a fresh virtual environment:

```bash
python3.12 -m venv .venv-ltx2
source .venv-ltx2/bin/activate
pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-ltx2-cuda.lock
pip install -e '.[ltx2-cuda]'
export DIFFLET_BACKEND=cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Export the workspace setting **before the process starts**, so it is in place
before CUDA initializes. The profile checks it when the pipeline is constructed and
refuses to start without it; it never sets or changes the variable itself. Python
callers explicitly select TP=1;
the shared registry's Trainium topology remains the default when `parallel` is omitted.

```python
from difflet import DiffletParallelConfig, DiffletPipeline

pipe = DiffletPipeline.from_pretrained(
    "Lightricks/LTX-2",
    backend="cuda",
    parallel=DiffletParallelConfig(tp_degree=1),
    dtype="bf16",
    height=512,
    width=768,
    num_frames=121,
    application_kwargs={
        "generation_profile": "ltx_official_bf16",
        "torch_compile": True,  # False selects eager
    },
)
result = pipe.app.generate_audio_video(
    "A small orange cat walks across a sunlit garden path, birds singing softly in the background.",
    negative_prompt="blurry, distorted, low quality",
    num_inference_steps=40,
    guidance_scale=4.0,
    seed=0,
)
print(result.frames.shape, result.audio.shape, result.sample_rate)
```

The default `generation_profile="diffusers"` preserves the stock Diffusers schedule
and sampler behavior. `ltx_official_bf16` explicitly selects the native-compatible
schedule, conditioning, rotary arithmetic and BF16 sampler/guidance/update order.
Compilation preserves BF16 intermediate casts and native RMS normalization and
retains eager GEMM selection. These policies are local to LTX-2; other GPU models
continue using the shared compiler's existing mode.

## Audiovisual correctness and fair performance

Pre-submission validation used one B200 180GB and checkpoint revision
`64f65d09e0b76c96202b5bc09997673184cf9b5d`. Native vLLM-Omni was pinned to
`74b022ec52668b7914bd3da742649f7d44975b99`. Both runtimes used Python 3.12.13,
PyTorch 2.13.0+cu130 and Triton 3.7.1. Difflet used Diffusers 0.38.0 /
Transformers 5.20.1; native used Diffusers 0.40.0 / Transformers 5.19.0 / vLLM 0.32.0.
These are pinned supported engine environments, not identical dependency stacks.

All four configurations passed full decoded video/audio, both latent streams,
ten effective inputs, warm/fresh-process repeats and observer-control gates on
both workloads. Reduced quality used seeds 0/1/2; larger quality used seed 0, with
two fresh processes per configuration/workload. Gates retain finite-output and
cosine>=0.99 checks, video PSNR>=25dB (or exact equality), other-field NRMSE<=0.1,
and per-field NRMSE<=max(1e-6,1.2*measured reference-repeat NRMSE). Absolute gates
also apply to reference repeats; an unstable reference cannot relax the bound.

Both compiled configurations separately demonstrated generated Triton CUDA
execution and 48 regional block wrappers, tied to the actual runtime/device/input
identity. Corrected native additionally demonstrated its original ATen RMS CUDA
inside compiled graphs. Wrapper counts or eager fallback alone were not admitted.

**Native compiled below uses a corrected private policy**, preserving BF16 casts
and original native RMS normalization. Unmodified upstream default-Inductor
compilation failed audiovisual correctness and is excluded from this table.
This PR neither modifies upstream Omni nor presents that default as qualified.

Both workloads use 40 steps, CFG 4,24 fps, BF16, the prompt above and timing seed 0.
The reduced shape is 320x512/17frames; larger is 512x768/121frames. Each cell has
three fresh launches with private compiler caches, one cold request, one untimed
warmup and five timed warm requests. All 24 launches/120 warm requests are retained.

| Workload | Configuration | Warm median | Load+first median | Compiler interval union |
|---|---|---:|---:|---:|
| Reduced | Difflet eager |4.885s|48.433s|0s|
| Reduced | Difflet compiled |3.225s|69.755s|20.214s|
| Reduced | Native eager |6.885s|78.987s|0s|
| Reduced | Native compiled, corrected private |3.649s|96.984s|21.441s|
| Larger | Difflet eager |20.795s|64.933s|0s|
| Larger | Difflet compiled |15.026s|81.559s|20.249s|
| Larger | Native eager |20.445s|92.183s|0s|
| Larger | Native compiled, corrected private |15.204s|107.632s|21.413s|

Warm clocks cover real prompt encoding through full unencoded CPU audio/video
output delivery, synchronized before/after. Native public-API dispatch/transport
and output adaptation remain timed. Load includes CUDA context/device inspection,
engine loading and policy installation. Interpreter imports, checkpoint download,
media encoding, evidence serialization/transfer and out-of-request validation are
excluded. Compiler cost is the actual worker's CompilationMetrics interval union,
not cold-minus-warm; incomplete backend totals are unavailable. No warm recompiles
were observed. GPU profilers and output observers were absent from timed requests.

All cells pass pooled population CV<=10% and max/min launch-median spread<=10%;
worst values are 3.57% and 9.14%. Paired Difflet/native warm time ratios across the
three launch rounds are 0.692-0.739 eager / 0.879-0.884 compiled on reduced, and
1.014-1.019 eager /0.977-0.999 compiled on larger (Difflet seconds divided by
native seconds throughout). Only three paired rounds
provide weak uncertainty/tail resolution; shared references are correlated.
Rotating four configurations over three rounds leaves residual order imbalance,
and OS checkpoint page-cache state is uncontrolled. No significance claim is made.

One separate observed memory launch per cell measured loading allocator peaks
of 63.137GiB. Sampled whole-lifetime device maxima (eager/compiled) were
Difflet 70.053/70.053GiB versus native 70.084/70.086GiB reduced, and
Difflet 74.098/73.174GiB versus native 73.752/76.455GiB larger. Configured 50ms
sampling delivered median 114-119ms intervals, with gaps up to 1.784s. Sampled
maxima are lower bounds; allocator counters exclude nonallocator/earlier-context
memory. One memory pass per cell gives no variance estimate; memory-observed
durations are excluded from qualified latency.

Difflet is faster on reduced; larger modes are close. Existing profiles did not
establish a causal optimization for the reduced turnaround gap, so no speculative
optimization or faster-but-failing candidate is included. The measurements
precede this scope-only submission extraction; model arithmetic/compiler policy
are preserved, but no post-extraction GPU run or other-model hardware regression
is claimed. Current CPU checks cover the extraction and public entry behavior.
