# Torch-native (neuron) backend against AoT: results

All figures were measured on one instance, under matched conditions, by the same procedure for both backends.

## Setup

| | |
|---|---|
| Instance | trn2.3xlarge, LNC = 2 (4 logical NeuronCores, 24 GiB HBM each); host Xeon Platinum 8488C, 6 cores / 12 vCPUs, 124 GB RAM |
| Storage | EBS gp3, 125 MB/s, 3000 IOPS (every cold figure is bounded by it) |
| AoT stack | neuronx-cc 2.26.6360, PyTorch 2.9.1, torch-neuronx 2.9.0 (XLA), NxD 0.19, NxD Inference 0.10, Python 3.12 |
| AoT code | Difflet `main` at 52b615f. Later `main` adds the all-reduce dtype to every compiled artifact's identity (`--tp-reduce-dtype`, so earlier artifacts are rebuilt) and decodes Wan on the host in fp32 by default |
| Native stack | Neuron SDK 2.33 public DLAMI: PyTorch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631, NKI 0.7.0, Python 3.12; diffusers 0.38.0, transformers 5.20.2 |
| Models | FLUX.1-dev (rev. 3de623fc); Wan 2.2 T2V A14B (rev. 5be7df96), high-noise expert for all steps, as both CLIs run it; HunyuanVideo 1.5 480p T2V (rev. 286be7ce), native only |
| Workloads | Flux 1024², 50 steps, guidance 3.5; Wan 480×832×9, 50 steps, CFG 4.0; HunyuanVideo 1.5 480×848×121, 50 steps, CFG 6.0 |

Component precision and placement, matched to the AoT configuration (diffusers' documented usage for Flux; for Wan, AoT's device path):

| Model | Text encoders | DiT | VAE |
|---|---|---|---|
| Flux | CLIP-L and T5-XXL bf16, device (both backends) | bf16 | bf16, device (both) |
| Wan 2.2 | umT5-XXL bf16, device (both) | bf16 | bf16 on the device up to 9 frames (both); beyond, fp32: AoT on the host, native on the device |
| HunyuanVideo 1.5 | Qwen2.5-VL and ByT5 bf16, host | bf16 | bf16, device, 128-pixel tiles |

## Method

- **Compile**: one build from empty caches.
- **Cold e2e**: one generation after dropping the OS page cache, compiled artifacts kept; the bytes read are the process's file-system input blocks × 512.
- **Warm e2e**: median of three generations for native; one generation for AoT.
- **Per step**: native, the median of the in-run step times (49 intervals); AoT Flux, the progress bar's 50-step average; AoT Wan, which reports no step timing, $t_{\text{step}} = (T_{50} - T_{10}) / 40$ from two warm runs. Each run decomposes as $T = T_{\text{fixed}} + N \cdot t_{\text{step}} + B / \text{BW}$, the last term only when cold.

## Performance, same layout (TP4)

| | | Compile | Cold e2e | Warm e2e | Per step |
|---|---|---|---|---|---|
| **Flux** | AoT | 976.8 s | 357.2 s (34.9 GB) | 45.7 s | 271 ms |
| | Native | **532.7 s** | **343.0 s** (35.7 GB) | 64.2 s | **268.6 ms** |
| **Wan 2.2, 9 frames** | AoT | 6,797 s | 621.4 s (40.1 GB) | 117.7 s | 1,165 ms |
| | Native | **807.7 s** + 51.5 s bf16 copy | **447.4 s** (41.7 GB) | 129.3 s | **1,063 ms** |
| **HunyuanVideo 1.5, 121 frames** (TP2 × CFG2) | Native | 2,837 s | 1,255 s (37.6 GB) | 934.7 s | 12,625 ms |

- **Compile**: native is 1.8× (Flux) and 8.4× (Wan) faster.
- **Per step**: native is on par for Flux and 9% faster for Wan.
- **Cold**: with equal bytes, native reads at the volume's throughput (≈125–131 MB/s) and AoT at ≈80–112 MB/s; native is 4% (Flux) and 28% (Wan) faster. Wan and HunyuanVideo 1.5 store their DiT in fp32; native writes a bf16 copy once at compile time (bit-identical to casting at load), which halves the DiT's cold read (Wan 70.5 → 41.7 GB; HunyuanVideo 1.5 DiT load 253 → 127 s).
- **Warm**: the gap is fixed per-launch cost, not compute. Native's fixed part comprises the Neuron process-group initialization, bimodal at ≈9 s or ≈18–21 s per launch (Flux TP4's three warm runs all fell in the slow mode; the fast mode gives 54.9 s), and a DiT warm-up forward per launch (Flux 8.7 s; Wan 13.6 s at TP4). AoT's fixed part is ≈32 s (Flux) and ≈59 s (Wan).

Native's best layouts (native only):

| | Layout | Per step | Warm e2e | Cold e2e |
|---|---|---|---|---|
| Flux | TP4 + SP | 261.1 ms (−3%) | 54.9 s | 334.0 s |
| Wan 2.2, 9 frames | TP2 × CFG2 | 933 ms (−12%) | 138.6 s | 453.9 s |

Wan's TP2 × CFG2 is faster per step but slower per launch: its warm-up forward takes 26.0 s (13.6 s at TP4) and each rank loads half the model rather than a quarter (+17 s of load); it pays off over longer jobs or in a long-lived process.

## Quality

Reference: diffusers in fp32 on the CPU. Baseline: diffusers in bf16, i.e. what bf16 inference itself costs. All runs start from the same initial noise; PSNR and SSIM over the decoded frames, seed 42. HunyuanVideo 1.5 is evaluated at 9 frames (a 121-frame fp32 reference takes ≈45 h on this CPU).

| Distance to diffusers fp32 | Flux | Wan 2.2, 9 frames | HunyuanVideo 1.5, 9 frames |
|---|---|---|---|
| diffusers bf16 (baseline) | 15.90 dB / 0.771 | 24.66 dB / 0.871 | 30.14 dB / 0.961 |
| **Native** | **16.08 dB / 0.775** | **24.20 dB / 0.865** | **28.09 dB / 0.946** |
| AoT | 31.13 dB / 0.977 | 11.24 dB / 0.544 | — |
| AoT latents, fp32 host VAE | — | 23.24 dB / 0.846 | — |

- **Native matches diffusers' bf16** on Flux and Wan. On HunyuanVideo 1.5 it is 2 dB below the baseline on this seed, visually indistinguishable; decoding its latents untiled in bf16 or fp32 gives the same 28.1 dB, so the decoder is excluded. One seed cannot separate this from the baseline's seed-to-seed spread.
- **AoT Flux is closer to fp32 than bf16 inference.** Its fp32 all-reduces are not the cause: native with fp32 cross-core sums stays at 16.07 dB (and costs +26 ms per step). The source lies elsewhere in AoT's compiled computation.
- **AoT Wan's device VAE degrades short clips.** Its latents decoded by the fp32 host VAE reach 23.24 dB; its compiled bf16 device decoder gives 11.24 dB, with blur and a grid pattern. AoT's long clips already decode on the host.
- **Seeds and PyTorch versions.** The same seed gives different initial noise on the two stacks when the noise is drawn in bf16: PyTorch 2.14 samples in fp32 and rounds, PyTorch 2.9 samples bf16 directly. Flux draws its noise in the embeddings' dtype (bf16), so AoT's noise was drawn in fp32 for this comparison; Wan draws in fp32 and was unaffected.

## Backend changes during the campaign

- Component precisions aligned with AoT and diffusers: Flux CLIP-L on the device in bf16, Flux VAE bf16, Wan umT5 bf16, Wan VAE bf16 up to 9 frames; each passed a parity gate against fp32 relative to the worst of several bf16 CPU controls (CLIP 1.018×, Flux VAE 1.001×, umT5 0.83–0.86×, Wan VAE 1.002×).
- fp32 GroupNorm for bf16 device inputs: the device's bf16 `group_norm` reached 1.35 relative error at 4.2M elements per group, and Flux's bf16 VAE 68× its control before the fix.
- bf16 copies of fp32 DiT checkpoints, made at compile time.
- The NKI trace cache, opt-in and versioned by stack, for HunyuanVideo 1.5's bounded attention kernel (≈10 min of tracing per process otherwise).

## Open items

- HunyuanVideo 1.5's steps take 12.6 s in the pipeline against 10.8 s for the DiT forward alone on the public stack (11.2 s against 10.8 s on Beta 5). The extra time is spent on the device; the stack's forward, the kernel's trace cache, the bf16 copy and the inputs are excluded; the launch environment is the remaining suspect.
- The process-group initialization's bimodality (≈10 s fixed offset).
- The per-launch DiT warm-up forwards (13.6 s for Wan at TP4).
- The cold compile of HunyuanVideo 1.5 contains ≈15 min not accounted for by its stages.
- More seeds for HunyuanVideo 1.5's quality, and diffusers references over several seeds for all three models.
- The source of AoT Flux's higher precision.
- Wan's VAE in bf16 at 81 frames (the default stays fp32 there).
