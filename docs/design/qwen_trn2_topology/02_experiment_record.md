# Experiment Record

## Environment

- Instance: `trn2.3xlarge`
- Logical NeuronCores: 4 (`0-3`)
- HBM: 96 GB
- Driver: 2.28.0.0
- Runtime: 2.32.31.0
- Container: AWS Neuron PyTorch inference SDK 2.30 base with Difflet commit `94e6c42`
- Persistent experiment location during this session: `/mnt/difflet-data`
- Warning: `/mnt/difflet-data` is instance store and is not durable across stop/terminate or host loss.

## Compilation

All three artifacts compiled successfully:

| Stage | TP | Artifact |
| --- | ---: | --- |
| Prompt encoder | 4 | `qwen_image_enc_tp4cp1_seq256/model.pt` |
| Denoiser | 4 | `qwen_image_dit_tp4cp1_h1024w1024/transformer/model.pt` |
| VAE decoder | 1 | `qwen_image_vae_h1024w1024/model.pt` |

Resident serving additionally compiles `qwen_image_vae_tp4_h1024w1024/model.pt`. Its compile completed in 413.16 seconds. The staged CLI continues to use the TP=1 artifact.

NxD embeds the compiled executable in the TorchScript `model.pt` archive and writes `neuron_config.json`; a standalone `.neff` is not required in the artifact directory.

## Offline generation

The staged CLI completed text, denoising, and VAE decoding in separate processes. It produced `/mnt/difflet-data/outputs/qwen-smoke.png`, a valid 1024x1024 RGB PNG.

Primary logs:

- `/mnt/difflet-data/logs/qwen-compile.log`
- `/mnt/difflet-data/logs/qwen-generate.log`

## Same-process resident co-load

Observed sequence:

1. Prompt encoder TP=4 loaded and warmed up.
2. Denoiser TP=4 loaded.
3. Decoder TP=1 reached `Initializing traced model weights for ranks: 0...0`.
4. The worker received SIGSEGV in `libtorchneuron.so`.

Logs:

- `/mnt/difflet-data/logs/qwen-serve-startup.log`
- `/mnt/difflet-data/logs/qwen-serve-kernel-crash.log`

## Separate-process experiment

The TP=4 process loaded encoder and denoiser and reached `TP4_READY`. A second process configured with one visible core then failed with:

```text
Logical Neuron Core(s) not available - Requested:lnc0-lnc0 Available:0
```

After stopping the TP=4 process, the same decoder process loaded successfully:

```text
Finished weights loading in 5.907568318999893 seconds
Warmup completed in 0.2734084129333496 seconds
TEST DECODER_READY
```

Logs:

- `/mnt/difflet-data/logs/qwen-tp4-resident-test.log`
- `/mnt/difflet-data/logs/qwen-decoder-concurrent-test.log`
- `/mnt/difflet-data/logs/qwen-decoder-alone-test.log`

## Official documentation assessment

- [Parallel Execution using NEURON_RT_NUM_CORES](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/appnotes/perf/neuron-cc/parallel-ncgs.html) is explicitly relevant to Inf1. It describes multiple models in one process, but its same-group PyTorch example uses models with the same group size; it does not validate this Trn2 NxD mixed-TP case.
- [Neuron Runtime configuration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/configuration-guide.html) confirms a core reserved by one process cannot be used by another until that process exits.
- [torch-neuronx core placement](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/inference/core-placement.html) applies to Inf2, Trn1, and Trn2 and supports explicit placement for `torch_neuronx.trace()` ScriptModules. The experiment showed that wrapping NxD ModelBuilder weight initialization in this context does not make mixed TP viable.
- [Logical NeuronCore configuration](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html) explains that Trn2 LNC=2 combines each pair of physical NeuronCore-v3 units into one logical core. This instance exposes four logical cores from eight physical cores.

## Mixed-TP experiment matrix

| Case | Load order | Placement | Result |
| --- | --- | --- | --- |
| A | TP=4 text, TP=4 denoiser, TP=1 VAE | Implicit rank 0 | Native SIGSEGV in `libtorchneuron.so` |
| B | TP=4 text, TP=4 denoiser, TP=1 VAE | Explicit core 0 | Deadlock in `nxd_model.initialize`; terminated after more than 80 seconds |
| C | TP=1 VAE, TP=4 text, TP=4 denoiser | Explicit core 0 | VAE loads; TP=4 text load fails with `Unknown Failure` |
| D | TP=1 VAE, TP=4 text, TP=4 denoiser | Implicit | Same failure as case C |
| E | TP=1 VAE only | Core 0 | Loads in 5.91 seconds plus 0.27 seconds warmup |
| F | TP=4 text, TP=4 denoiser, TP=4 VAE | Shared process | All stages load and full generation passes |

Additional logs:

- `/mnt/difflet-data/logs/qwen-mixed-tp-explicit-decoder-last.log`
- `/mnt/difflet-data/logs/qwen-mixed-tp-explicit-decoder-first.log`
- `/mnt/difflet-data/logs/qwen-mixed-tp-implicit-decoder-first.log`
- `/mnt/difflet-data/logs/qwen-vae-tp4-compile.log`
- `/mnt/difflet-data/logs/qwen-all-tp4-smoke-4step.log`
- `/mnt/difflet-data/logs/qwen-service-all-tp4.log`

## Validated resident service

- Stage load times in the standalone all-TP4 smoke: text 12.32 seconds, denoiser 16.85 seconds, VAE 2.11 seconds.
- Four-step full generation smoke: 2.80 seconds after load.
- Smoke output: 1024x1024 RGB PNG, SHA-256 `419343f3d69dab96db3f6015eb8fbd82b80ab32ae1dcfaf2573fa6c94b12788a`.
- Real server: `/health` 200, `/ready` 200, external four-step chat request completed in about 12.5 seconds.
- Resident device memory: 73,401,942,984 bytes, about 68.36 GiB; each logical core accounts for about 17.09 GiB.
