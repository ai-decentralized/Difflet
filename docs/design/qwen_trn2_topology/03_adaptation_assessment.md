# Adaptation Assessment

## Goal

Keep Qwen-Image serving warm on one four-core Trn2 device without recompiling or loading the decoder for every request.

## Rejected: shared-process mixed TP

Reserve cores 0-3 once, keep both TP=4 models loaded, and place the TP=1 decoder on core 0. This has the best latency and preserves the existing in-memory request path.

Risks:

- Current implicit NxD initialization already crashes in this topology.
- The documented explicit placement API is specified for `torch_neuronx.trace()` modules, while these artifacts use NxD ModelBuilder and explicit weight initialization.
- Different NxD model-parallel world sizes mutate global parallel state in one Python process.

Result: rejected by implicit, explicit-placement, and reversed-order runtime experiments.

## Accepted: shared-process all TP=4

Compile a serving-only VAE with `tp_degree=4` and `world_size=4`, while preserving the staged CLI's TP=1 VAE artifact. The resident worker loads all three TP=4 artifacts under one immutable four-core runtime.

Measured tradeoff:

- No decoder process startup per request.
- No mixed-world-size NxD initialization.
- About 68.36 GiB resident device memory across the four logical cores.
- VAE is replicated because its convolution layers are not tensor-parallel aware.
- Four-step startup smoke and real API generation both pass.

## Hypothesis B: TP=2 resident producer plus TP=1 decoder

Recompile prompt encoder and denoiser at TP=2, reserve cores 0-1 for their process, and reserve core 2 for a decoder process. This avoids mixed TP in one process and keeps both workers warm.

Risks:

- TP=2 may exceed per-core HBM or fail compilation.
- Throughput and denoising latency may regress.
- Requires new TP=2 artifacts and serving IPC.

Decision gate: compile succeeds, both processes coexist, and end-to-end output passes numerical/visual smoke with acceptable latency.

## Hypothesis C: rotating stage process

Keep TP=4 stages resident during text/denoising, then terminate that worker and start the decoder process. The decoder control test measured about 5.91 seconds for weight load plus 0.27 seconds warmup, excluding broader container/Python startup.

This is functionally viable but does not meet low-latency resident serving goals. It is a fallback, not the preferred fix.

## Hypothesis D: decoder on another device

Place the decoder on another Neuron device/instance and pass latents over IPC/RPC. This preserves warm residency and avoids mixed TP but increases infrastructure cost and transfer complexity.

## Decision

Use the all-TP4 serving artifact topology for the current four-core Trn2 resident worker. Keep TP=1 VAE for staged CLI generation. Do not implement `torchrun` rank-0-only logic: the resident worker uses one Python process controlling an NxD four-rank model, not four independent OS rank processes.

TP=2 text compilation was also proven possible, but further TP=2 denoiser/IPC work is unnecessary for the accepted topology and remains a fallback investigation.
