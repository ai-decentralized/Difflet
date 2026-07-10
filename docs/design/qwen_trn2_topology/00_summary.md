# Qwen-Image Trn2 Mixed-TP Serving Investigation

## Research anchor

- Repository: `/Users/clark/project/yotta/Difflet`
- Remote: `git@github.com:ai-decentralized/Difflet.git`
- Branch: `feature/serving_t2i`
- Commit: `94e6c42e296b93af63aa0d722d579d682a4e250b`
- Research date: 2026-07-10
- Research owner: Codex
- Target: EC2 `i-0980bd14d2dafe6d5`, `trn2.3xlarge`, four logical NeuronCores, 96 GB HBM
- Relevant uncommitted changes at research start: Qwen serving artifact detection, its unit tests, and `tasks/todo.md`

## Scope

This investigation determines whether the Qwen-Image prompt encoder and denoiser can remain resident at TP=4 while the TP=1 VAE decoder executes serially in the same Neuron process and core allocation.

Non-scope: changing image quality, model architecture, API contracts, or production deployment infrastructure.

## Documents

- [Architecture and lifecycle](01_architecture_lifecycle.md)
- [Experiment record](02_experiment_record.md)
- [Adaptation assessment](03_adaptation_assessment.md)
- [Other model topology audit](04_other_models_topology_audit.md)
- [Flux Trn2 runtime validation](05_flux_runtime_validation.md)

## Key findings

1. Compilation is already stage-correct: prompt encoder TP=4, denoiser TP=4, decoder TP=1.
2. Offline subprocess execution succeeds end to end and produces a valid 1024x1024 PNG.
3. The original resident process successfully loads both TP=4 stages, then segfaults in `libtorchneuron.so` while initializing the TP=1 decoder.
4. Splitting decoder into another process avoids the segfault but cannot run concurrently: the TP=4 process reserves all four logical NeuronCores and the decoder receives `Logical Neuron Core(s) not available`.
5. The cited parallel NCG application note is explicitly scoped to Inf1. Its PyTorch examples do not establish that a Trn2 NxD weight-separated TP=4 model and TP=1 model can overlap on one core group.
6. Explicit core-0 placement did not make mixed TP viable: decoder-last deadlocked in NxD initialization, while decoder-first loaded TP=1 and then failed to load the first TP=4 model.
7. Compiling a serving-only VAE at TP=4 fixed the topology. All three TP=4 stages co-loaded, a four-step 1024x1024 generation smoke passed, and the real API became healthy and ready.
8. The resident all-TP4 service uses 73,401,942,984 device bytes, about 68.36 GiB, leaving about 27.6 GiB of the 96 GiB device memory available.

## Decision status

- **Validated:** staged subprocess execution is functional.
- **Rejected:** mixed TP=4/TP=1 in one process and two simultaneously resident processes on this four-core instance.
- **Accepted:** one resident process with serving artifacts at TP=4 for prompt encoder, denoiser, and VAE decoder.
- **Fallback only:** TP=2 producer plus a dedicated TP=1 core, another Neuron device/instance, or rotating stage processes.

Findings are guaranteed only for the recorded commit, Neuron runtime 2.32.31, driver 2.28.0, and the SDK 2.30 container used by these experiments.
