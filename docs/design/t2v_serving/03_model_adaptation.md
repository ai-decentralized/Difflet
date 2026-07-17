# T2V Model Adaptation Assessment

> **2026-07-16 status:** provisional adapters now exist for LTX-2, Wan 2.1,
> and HunyuanVideo 1.0. They deliberately reuse existing lower-layer model
> primitives and explicit host placements. Their local implementation status is
> separate from the still-pending per-profile Trn2 co-load, memory, correctness,
> recovery, and long-run release gates.

The checked-in Trn2 benchmark is stronger than a compile-only artifact check:
it records real fixed-profile offline generation for all three candidates. Its
samples still reload weights in separate CLI processes and retain `.pt` tensor
evidence, so it does not close the resident, memory, MP4, or HTTP gates. See
[Trn2 benchmark evidence audit](07_trn2_benchmark_evidence.md).

## Assessment rules

“Can run in one process” means more than Python objects fitting in memory. The
exact compiled artifacts must co-load under one immutable Neuron runtime plan,
complete warmup and inference, remain reusable after errors/internal recovery,
and fit the 96 GiB HBM target with margin.

The table separates repository observations from recommended serving changes.

## LTX-2

### Observed

- `difflet/cli/orchestrators/ltx_2.py` runs compile and generation directly in
  one process rather than through CLI stage subprocesses.
- `difflet/models/ltx_2/application.py` states that only the dual-stream DiT is
  on Neuron; text encoding, connectors, scheduler, video VAE, audio VAE, and
  vocoder remain host-side.
- CP is currently unsupported; TP defaults to 4. CFG parallel can increase the
  transformer world size.
- The CLI reads `output.frames` and writes video; it does not mux model audio.
- The real Trn2 benchmark completed TP4/CP1 generation at 480x704x49, 20 steps,
  and guidance 1.0 with a finite output tensor. Its warm runs are page-cache
  warm CLI processes, not repeated requests against one resident pipeline.

### Implemented serving posture

- Use one hybrid serving stage, `pipeline`.
- Keep TP=4, CP=1, CFG parallel disabled for the first four-core profile.
- Return a silent MP4 until audio export has an explicit contract.
- Measure host RAM as carefully as HBM because all decode components are hosted.

### Residency conclusion

**Provisionally treated as fitting one model/profile per process, but not yet
proven by a resident serving smoke.** It has no known mixed-world Neuron
component in its current design.

## Wan 2.1

### Observed

- `difflet/cli/orchestrators/wan.py` has two process stages: `transformer` and
  `vae`.
- The transformer stage loads UMT5 and the DiT under the configured world size,
  normally TP=4/CP=1.
- The Neuron VAE stage is compiled and loaded at TP=1/world=1.
- `--host-vae` already skips the Neuron VAE subprocess and decodes latents on
  the host.
- Longer VAE shapes have already encountered compiler instruction limits in the
  existing topology investigation, making host decode operationally relevant.
- The real Trn2 benchmark completed the current TP4/CP1 staged path at
  480x832x9, 20 steps, and guidance 1.0 with a finite output tensor.

### Implemented serving posture

- Use `prompt_encoder`, `denoiser`, and `decoder` logical stages.
- Use host VAE in the resident adapter.
- Pass latents in memory; do not reproduce the CLI work-directory handoff.
- Keep stage-boundary cancellation checks for internal deadline/shutdown
  recovery. Do not claim step-level denoising or decode preemption, and do not
  expose running DELETE as a user-cancellable operation.
- Treat an all-Neuron decoder as a later experiment compiled for the same W4
  serving topology.

### Residency conclusion

**Provisionally treated as fitting one process with host VAE.** The existing W1
Neuron VAE artifact is not loaded beside the W4 transformer artifact. Real
resident reuse and memory measurement are still required.

## Wan 2.2

### Observed

- Difflet's README and offline verification matrix list Wan 2.2 as supported.
  The real Trn2 benchmark completed the current TP4/CP1 path at 480x832x9 and
  produced a finite tensor, so this path is more than command-only plumbing.
- The model application supports `transformer` and `transformer_2` components.
- The current CLI constructs the application with
  `enable_transformer_2=False` in both transformer and VAE stages.
- The recorded run reused the Wan 2.1 NEFF under the historical shape-based
  cache key, and its per-step number was carried over from Wan 2.1. Therefore
  the successful result demonstrates only the single/high-noise expert
  compatibility path, not the intended dual-transformer switching behavior.

### Future requirement

1. Correct and verify the offline Wan 2.2 pipeline first.
2. Compile/load both transformers with one uniform runtime plan.
3. Measure their combined resident HBM plus UMT5 and denoising workspace.
4. Use host VAE initially if the generation components fit.
5. If they do not fit with safe margin, add a rotating/reloading execution plan
   or a future multi-process stage executor; do not hide that behind logical
   stage names.

### Residency conclusion

**Do not enable yet.** Correctness is blocked, and the larger two-transformer
resident set has no measured fit evidence on the current 96 GiB target.

## HunyuanVideo 1.0

### Observed

- `difflet/cli/orchestrators/hunyuan_video.py` runs `clip`, `llama`, and
  `generate` as separate subprocess stages.
- CLIP uses a one-core topology; Llama uses the configured TP topology; generate
  owns the DiT and VAE.
- The DiT world size is TP times CP. The current VAE configuration uses TP as its
  world size, so CP greater than 1 creates incompatible world-size assumptions.
- Existing audit evidence also identifies a Llama core-count/artifact identity
  mismatch that must be resolved before serving reuse.
- The real Trn2 benchmark completed the staged TP4/CP1 path at 320x512x61,
  20 steps, and guidance 6.0 with a finite output tensor. It did not co-load the
  Serving adapter's host CLIP/VAE plus resident W4 Llama/DiT plan.

### Implemented serving posture

- Represent the model as four logical stages: `clip`, `llama`, `denoiser`, and
  `decoder`.
- Restrict the first experiment to TP=4/CP=1.
- Use explicit host CLIP and host VAE placement; do not load their current W1
  artifacts in the resident W4 process.
- Keep Llama and DiT in the resident W4 world and measure the full combined
  host/HBM footprint before production enablement.

### Residency conclusion

**Provisionally treated as fitting one process using host CLIP/VAE and W4
Llama/DiT, but still unproven on the target.** A full prompt-to-MP4 comparison,
co-load, recovery, and memory experiment is a release gate.

## HunyuanVideo 1.5

### Observed

- The HunyuanVideo entry in the README supported-model table is HunyuanVideo
  1.0. It is not evidence for HunyuanVideo 1.5 support.
- `difflet/cli/orchestrators/hunyuan_video_15.py` is a scaffold.
- Both `compile()` and `generate()` raise `NotImplementedError`.
- The planned path includes Qwen2.5-VL, ByT5 glyph, image-semantic conditioning,
  a segmented transformer, and VAE decode.
- The default fixed shape is 480x848x121 and TP=4; CP support is deferred.
- Its Trn2 benchmark entry has no compile, e2e, or output result and remains
  explicitly `pending`.

### Future requirement

- Finish offline compilation and generation before serving design is frozen.
- Start with one combined conditioning stage unless encoder artifact lifecycles
  require separate stages, followed by `denoise` and `decode_export`.
- Revisit stage count after real encoder outputs and memory lifetimes exist.

### Residency conclusion

**Not ready for serving.** A three-to-five-stage estimate is architectural only,
not an implementation commitment or memory-fit claim.

## Comparative decision matrix

| Model | Real Trn2 offline path | Uniform resident Neuron topology | Host decode available | Suggested priority |
| --- | --- | --- | --- | ---: |
| LTX-2 | Yes, fixed-profile direct CLI | Provisional hybrid adapter | Yes, explicit | 1 |
| Wan 2.1 | Yes, fixed-profile staged CLI | Provisional W4 adapter with host VAE | Yes, explicit | 2 |
| HunyuanVideo 1.0 | Yes, fixed-profile staged CLI | Provisional W4 Llama/DiT with host CLIP/VAE | Yes, explicit | 3 |
| Wan 2.2 | Partial single-expert path only | Unknown | Yes | 4 |
| HunyuanVideo 1.5 | No; benchmark pending | Unknown | Future | 5 |

Host placement in this table is an adapter-specific execution choice. Difflet
does not currently provide a generic CPU-offload switch that can spill arbitrary
Neuron weights or runtime buffers into the 128 GB host memory.
