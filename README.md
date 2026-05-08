# Nova — Trainium3 Diffusion Inference

A focused, lean inference framework for diffusion models (DiT-based image and
video generation) on AWS Trainium3.

## Status

**Pre-alpha.** Repository scaffolding only. Code fork from
[neuronx-distributed-inference](https://github.com/aws-neuron/neuronx-distributed-inference)
in progress (M0).

## Design

- **Positioning**: like xDiT — focused, Python API + `torchrun`, no scheduler / HTTP server.
- **Internals**: NxDI patterns — subclass HuggingFace pipelines, swap transformer
  module with a Neuron-compiled backbone, AOT `compile()` + `load()`.
- **Strategy**: **fork-and-own**. Diffusion-specific code (Layer 1) and shared
  inference infrastructure (Layer 2) from NxDI are forked into this repo.
  Kernel/parallel toolchain (Layer 3: NXD, nkilib, NKI, neuronx-cc) remains a pip
  dependency.

See [`docs/architecture.md`](docs/architecture.md) (TBD) and the design plan in
`/home/ubuntu/.claude/plans/crystalline-percolating-sundae.md` for details.

## Quick start (target API, not yet implemented)

```python
from nova import NovaPipeline, NovaParallelConfig

pipe = NovaPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    parallel=NovaParallelConfig(tp_degree=8, cp_enabled=True),
)
out = pipe(prompt="A cat walking", num_inference_steps=30, num_frames=49)
out.save("out.mp4")
```

Launch:

```bash
torchrun --nproc_per_node=8 user_run.py
```

## Supported models (target)

| Model | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| Flux.1-dev (forked from NxDI) | ✓ | ✓ | | | |
| Wan 2.2 T2V/I2V | | | ✓ | | |
| HunyuanVideo + 1.5 | | | | ✓ | |
| Qwen-Image, LTX-2, Z-Image | | | | | ✓ |

## License

Apache-2.0. Includes code derived from NeuronX Distributed Inference under the
same license. See [`NOTICE`](NOTICE).
