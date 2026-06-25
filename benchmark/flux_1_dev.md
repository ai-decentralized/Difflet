# Benchmark — black-forest-labs/FLUX.1-dev

**Status:** pending — gated repository (not benchmarked)  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device

> Best-performing configuration (planned): tp=4 (registry default tp=8 → 4 on
> trn2.3xlarge), bf16, attention_cte. 1024×1024, 28 steps, guidance 3.5.

## Why no numbers yet

`FLUX.1-dev` is a **gated** Hugging Face repo. `difflet download` fails with:

```
huggingface_hub.errors.GatedRepoError: 401 ... Access to model
black-forest-labs/FLUX.1-dev is restricted. You must have access to it and be
authenticated to access it.
```

No `HF_TOKEN` / `~/.cache/huggingface/token` is present on this box, so the
weights cannot be fetched. Everything else is ready: FLUX is wired into difflet
(`difflet/models/flux/modeling_flux.py`, routed through `attention_cte`) and the
benchmark matrix entry exists (`benchmark/models.py::MATRIX["flux_1_dev"]`).

## To unblock and run

```bash
# 1. accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev
# 2. authenticate
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
huggingface-cli login        # or: export HF_TOKEN=hf_xxx
# 3. run the benchmark (download + compile + generate + report)
python -m benchmark.bench --model flux_1_dev
```

This will overwrite this file with the measured report (compile time, weights
load, end-to-end image latency, per-step latency, peak HBM, output validity).

## Configuration (planned)

| key | value |
|---|---|
| model type | flux |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | 1024×1024 (image) |
| steps | 28 |
| guidance | 3.5 |

## Notes

- Registry default is `tp=8`; overridden to `tp=4` because trn2.3xlarge has a
  single Neuron device (4 cores). On a trn2.48xlarge, `tp=8` would be the
  best-perf config.
- FLUX is an image model (no VAE-temporal / video stages), so its end-to-end is
  text-encode (CPU) + N DiT steps (Neuron) + VAE decode (1 image).
