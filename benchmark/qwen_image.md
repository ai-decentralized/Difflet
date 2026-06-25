# Benchmark — Qwen/Qwen-Image

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 06:26 UTC

> Best-performing configuration: tp=4, bf16, joint attention via attention_cte

## Configuration

| key | value |
|---|---|
| model type | qwen_image |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 1024, 'width': 1024, 'num_frames': None} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 13.4 min (801 s) |
| weights load (per process) | 6.36 s |
| **end-to-end generate (cold)** | **8.3 min (496 s)** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |

## Compile breakdown

| component | build time |
|---|---|
| component_0 | 96.17 s |
| component_1 | 4.5 min (269 s) |
| component_2 | 6.9 min (412 s) |
| wall_total | 13.4 min (801 s) |

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 1024, 1024] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-0.9531, 0.8867] (mean 0.1317, std 0.5018) |
| note | saved qwen_image_out.pt |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Reproduce

```bash
# all runs use the Neuron inference venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -m benchmark.bench --model qwen_image
```
