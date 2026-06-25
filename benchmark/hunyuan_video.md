# Benchmark — hunyuanvideo-community/HunyuanVideo

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 07:05 UTC

> Best-performing configuration: tp=4, bf16, attention_cte

## Configuration

| key | value |
|---|---|
| model type | hunyuan_video |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 320, 'width': 512, 'num_frames': 61} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 40.2 min (2413 s) |
| weights load (per process) | 3.5 min (210 s) |
| **end-to-end generate (cold)** | **8.8 min (527 s)** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |

## Compile breakdown

| component | build time |
|---|---|
| component_0 | 720.4 ms |
| component_1 | 56.19 s |
| component_2 | 9.9 min (595 s) |
| wall_total | 40.2 min (2413 s) |

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 61, 320, 512] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-2.3281, 2.1562] (mean -0.5328, std 0.5875) |
| note | saved hunyuanvideo_out.pt |

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
python -m benchmark.bench --model hunyuan_video
```
