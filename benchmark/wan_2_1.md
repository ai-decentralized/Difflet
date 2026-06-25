# Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 06:14 UTC

> Best-performing configuration: tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline

## Configuration

| key | value |
|---|---|
| model type | wan |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 832, 'num_frames': 9} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 108.5 min (6507 s) |
| weights load (per process) | 14.97 s |
| **end-to-end generate (cold)** | **11.9 min (713 s)** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |

## Compile breakdown

| component | build time |
|---|---|
| text_encoder(UMT5) | 31.00 s |
| transformer(WanTransformer3DModel, 14B) | 7.2 min (431 s) |
| vae_decoder | 100.8 min (6045 s) |

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 9, 480, 832] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-0.8711, -0.2578] (mean -0.5518, std 0.0840) |
| note | saved wan2_1_t2v_14b_diffusers_out.pt |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- compile time from a dedicated clean run; the VAE decoder dominates (~100 min) — the Wan video VAE is conv-heavy and slow on neuronx-cc.
- single-transformer (no MoE); attention is unmasked -> attention_cte.

## Reproduce

```bash
# all runs use the Neuron inference venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -m benchmark.bench --model wan_2_1
```
