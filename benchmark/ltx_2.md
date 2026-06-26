# Benchmark — Lightricks/LTX-2

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 15:18 UTC

> Best-performing configuration: tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF)

## Configuration

| key | value |
|---|---|
| model type | ltx_2 |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 704, 'num_frames': 49} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | — |
| weights load (per process) | 3.0 min (183 s) |
| **end-to-end generate (cold)** | **4.9 min (293 s)** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 473.0 ms | 473.0 ms | 473.0 ms | 473.0 ms | 1 |

**Throughput:** 2.114 DiT steps/s

## Output validity

| field | value |
|---|---|
| shape | [1, 49, 3, 480, 704] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [0.0000, 0.8828] (mean 0.3695, std 0.1457) |
| note | saved ltx_2_out.pt |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- Default registry shape 512x768x121 also compiles; 480x704x49 used here as the representative fast shape. CFG (guidance>1) needs a batch-2 NEFF.
- per-step = 0.473 s/DiT-forward (warm, in-process) from scripts/ltx_2_transformer_parity.py — the stable Neuron-compute metric; video cosine vs CPU = 0.99992 (lossless). e2e generate is load-dominated and noisy across processes (text-encoder load swings ~80-375 s with disk/page-cache state); treat e2e as indicative, per-step as the optimal metric.

## Reproduce

```bash
# all runs use the Neuron inference venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -m benchmark.bench --model ltx_2
```
