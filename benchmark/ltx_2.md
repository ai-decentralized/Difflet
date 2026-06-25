# Benchmark — Lightricks/LTX-2

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 06:09 UTC

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
| compile (AOT, one-time) | 8.2 min (495 s) |
| weights load (per process) | 82.47 s |
| **end-to-end generate (cold)** | **4.9 min (293 s)** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |

## Compile breakdown

| component | build time |
|---|---|
| transformer(LTX2VideoTransformer3DModel) | 8.2 min (495 s) |

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
- compile time from a dedicated clean run (no concurrent downloads); the e2e generate above was measured clean with --skip-compile.
- per-step DiT forward (warm, attention_cte self-attn) = 0.473 s, from the transformer parity harness (Neuron-only, download-immune); video cosine vs CPU reference = 0.99992 (lossless).

## Reproduce

```bash
# all runs use the Neuron inference venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -m benchmark.bench --model ltx_2
```
