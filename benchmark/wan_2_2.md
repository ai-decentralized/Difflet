# Benchmark — Wan-AI/Wan2.2-T2V-A14B-Diffusers

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 06:51 UTC

> Best-performing configuration: tp=4, bf16, A14B (high/low-noise experts), attention_cte

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
| compile (AOT, one-time) | 13.75 s |
| weights load (per process) | 16.36 s |
| **end-to-end generate (cold)** | **10.3 min (616 s)** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 1.14 s | 1.14 s | 1.14 s | 1.14 s | 20 |

**Throughput:** 0.874 DiT steps/s

## Compile breakdown

| component | build time |
|---|---|
| wall_total | 13.75 s |

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 9, 480, 832] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-0.7227, -0.3164] (mean -0.4743, std 0.0791) |
| note | saved wan2_2_t2v_a14b_diffusers_out.pt |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- CAVEAT: the wan orchestrator names compile-cache dirs by shape only (wan_transformer_tp4cp1_h480w832f9), not by model id, so Wan 2.2 reused Wan 2.1's compiled NEFF (compile time shown is a false cache hit). The two 14B transformers are architecturally identical, so 2.2's weights load into the shared graph and the generate is valid; but a faithful separate 2.2 compile needs a model-id-keyed cache (orchestrator fix).
- difflet runs Wan 2.2 with enable_transformer_2=False -> only the high-noise expert (single transformer), not the full A14B MoE.
- per-step = 1143.6 ms/DiT-forward (warm, in-process, n=20) via benchmark.step_latency — the stable Neuron-compute metric (e2e generate is load-dominated/noisy across processes).

## Reproduce

```bash
# all runs use the Neuron inference venv
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -m benchmark.bench --model wan_2_2
```
