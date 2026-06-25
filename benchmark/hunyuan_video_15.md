# Benchmark — hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v

**Status:** pending — compile not implemented in difflet (not benchmarked)  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device

> Planned best-perf config: tp=4, bf16, attention_cte + MX precision ops.
> 480×848, 121 frames.

## Why no numbers

Weights are downloaded (≈50 GB, `weights ready`), but `difflet compile` for this
model raises:

```
NotImplementedError: difflet compile --model-id
hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v is not yet implemented.
HunyuanVideo 1.5 requires Qwen2.5-VL, ByT5 glyph, and image-semantic text
encoders (HunyuanVideo15DiTInputBundle) — stage logic TBD.
```

(`difflet/cli/orchestrators/hunyuan_video_15.py::compile`). The DiT backbone +
VAE are partially in the tree (`backbone15.py`, `segmented15.py`, `vae15.py`, and
the model is registered), but the **end-to-end orchestrator** (the three text
encoders + input-bundle assembly) is not finished, so a full compile→generate
cannot run yet.

## To unblock

Implement the HunyuanVideo-1.5 compile/generate stages in
`difflet/cli/orchestrators/hunyuan_video_15.py` (text-encoder + DiT + VAE stages,
mirroring `hunyuan_video.py`), then:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -m benchmark.bench --model hunyuan_video_15 --skip-download
```

## Configuration (planned)

| key | value |
|---|---|
| model type | hunyuan_video_15 |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | 480×848, 121 frames |
| steps | 20 |

## Notes

- This is the only registered model whose difflet orchestrator is an explicit
  stub. The others — flux, qwen_image, wan, hunyuan_video, ltx_2 — have working
  compile paths; FLUX is blocked only by HF gating, not missing code.
