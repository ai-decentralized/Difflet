#!/usr/bin/env python3
"""B0 sanity (cclog 84): time ONE full-shape Qwen DiT forward on CPU.

The CPU eager cache-form ablation (Option B) needs ~50 steps x N bundles x 3
cache-forms of diffusers CPU forwards. The 1-4h estimate rests on an unvalidated
CPU-vs-tp4 slowdown. This times a single 1024^2 forward so we know whether the
full ablation is 1-4h (proceed on CPU) or must drop steps/bundles or move to a
single Trainium run.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import torch  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

MODEL_DIR = Path("/home/ubuntu/.cache/huggingface/hub/qwen-image-real/transformer")
BUNDLE = ROOT / ".nova-cache" / "qwen_image_dit_inputs" / "m9_calib_50step" / "calibration_00_a_busy_city_street_with_50step.safetensors"


def main() -> int:
    torch.set_num_threads(os.cpu_count() or 8)
    dtype = torch.bfloat16
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    print(f"[b0] loading diffusers QwenImageTransformer2DModel (CPU, {dtype}) ...", flush=True)
    t0 = time.perf_counter()
    model = QwenImageTransformer2DModel.from_pretrained(
        MODEL_DIR, torch_dtype=dtype, local_files_only=True
    ).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[b0] loaded {n_params/1e9:.1f}B params in {time.perf_counter()-t0:.1f}s", flush=True)

    tensors = load_safetensors_file(str(BUNDLE), device="cpu")
    hs = tensors["latents_init"].to(dtype)
    ehs = tensors["encoder_hidden_states"].to(dtype)
    mask = tensors["encoder_hidden_states_mask"].to(torch.bool)
    ts = (tensors["timesteps"][:1].to(dtype) / 1000.0)  # diffusers feeds timestep/1000
    guidance_embeds = bool(getattr(model.config, "guidance_embeds", False))
    guidance = None
    print(f"[b0] shapes hs={tuple(hs.shape)} ehs={tuple(ehs.shape)} guidance_embeds={guidance_embeds}", flush=True)

    def one_forward():
        with torch.no_grad():
            out = model(
                hidden_states=hs,
                timestep=ts,
                encoder_hidden_states=ehs,
                encoder_hidden_states_mask=mask,
                guidance=guidance,
                img_shapes=[[(1, 64, 64)]],
                txt_seq_lens=[int(mask.sum().item())],
                return_dict=False,
            )[0]
        return out

    print("[b0] forward #1 (cold)...", flush=True)
    t = time.perf_counter()
    o1 = one_forward()
    f1 = time.perf_counter() - t
    print(f"[b0] forward #1 = {f1:.1f}s  out={tuple(o1.shape)}", flush=True)

    t = time.perf_counter()
    _ = one_forward()
    f2 = time.perf_counter() - t
    print(f"[b0] forward #2 (warm) = {f2:.1f}s", flush=True)

    # extrapolate the Option-B ablation cost
    for label, steps, bundles, forms in (("full 50x3x3", 50, 3, 3), ("reduced 20x1x3", 20, 1, 3)):
        total = f2 * steps * bundles * forms
        print(f"[b0] est Option-B {label}: {total/60:.0f} min ({total/3600:.1f} h)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
