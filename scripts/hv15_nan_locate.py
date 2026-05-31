#!/usr/bin/env python3
"""Goal: find the op that produces the HV-1.5 NaN.

Runs the diffusers HunyuanVideo15Transformer3DModel on CPU EAGER (no Neuron compiler)
at a tiny shape, with a forward hook on EVERY submodule that flags the first module
whose output contains NaN/Inf. Runs in both fp32 and bf16.

Decision:
  - first-NaN module name  -> the offending op.
  - CPU-eager fp32 finite, bf16 NaN -> pure bf16 precision.
  - CPU-eager (both) NaN  -> model-math bug (not compiler/bf16); since Trainium also
    NaNs, the bug is in the math, reproduced on CPU.
  - CPU-eager finite, Trainium NaN -> compiler-introduced (would need the Trainium path).

Prime hypothesis: the all-zero ByT5 stream (encoder_hidden_states_2) -> bias-free
k/q proj -> RMSNorm (norm_added_k/q) -> 0/0.
"""

from __future__ import annotations

import os
import sys
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

import json  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

MODEL_DIR = Path("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_t2v/snapshots/f4dbc4a1efa4ac8ea56680cdf79d9f455105e814")
BUNDLE = ROOT / ".nova-cache" / "hunyuan15_dit_inputs" / "real_320x512x61_4step.safetensors"


def _bad(t):
    if not torch.is_tensor(t):
        return False
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def _any_bad(out):
    if torch.is_tensor(out):
        return _bad(out)
    if isinstance(out, (tuple, list)):
        return any(_any_bad(o) for o in out)
    return False


def run(dtype, relax_mask2: bool, zero_byt5: bool, relax_mask1: bool = False):
    import torch.nn as nn
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    cfg = json.loads((MODEL_DIR / "transformer" / "config.json").read_text())
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    cfg.pop("_name_or_path", None)
    model = HunyuanVideo15Transformer3DModel.from_pretrained(
        MODEL_DIR / "transformer", torch_dtype=dtype
    ).eval()

    tns = load_safetensors_file(str(BUNDLE), device="cpu")
    nf = 1
    lf = 1
    lh, lw = 20, 32
    latent = torch.randn(1, 32, lf, lh, lw, dtype=dtype)
    cond = torch.zeros(1, 33, lf, lh, lw, dtype=dtype)
    hs = torch.cat([latent, cond], dim=1)
    ehs = tns["encoder_hidden_states"].to(dtype)
    eam = tns["encoder_attention_mask"].to(torch.int64)
    if relax_mask1:
        eam = torch.ones_like(eam)  # all conditioning tokens valid -> no all-masked query row
    ehs2 = tns["encoder_hidden_states_2"].to(dtype)
    if not zero_byt5:
        ehs2 = ehs2 + 0.01  # tiny non-zero, to test the zero-stream hypothesis
    eam2 = tns["encoder_attention_mask_2"].to(torch.int64)
    if relax_mask2 and int(eam2.sum()) == 0:
        eam2 = torch.ones_like(eam2)
    img = tns["image_embeds"].to(dtype)
    # use_meanflow=False for this model -> the trace module + compiled NEFF pass
    # timestep_r=None (the diffusers forward branches on `timestep_r is not None`
    # and would crash with no time_proj_r). Match the real path.
    tr = None if not bool(getattr(model.config, "use_meanflow", False)) else tns["timestep_r"].to(dtype)

    first = {"name": None}
    handles = []

    def mk(name):
        def hook(mod, inp, out):
            if first["name"] is None and _any_bad(out):
                in_bad = any(_bad(x) for x in inp if torch.is_tensor(x))
                first["name"] = f"{name}  [{type(mod).__name__}]  input_already_bad={in_bad}"
        return hook

    for name, mod in model.named_modules():
        if name and len(list(mod.children())) == 0:  # leaf modules
            handles.append(mod.register_forward_hook(mk(name)))

    with torch.no_grad():
        out = model(
            hidden_states=hs,
            timestep=torch.tensor([1000.0], dtype=dtype),
            encoder_hidden_states=ehs,
            encoder_attention_mask=eam,
            timestep_r=tr,
            encoder_hidden_states_2=ehs2,
            encoder_attention_mask_2=eam2,
            image_embeds=img,
            return_dict=False,
        )[0]
    for h in handles:
        h.remove()
    return _any_bad(out), first["name"]


def main() -> int:
    # Test matrix (fp32, CPU eager): isolate the cause.
    #  A) masks as-is (stream1 = 13 valid)          -> expect NaN at the attention
    #  B) relax BOTH masks to all-valid             -> expect FINITE (confirms the
    #     symmetric-mask all-masked-query-row -> softmax 0/0 cause)
    #  C) relax only mask2 (ByT5)                   -> expect NaN (stream1 padding remains)
    cases = [
        ("fp32 masks=as-is", torch.float32, False, False),
        ("fp32 masks=BOTH-relaxed", torch.float32, True, True),
        ("fp32 masks=only-mask2-relaxed", torch.float32, False, True),
        ("bf16 masks=BOTH-relaxed", torch.bfloat16, True, True),
    ]
    for tag, dtype, relax1, relax2 in cases:
        try:
            bad, first = run(dtype, relax_mask2=relax2, zero_byt5=True, relax_mask1=relax1)
            print(f"[locate] {tag}: out_bad={bad}  first_nan_op={first}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[locate] {tag}: EXC {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
