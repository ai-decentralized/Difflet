#!/usr/bin/env python3
"""Cheap CPU pre-check for the key-only-mask fix (before the Trainium recompile).

Runs the diffusers HV-1.5 model on CPU (fp32) with (a) the default attention and
(b) the key-only-mask processor, on the SAME inputs + real masks, and compares the
noise_pred. Expect: both finite, cosine ~1.0 (the fix only changes discarded
padding-QUERY rows; the image output should be unchanged). Confirms the processor
is correct before spending ~15 min compiling.
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

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

from nova.backends.trainium.hunyuan_video.backbone15 import (  # noqa: E402
    _HunyuanVideo15KeyMaskAttnProcessor,
)

MD = Path("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_t2v/snapshots/f4dbc4a1efa4ac8ea56680cdf79d9f455105e814")
BUNDLE = ROOT / ".nova-cache" / "hunyuan15_dit_inputs" / "real_320x512x61_4step.safetensors"


def main() -> int:
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    t = load_safetensors_file(str(BUNDLE), device="cpu")
    dtype = torch.float32
    lf = 1
    hs = torch.cat([torch.randn(1, 32, lf, 20, 32), torch.zeros(1, 33, lf, 20, 32)], 1).to(dtype)
    kw = dict(
        hidden_states=hs,
        timestep=torch.tensor([1000.0], dtype=dtype),
        encoder_hidden_states=t["encoder_hidden_states"].to(dtype),
        encoder_attention_mask=t["encoder_attention_mask"].to(torch.int64),
        timestep_r=None,
        encoder_hidden_states_2=t["encoder_hidden_states_2"].to(dtype),
        encoder_attention_mask_2=t["encoder_attention_mask_2"].to(torch.int64),
        image_embeds=t["image_embeds"].to(dtype),
        return_dict=False,
    )

    m = HunyuanVideo15Transformer3DModel.from_pretrained(MD / "transformer", torch_dtype=dtype).eval()
    with torch.no_grad():
        out_default = m(**kw)[0]
    print(f"[chk] default: finite={bool(torch.isfinite(out_default).all())}", flush=True)

    import types

    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15IndividualTokenRefiner,
        HunyuanVideo15Transformer3DModel,
    )

    from nova.backends.trainium.hunyuan_video.backbone15 import _hv15_static_reorder_forward

    # option A: static-reorder forward (the monolithic-fix). out_default above used the
    # ORIGINAL dynamic-reorder forward; patch the class now to compare.
    HunyuanVideo15Transformer3DModel.forward = _hv15_static_reorder_forward

    proc = _HunyuanVideo15KeyMaskAttnProcessor()
    for block in m.transformer_blocks:
        block.attn.set_processor(proc)

    def _keyonly_refiner_forward(rself, hidden_states, temb, attention_mask=None):
        self_attn_mask = None
        if attention_mask is not None:
            bs = attention_mask.shape[0]
            seq = attention_mask.shape[1]
            am = attention_mask.to(hidden_states.device).bool()
            self_attn_mask = am.view(bs, 1, 1, seq).repeat(1, 1, seq, 1)
        for block in rself.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)
        return hidden_states

    for mod in m.modules():
        if isinstance(mod, HunyuanVideo15IndividualTokenRefiner):
            mod.forward = types.MethodType(_keyonly_refiner_forward, mod)

    with torch.no_grad():
        out_key = m(**kw)[0]
    finite = bool(torch.isfinite(out_key).all())
    cos = float(
        F.cosine_similarity(out_default.reshape(1, -1), out_key.reshape(1, -1), dim=1).item()
    )
    md = float((out_default - out_key).abs().max().item())
    print(f"[chk] key-mask: finite={finite}  cosine_vs_default={cos:.6f}  max_abs_diff={md:.3e}", flush=True)
    print(f"[chk] VERDICT: {'OK (matches default, finite)' if (finite and cos > 0.999) else 'CHECK — differs from default'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
