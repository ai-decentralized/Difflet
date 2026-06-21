#!/usr/bin/env python3
"""Cheap CPU smoke for the host refiner (cclog 86) before the Trainium recompile.

Builds a standalone HunyuanVideo15TokenRefiner, loads context_embedder.* weights from the
checkpoint, and ALSO compares against the diffusers model's own context_embedder on the
real bundle — confirming the host split is exact (cosine ~1.0) and outputs inner_dim=2048.
"""

from __future__ import annotations

import glob
import os
import sys
import types
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
from safetensors.torch import load_file  # noqa: E402

MD = Path("/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_t2v/snapshots/f4dbc4a1efa4ac8ea56680cdf79d9f455105e814")
TRANS = MD / "transformer"
BUNDLE = ROOT / ".difflet-cache" / "hunyuan15_dit_inputs" / "real_320x512x61_4step.safetensors"


def build_host_refiner(cfg):
    from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15TokenRefiner

    refiner = HunyuanVideo15TokenRefiner(
        in_channels=int(cfg["text_embed_dim"]),
        num_attention_heads=int(cfg["num_attention_heads"]),
        attention_head_dim=int(cfg["attention_head_dim"]),
        num_layers=int(cfg["num_refiner_layers"]),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
    )
    state = {}
    for f in sorted(glob.glob(str(TRANS / "*.safetensors"))):
        for k, v in load_file(f).items():
            if k.startswith("context_embedder."):
                state[k[len("context_embedder."):]] = v
    missing, unexpected = refiner.load_state_dict(state, strict=False)
    print(f"[smoke] loaded {len(state)} weights; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    assert not missing and not unexpected, (missing[:4], unexpected[:4])

    def _host_keyonly(rself, hs, temb, am=None):
        mask = None
        if am is not None:
            bs, seq = int(am.shape[0]), int(am.shape[1])
            neg = torch.finfo(hs.dtype).min
            mask = torch.zeros((bs, 1, 1, seq), dtype=hs.dtype).masked_fill(~am.bool().view(bs, 1, 1, seq), neg)
        for blk in rself.refiner_blocks:
            hs = blk(hs, temb, mask)
        return hs

    refiner.token_refiner.forward = types.MethodType(_host_keyonly, refiner.token_refiner)
    return refiner.to(torch.float32).eval()


def main() -> int:
    import json

    cfg = json.load(open(TRANS / "config.json"))
    t = load_file(str(BUNDLE))
    mllm = t["encoder_hidden_states"].to(torch.float32)
    ts = torch.tensor([1000.0], dtype=torch.float32)
    mask = t["encoder_attention_mask"].to(torch.int64)
    print(f"[smoke] mllm in {tuple(mllm.shape)}  mask valid={int(mask.sum())}", flush=True)

    host = build_host_refiner(cfg)
    with torch.no_grad():
        out = host(mllm, ts, mask)
    print(f"[smoke] host refiner out {tuple(out.shape)} finite={bool(torch.isfinite(out).all())}", flush=True)

    # reference: the diffusers model's OWN context_embedder (symmetric mask) on valid rows
    from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15Transformer3DModel

    model = HunyuanVideo15Transformer3DModel.from_pretrained(TRANS, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ref = model.context_embedder(mllm, ts, mask)
    # compare only VALID rows (padding-query rows differ by design; masked downstream)
    v = mask.bool()[0]
    cos = float(F.cosine_similarity(out[0][v].reshape(1, -1), ref[0][v].reshape(1, -1), dim=1).item())
    print(f"[smoke] cosine(host vs model context_embedder, valid rows)={cos:.6f}", flush=True)
    print(f"[smoke] VERDICT: {'OK' if cos > 0.999 and out.shape[-1] == int(cfg['num_attention_heads'])*int(cfg['attention_head_dim']) else 'CHECK'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
