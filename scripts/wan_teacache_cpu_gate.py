#!/usr/bin/env python3
"""CPU TeaCache signal gate for Wan 2.2 (high-noise ``transformer`` stage).

Question this script answers
----------------------------
TeaCache skips diffusion steps when a cheap *signal* derived from the block-0
self-attention AdaLN modulated input is small. That only works if the signal
**correlates** with the model's real per-step output change. This gate runs a
short CPU flow-matching denoise loop on Nova's ``WanTransformer3DModel`` and
measures the Pearson correlation between

    signal[i]     = mean|mod_inp[i] - mod_inp[i-1]| / mean|mod_inp[i-1]|
    true_delta[i] = mean|noise_pred[i] - noise_pred[i-1]| / mean|noise_pred[i-1]|

A high Pearson (say >= ~0.8) means adaptive TeaCache is viable for Wan 2.2.

Notes / verified facts
----------------------
* Modeling: ``nova/models/wan/modeling_wan.py`` :: ``WanTransformer3DModel``.
  The AdaLN modulation comes from ``timestep_proj`` (timestep ONLY); text
  embeddings enter only via cross-attention (attn2), NOT the block-0
  self-attn modulation. So the signal is purely timestep-driven and the
  encoder_hidden_states only need to be *constant* across steps for the gate.
* ``WanTransformerBlock.forward(hidden_states, encoder_hidden_states, temb,
  rotary_emb)`` receives ``temb == timestep_proj`` shaped ``(B, 6, inner_dim)``
  (ndim==3) and computes
  ``shift_msa, scale_msa, ... = (scale_shift_table + temb.float()).chunk(6, 1)``
  then ``norm1(hidden)*(1+scale_msa)+shift_msa`` — that last quantity is the
  TeaCache "modulated input".
* We capture block-0's ``(hidden_states, temb)`` via a ``forward_pre_hook``
  and recompute the modulated input with block0.norm1 / block0.scale_shift_table.

Embeds source: this gate uses a FIXED RANDOM ``encoder_hidden_states`` tensor
(option b in the task). Loading the 11GB UMT5 text encoder is unnecessary
because the measured signal is timestep-driven and the embeds are held
constant across all steps. The chosen source is logged at startup.

Run:
    PATH="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH" \
    NOVA_BACKEND=cpu PYTHONPATH=/home/ubuntu/nova \
    /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \
        scripts/wan_teacache_cpu_gate.py

(The script self-execs into the neuron venv python and sets PATH/NOVA_BACKEND
if needed, so a bare ``python scripts/wan_teacache_cpu_gate.py`` also works.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]

WAN_MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WAN_SUBFOLDER = "transformer"  # high-noise stage
LATENT_CACHE = ROOT / ".nova-cache" / "wan_smoke_latents.pt"
DEFAULT_LATENT_SHAPE = (1, 16, 3, 60, 104)


def ensure_runtime_python() -> None:
    """Re-exec into the neuron venv python with PATH + NOVA_BACKEND set.

    torch_xla's init shells out to ``libneuronpjrt-path`` (only on the venv
    PATH), and ``neuronx_distributed`` is imported transitively even for the
    CPU op backend, so PATH must include the venv bin. ``NOVA_BACKEND=cpu``
    selects the numerical CPU reference ops.
    """
    need_python = Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists()
    venv_bin = str(NEURON_VENV / "bin")
    need_path = venv_bin not in os.environ.get("PATH", "").split(os.pathsep)
    need_backend = os.environ.get("NOVA_BACKEND") != "cpu"
    try:
        import torch  # noqa: F401

        torch_ok = True
    except ModuleNotFoundError:
        torch_ok = False

    if torch_ok and not need_path and not need_backend:
        return
    if not NEURON_PYTHON.exists():
        # Best effort: just set the env knobs in-process.
        os.environ["NOVA_BACKEND"] = "cpu"
        return

    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["NOVA_BACKEND"] = "cpu"
    target = NEURON_PYTHON if need_python else Path(sys.executable)
    os.execve(str(target), [str(target), *sys.argv], env)


ensure_runtime_python()

import importlib  # noqa: E402

import torch  # noqa: E402

# Import the modeling module DIRECTLY (not via ``nova.models.wan`` package
# __init__, which imports application.py -> neuronx_distributed eagerly and is
# unnecessary here).
_wan = importlib.import_module("nova.models.wan.modeling_wan")
WanTransformer3DModel = _wan.WanTransformer3DModel
WanTransformerConfig = _wan.WanTransformerConfig

_ckpt = importlib.import_module("nova.models.wan.checkpoint.backbone")
convert_backbone_state_dict = _ckpt.convert_backbone_state_dict


LOG = "[wan-gate]"


def log(*args) -> None:
    print(LOG, *args, flush=True)


def resolve_transformer_dir(model_id: str, subfolder: str) -> Path:
    """Return a local dir holding ``<subfolder>/config.json`` + safetensors.

    Downloads only the transformer subfolder from HF if not already cached.
    """
    from huggingface_hub import snapshot_download

    log(f"resolving {model_id} :: {subfolder}/ (downloading subfolder if needed)")
    snap = snapshot_download(
        model_id,
        allow_patterns=[f"{subfolder}/*"],
    )
    tdir = Path(snap) / subfolder
    cfg = tdir / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"missing {cfg} after download")
    return tdir


def load_state_dict_from_dir(tdir: Path) -> dict:
    """Load a (possibly sharded) safetensors state dict from a directory."""
    from safetensors.torch import load_file

    index = tdir / "diffusion_pytorch_model.safetensors.index.json"
    state: dict = {}
    if index.exists():
        weight_map = json.load(open(index))["weight_map"]
        shards = sorted(set(weight_map.values()))
        log(f"loading {len(shards)} safetensors shards")
        for i, shard in enumerate(shards):
            log(f"  shard {i + 1}/{len(shards)}: {shard}")
            state.update(load_file(str(tdir / shard)))
    else:
        single = tdir / "diffusion_pytorch_model.safetensors"
        if not single.exists():
            sts = sorted(tdir.glob("*.safetensors"))
            if not sts:
                raise FileNotFoundError(f"no safetensors under {tdir}")
            single = sts[0]
        log(f"loading single safetensors: {single.name}")
        state.update(load_file(str(single)))
    return state


def build_transformer(tdir: Path, dtype: torch.dtype) -> WanTransformer3DModel:
    raw_cfg = json.load(open(tdir / "config.json"))
    config = WanTransformerConfig.from_diffusers_dict(raw_cfg)
    log(
        f"config: layers={config.num_layers} heads={config.num_attention_heads} "
        f"head_dim={config.attention_head_dim} inner_dim={config.inner_dim} "
        f"ffn_dim={config.ffn_dim} text_dim={config.text_dim}"
    )
    log("instantiating WanTransformer3DModel on CPU (meta-free) ...")
    model = WanTransformer3DModel(config, dtype=dtype)

    state = load_state_dict_from_dir(tdir)
    state = convert_backbone_state_dict(state, config=config)
    state = {k: v.to(dtype) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log(f"WARNING: {len(missing)} missing keys (first few): {missing[:5]}")
    if unexpected:
        log(f"WARNING: {len(unexpected)} unexpected keys (first few): {unexpected[:5]}")
    model = model.to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log("transformer loaded and set to eval()")
    return model


def make_inputs(dtype: torch.dtype, text_seq_len: int, text_dim: int):
    """Latents (from cache or random) + a FIXED random encoder_hidden_states."""
    g = torch.Generator().manual_seed(1234)
    if LATENT_CACHE.exists():
        latents = torch.load(LATENT_CACHE, map_location="cpu")
        if not torch.is_tensor(latents):
            latents = next(v for v in latents.values() if torch.is_tensor(v))
        latents = latents.to(dtype)
        log(f"latents_init: loaded cached {tuple(latents.shape)} from {LATENT_CACHE.name}")
    else:
        latents = torch.randn(DEFAULT_LATENT_SHAPE, generator=g).to(dtype)
        log(f"latents_init: random {tuple(latents.shape)} (no cache found)")

    b = latents.shape[0]
    encoder_hidden_states = torch.randn(
        b, text_seq_len, text_dim, generator=g
    ).to(dtype)
    log(
        f"embed source: FIXED RANDOM encoder_hidden_states "
        f"{tuple(encoder_hidden_states.shape)} (constant across all steps; "
        "block-0 self-attn modulation is timestep-only so text embeds need "
        "only be constant)"
    )
    return latents, encoder_hidden_states


class Block0Capture:
    """forward_pre_hook on blocks[0] capturing (hidden_states, temb)."""

    def __init__(self, block):
        self.block = block
        self.hidden_states = None
        self.temb = None
        self._handle = block.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        # WanTransformerBlock.forward(hidden_states, encoder_hidden_states,
        #                             temb, rotary_emb)
        names = ["hidden_states", "encoder_hidden_states", "temb", "rotary_emb"]
        bound = dict(zip(names, args))
        bound.update(kwargs)
        self.hidden_states = bound["hidden_states"].detach().clone()
        self.temb = bound["temb"].detach().clone()
        return None

    def modulated_input(self) -> torch.Tensor:
        """Recompute norm1(hidden)*(1+scale_msa)+shift_msa — the TeaCache signal.

        Mirrors WanTransformerBlock.forward for temb.ndim == 3:
            shift_msa, scale_msa, ... = (scale_shift_table + temb.float()).chunk(6, 1)
            norm_h = norm1(hidden.float()) * (1 + scale_msa) + shift_msa
        """
        block = self.block
        h = self.hidden_states.float()
        temb = self.temb.float()
        sst = block.scale_shift_table  # (1, 6, inner_dim)
        if temb.ndim == 4:
            chunks = (sst.unsqueeze(0) + temb).chunk(6, dim=2)
            shift_msa = chunks[0].squeeze(2)
            scale_msa = chunks[1].squeeze(2)
        else:
            chunks = (sst + temb).chunk(6, dim=1)
            shift_msa = chunks[0]
            scale_msa = chunks[1]
        norm_h = block.norm1(h) * (1 + scale_msa) + shift_msa
        return norm_h.detach()

    def close(self):
        self._handle.remove()


def pearson(x, y) -> float:
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = (xm.norm() * ym.norm()).item()
    if denom == 0.0:
        return float("nan")
    return float((xm @ ym).item() / denom)


def rel_l1(curr: torch.Tensor, prev: torch.Tensor) -> float:
    curr = curr.float()
    prev = prev.float()
    num = (curr - prev).abs().mean().item()
    den = prev.abs().mean().item()
    return num / den if den != 0.0 else float("nan")


def run_gate(num_steps: int, dtype: torch.dtype) -> None:
    tdir = resolve_transformer_dir(WAN_MODEL_ID, WAN_SUBFOLDER)
    model = build_transformer(tdir, dtype)

    text_seq_len = 226  # typical Wan UMT5 prompt length; arbitrary for the gate
    latents, encoder_hidden_states = make_inputs(
        dtype, text_seq_len, model.config.text_dim
    )

    capture = Block0Capture(model.blocks[0])

    # Flow-matching: linear sigma 1 -> 0, Euler update. Wan timestep in [0,1000].
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    log(f"steps={num_steps}  sigma schedule {sigmas[0].item():.3f} -> {sigmas[-1].item():.3f}")

    x = latents.clone()
    mod_inps: list[torch.Tensor] = []
    noise_preds: list[torch.Tensor] = []

    for i in range(num_steps):
        sigma_t = sigmas[i].item()
        sigma_next = sigmas[i + 1].item()
        t_val = sigma_t * 1000.0
        timestep = torch.full((x.shape[0],), t_val, dtype=dtype)

        with torch.no_grad():
            noise_pred = model(x, timestep, encoder_hidden_states)
        if not torch.is_tensor(noise_pred):
            noise_pred = noise_pred[0]
        noise_pred = noise_pred.float()

        mod_inp = capture.modulated_input()
        mod_inps.append(mod_inp)
        noise_preds.append(noise_pred.clone())

        # Flow-matching Euler: x <- x - (sigma_t - sigma_next) * noise_pred
        x = x - (sigma_t - sigma_next) * noise_pred.to(dtype)
        log(
            f"step {i:2d}  t={t_val:7.2f}  sigma={sigma_t:.4f}->{sigma_next:.4f}  "
            f"|noise_pred|={noise_pred.abs().mean().item():.4e}  "
            f"|mod_inp|={mod_inp.abs().mean().item():.4e}"
        )

    capture.close()

    signals: list[float] = []
    deltas: list[float] = []
    for i in range(1, num_steps):
        signals.append(rel_l1(mod_inps[i], mod_inps[i - 1]))
        deltas.append(rel_l1(noise_preds[i], noise_preds[i - 1]))

    r = pearson(signals, deltas)

    log("=" * 64)
    log("RESULTS")
    log(f"  steps                : {num_steps}")
    log(f"  embed source         : FIXED RANDOM (constant across steps)")
    log(f"  consecutive pairs    : {len(signals)}")
    log(f"  signal (relL1 mod)   : " + ", ".join(f"{s:.4f}" for s in signals))
    log(f"  true_delta (relL1 np): " + ", ".join(f"{d:.4f}" for d in deltas))
    if signals:
        log(f"  signal min/max       : {min(signals):.4f} / {max(signals):.4f}")
        log(f"  delta  min/max       : {min(deltas):.4f} / {max(deltas):.4f}")
    log(f"  PEARSON(signal,delta): {r:.4f}")
    log("=" * 64)
    verdict = (
        "viable" if (r == r and r >= 0.8) else
        "weak/uncertain" if (r == r and r >= 0.5) else
        "NOT viable (low correlation)" if r == r else
        "undefined (constant arrays)"
    )
    log(f"  TeaCache verdict     : {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Wan 2.2 TeaCache CPU signal gate")
    ap.add_argument("--steps", type=int, default=16, help="denoise steps (default 16)")
    ap.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="compute dtype (default bf16; fp32 ok with 124GB RAM)",
    )
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    log(f"backend={os.environ.get('NOVA_BACKEND')} dtype={dtype} python={sys.executable}")
    run_gate(args.steps, dtype)


if __name__ == "__main__":
    main()
