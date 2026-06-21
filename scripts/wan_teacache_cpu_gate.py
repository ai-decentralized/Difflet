#!/usr/bin/env python3
"""CPU TeaCache signal gate for Wan 2.2 (high-noise ``transformer`` stage).

Question this script answers
----------------------------
TeaCache skips diffusion steps when a cheap *signal* derived from the block-0
self-attention AdaLN modulated input is small. That only works if the signal
**correlates** with the model's real per-step output change. This gate runs a
short CPU flow-matching denoise loop on Difflet's ``WanTransformer3DModel`` and
measures the Pearson correlation between

    signal[i]     = mean|mod_inp[i] - mod_inp[i-1]| / mean|mod_inp[i-1]|
    true_delta[i] = mean|noise_pred[i] - noise_pred[i-1]| / mean|noise_pred[i-1]|

A high Pearson (say >= ~0.8) means adaptive TeaCache is viable for Wan 2.2.

Notes / verified facts
----------------------
* Modeling: ``difflet/models/wan/modeling_wan.py`` :: ``WanTransformer3DModel``.
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
    DIFFLET_BACKEND=cpu PYTHONPATH=/home/ubuntu/difflet \
    /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \
        scripts/wan_teacache_cpu_gate.py

(The script self-execs into the neuron venv python and sets PATH/DIFFLET_BACKEND
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
LATENT_CACHE = ROOT / ".difflet-cache" / "wan_smoke_latents.pt"
DEFAULT_LATENT_SHAPE = (1, 16, 3, 60, 104)


def ensure_runtime_python() -> None:
    """Re-exec into the neuron venv python with PATH + DIFFLET_BACKEND set.

    torch_xla's init shells out to ``libneuronpjrt-path`` (only on the venv
    PATH), and ``neuronx_distributed`` is imported transitively even for the
    CPU op backend, so PATH must include the venv bin. ``DIFFLET_BACKEND=cpu``
    selects the numerical CPU reference ops.
    """
    need_python = Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists()
    venv_bin = str(NEURON_VENV / "bin")
    need_path = venv_bin not in os.environ.get("PATH", "").split(os.pathsep)
    need_backend = os.environ.get("DIFFLET_BACKEND") != "cpu"
    try:
        import torch  # noqa: F401

        torch_ok = True
    except ModuleNotFoundError:
        torch_ok = False

    if torch_ok and not need_path and not need_backend:
        return
    if not NEURON_PYTHON.exists():
        # Best effort: just set the env knobs in-process.
        os.environ["DIFFLET_BACKEND"] = "cpu"
        return

    env = os.environ.copy()
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["DIFFLET_BACKEND"] = "cpu"
    target = NEURON_PYTHON if need_python else Path(sys.executable)
    os.execve(str(target), [str(target), *sys.argv], env)


ensure_runtime_python()

import importlib  # noqa: E402

import torch  # noqa: E402

# Import the modeling module DIRECTLY (not via ``difflet.models.wan`` package
# __init__, which imports application.py -> neuronx_distributed eagerly and is
# unnecessary here).
_wan = importlib.import_module("difflet.models.wan.modeling_wan")
WanTransformer3DModel = _wan.WanTransformer3DModel
WanTransformerConfig = _wan.WanTransformerConfig

_ckpt = importlib.import_module("difflet.models.wan.checkpoint.backbone")
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


def make_inputs(dtype: torch.dtype, text_seq_len: int, text_dim: int, *, prompt=None, transformer_dir=None):
    """Latents (cache/random) + encoder_hidden_states.

    If ``prompt`` is given, encode it with the real UMT5 text encoder (cclog 88: the
    fixed-random embeds were a stand-in; this is the real-prompt re-calibration path).
    Otherwise use a fixed-random tensor (the timestep-only signal only needs it constant).
    """
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
    if prompt:
        import gc as _gc
        from pathlib import Path as _P

        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, UMT5EncoderModel

        # resolve_transformer_dir only fetched transformer/*; the real-embed path also
        # needs tokenizer/ + text_encoder/ (the ~11 GB UMT5). Fetch them now and use the
        # canonical snapshot dir (do NOT rely on transformer_dir.parent — it may be None).
        snap = _P(snapshot_download(WAN_MODEL_ID, allow_patterns=["tokenizer/*", "text_encoder/*"]))
        log(f"fetched tokenizer/ + text_encoder/ under {snap} (real-embed path)")
        tok = AutoTokenizer.from_pretrained(str(snap / "tokenizer"))
        te = UMT5EncoderModel.from_pretrained(str(snap / "text_encoder"), torch_dtype=dtype).eval()
        enc = tok([prompt] * b, return_tensors="pt", padding="max_length",
                  max_length=text_seq_len, truncation=True)
        with torch.no_grad():
            encoder_hidden_states = te(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask
            ).last_hidden_state.to(dtype)
        del te, tok
        _gc.collect()
        log(f"embed source: REAL UMT5 encoder_hidden_states {tuple(encoder_hidden_states.shape)} "
            f"prompt={prompt!r}")
    else:
        encoder_hidden_states = torch.randn(b, text_seq_len, text_dim, generator=g).to(dtype)
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


def run_gate(num_steps: int, dtype: torch.dtype, prompt=None, write_calib=False) -> None:
    tdir = resolve_transformer_dir(WAN_MODEL_ID, WAN_SUBFOLDER)
    model = build_transformer(tdir, dtype)

    text_seq_len = 226  # typical Wan UMT5 prompt length; arbitrary for the gate
    latents, encoder_hidden_states = make_inputs(
        dtype, text_seq_len, model.config.text_dim, prompt=prompt, transformer_dir=tdir
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
    log(f"  embed source         : "
        + (f"REAL UMT5 (prompt={prompt!r})" if prompt else "FIXED RANDOM (constant across steps)"))
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

    if write_calib and not (r == r and r >= 0.8):
        log(f"  SKIP write-calib: Pearson {r:.4f} < 0.8 (not viable) — refusing to clobber "
            "the production calibration with a non-viable fit")
    elif write_calib:
        import json
        from pathlib import Path as _P

        import numpy as np

        sig = np.array(signals)
        dlt = np.array(deltas)
        deg = 4
        scale = float(sig.max())
        desc = np.polyfit(sig / scale, dlt, deg)  # highest->lowest, in (x/scale)
        asc = [float(desc[deg - k]) / (scale ** k) for k in range(deg + 1)]  # ascending, raw x
        out = {
            "schema": "difflet-m9-teacache-calibration-v1",
            "model": "wan",
            "shape_label": "832x480x13",
            "poly_coef": asc,
            "threshold": 0.20,
            "warmup_steps": 2,
            "cooldown_steps": 2,
            "num_steps": num_steps,
            "skip_run_length": 1,
            "accumulate": True,
            "cadence": 0,
            "notes": (
                f"cclog 88 re-fit from REAL UMT5 embeds (prompt={prompt!r}, {num_steps}-step gate, "
                f"Pearson {r:.4f}). Supersedes the fixed-random-embed calibration."
            ),
        }
        calib_path = _P(__file__).resolve().parents[1] / "cclogs" / "m9-teacache" / "teacache_calib_wan.json"
        backup = calib_path.with_suffix(".randomembed.json")
        if calib_path.exists() and not backup.exists():
            backup.write_text(calib_path.read_text(encoding="utf-8"), encoding="utf-8")
            log(f"  backed up old calib -> {backup.name}")
        calib_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        log(f"  WROTE real-embed calib -> {calib_path} (coef={[round(c, 4) for c in asc]})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Wan 2.2 TeaCache CPU signal gate")
    ap.add_argument("--steps", type=int, default=50,
                    help="denoise steps (default 50 = production schedule; <50 under-samples the signal)")
    ap.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="compute dtype (default bf16; fp32 ok with 124GB RAM)",
    )
    ap.add_argument("--prompt", default=None,
                    help="if set, encode with the real UMT5 text encoder instead of random embeds")
    ap.add_argument("--write-calib", action="store_true",
                    help="re-fit + write teacache_calib_wan.json from this gate's trajectory")
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    log(f"backend={os.environ.get('DIFFLET_BACKEND')} dtype={dtype} python={sys.executable}")
    run_gate(args.steps, dtype, prompt=args.prompt, write_calib=args.write_calib)


if __name__ == "__main__":
    main()
