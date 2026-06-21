#!/usr/bin/env python3
"""CPU TeaCache signal gate for the LTX-2 audiovisual video DiT.

Question: does the block-0 *video self-attention AdaLN modulated input* (the
TeaCache "signal" -- a timestep-only quantity, cheap to compute) correlate
(Pearson) with the model's per-step output change (the "true delta", expensive)?
If yes, adaptive TeaCache (skip the DiT when the signal barely moves) is viable
for LTX-2.

The script:
  1. Loads ``LTX2VideoTransformer3DModel`` from the real Lightricks/LTX-2
     ``transformer`` subfolder on CPU (bf16, .eval()).
  2. Loads the cached dual-stream input bundle
     ``.difflet-cache/ltx_2_dit_inputs/full_512x768x121_4step.safetensors``.
  3. Builds a flow-matching denoise schedule (sigmas ~1.0 -> ~0.0) and runs an
     Euler loop on CPU, mirroring the host-frontend conventions in
     ``difflet/backends/trainium/ltx_2/segmented.py`` and
     ``difflet/models/ltx_2/pipeline.py`` (per-token timestep already scaled by
     ``timestep_scale_multiplier``; ``timestep`` and ``sigma`` are the same
     scheduler value; ``video_coords`` / ``audio_coords`` supplied so RoPE does
     not need num_frames/height/width).
  4. At each step it captures the block-0 modulated self-attn input via a
     ``forward_pre_hook`` on ``transformer_blocks[0]`` and recomputes
     ``norm1(hidden) * (1 + scale_msa) + shift_msa`` (transformer_ltx2.py
     :628-634), and records the video ``noise_pred``.
  5. Computes per consecutive-step relative-L1 of the signal and of the
     noise_pred, then their Pearson correlation.

Run with the Neuron venv python (the script re-execs into it automatically):

    /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \\
        scripts/ltx2_teacache_cpu_gate.py --steps 16

Useful flags:
    --steps N        number of denoise steps (default 16)
    --max-steps M    stop after M model forwards (smoke; default: all)
    --float32        load in float32 instead of bf16
"""

from __future__ import annotations

import argparse
import json
import math
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

# Make ``difflet`` importable (for the RoPE patch + scheduler helpers) and ensure the
# Neuron venv bin dir is on PATH so the diffusers->torch_xla lazy init can find
# ``libneuronpjrt-path`` (when xla is not disabled).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if (NEURON_VENV / "bin").exists():
    os.environ["PATH"] = f"{NEURON_VENV / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"


def _disable_torch_xla_lazy_import() -> None:
    """Stop diffusers' attention_processor from triggering torch_xla/Neuron init.

    Importing ``LTX2VideoTransformer3DModel`` pulls in ``attention_processor``,
    whose top-level ``from torch_xla.experimental.custom_kernel import
    flash_attention`` runs ``torch_xla`` Neuron init (needs ``libneuronpjrt-path``
    on PATH). We only need a CPU forward, so flip the diffusers availability flag
    off first -- mirrors ``difflet.models.ltx_2.pipeline.disable_ltx_2_xla_lazy_import``.
    """
    try:
        import diffusers.utils.import_utils as import_utils

        import_utils._torch_xla_available = False
    except Exception:  # pragma: no cover - best effort
        pass


_disable_torch_xla_lazy_import()

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

LOG = "[ltx2-gate]"
BUNDLE = ROOT / ".difflet-cache" / "ltx_2_dit_inputs" / "full_512x768x121_4step.safetensors"
META = BUNDLE.with_suffix(BUNDLE.suffix + ".meta.json")


def log(*parts: object) -> None:
    print(LOG, *parts, flush=True)


def resolve_transformer_dir() -> str:
    """Find the LTX-2 ``transformer`` subfolder, downloading if needed."""
    # 1. The model_id recorded in the bundle meta (if its transformer/ exists).
    candidates: list[Path] = []
    if META.exists():
        meta = json.loads(META.read_text())
        model_id = meta.get("model_id")
        if model_id:
            candidates.append(Path(model_id) / "transformer")
            candidates.append(Path(model_id))
    for cand in candidates:
        if (cand / "config.json").exists() and any(cand.glob("*.safetensors")):
            log(f"using transformer dir: {cand}")
            return str(cand)

    # 2. Resolve / download from the hub (public repo).
    log("transformer not found locally; resolving Lightricks/LTX-2 transformer via hub")
    from huggingface_hub import snapshot_download

    snap = snapshot_download(
        "Lightricks/LTX-2",
        allow_patterns=["transformer/*", "scheduler/*"],
    )
    cand = Path(snap) / "transformer"
    if not ((cand / "config.json").exists() and any(cand.glob("*.safetensors"))):
        raise FileNotFoundError(f"transformer weights missing under {cand}")
    log(f"using transformer dir: {cand}")
    return str(cand)


def patch_rope() -> None:
    """Mirror segmented.py's RoPE override so the CPU forward matches runtime."""
    try:
        from difflet.backends.trainium.ltx_2.segmented import _patch_ltx2_rope

        _patch_ltx2_rope()
        log("applied difflet split-rotary RoPE patch")
    except Exception as exc:  # pragma: no cover - patch is best-effort
        log(f"WARNING: could not apply difflet RoPE patch ({exc!r}); using diffusers default")


def build_sigmas(num_steps: int, transformer_dir: str) -> torch.Tensor:
    """Flow-matching sigma schedule, preferring the LTX-2 scheduler if present.

    Returns sigmas of length ``num_steps + 1`` (sigmas[-1] == 0.0) so each step i
    denoises from sigmas[i] to sigmas[i + 1] (Euler flow-matching).
    """
    import numpy as np

    scheduler_dir = Path(transformer_dir).parent / "scheduler"
    if num_steps >= 2 and (scheduler_dir / "scheduler_config.json").exists():
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler

            from difflet.models.ltx_2.pipeline import _retrieve_timesteps, ltx_2_scheduler_mu

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_dir))
            lin = np.linspace(1.0, 1.0 / num_steps, num_steps)
            _retrieve_timesteps(
                scheduler,
                num_steps,
                "cpu",
                sigmas=lin,
                mu=ltx_2_scheduler_mu(getattr(scheduler, "config", {})),
            )
            sig = torch.as_tensor(scheduler.sigmas, dtype=torch.float32)
            log(f"using LTX-2 FlowMatchEulerDiscreteScheduler sigmas ({sig.numel()} entries)")
            return sig
        except Exception as exc:
            log(f"scheduler load failed ({exc!r}); falling back to linear sigmas")

    sig = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float32)
    log(f"using linear sigma schedule ({sig.numel()} entries)")
    return sig


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    ta = ta - ta.mean()
    tb = tb - tb.mean()
    denom = ta.norm() * tb.norm()
    if denom == 0:
        return float("nan")
    return float((ta @ tb) / denom)


def rel_l1(cur: torch.Tensor, prev: torch.Tensor) -> float:
    cur = cur.float()
    prev = prev.float()
    denom = prev.abs().mean()
    if denom == 0:
        return float("nan")
    return float((cur - prev).abs().mean() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="stop after this many model forwards (smoke); default: all steps",
    )
    parser.add_argument("--float32", action="store_true", help="load in fp32 instead of bf16")
    args = parser.parse_args()

    dtype = torch.float32 if args.float32 else torch.bfloat16
    torch.manual_seed(0)

    transformer_dir = resolve_transformer_dir()
    patch_rope()

    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    log(f"loading transformer (dtype={dtype}) ...")
    model = LTX2VideoTransformer3DModel.from_pretrained(transformer_dir, torch_dtype=dtype)
    model = model.to(dtype=dtype).eval()
    ts_mult = float(getattr(model.config, "timestep_scale_multiplier", 1000))
    log(f"loaded transformer; timestep_scale_multiplier={ts_mult}")

    # ---- load bundle -----------------------------------------------------
    log(f"loading bundle: {BUNDLE}")
    b = load_file(str(BUNDLE))
    latents = b["latents_init"].to(dtype)
    audio_latents = b["audio_latents_init"].to(dtype)
    encoder_hidden_states = b["encoder_hidden_states"].to(dtype)
    audio_encoder_hidden_states = b["audio_encoder_hidden_states"].to(dtype)
    encoder_attention_mask = b["encoder_attention_mask"]
    audio_encoder_attention_mask = b["audio_encoder_attention_mask"]
    video_coords = b["video_coords"].to(torch.float32)
    audio_coords = b["audio_coords"].to(torch.float32)
    batch_size = latents.shape[0]
    num_video_tokens = latents.shape[1]
    log(
        f"latents={tuple(latents.shape)} audio_latents={tuple(audio_latents.shape)} "
        f"enc={tuple(encoder_hidden_states.shape)}"
    )

    # ---- schedule --------------------------------------------------------
    sigmas = build_sigmas(args.steps, transformer_dir)
    num_steps = sigmas.numel() - 1
    if args.max_steps is not None:
        num_steps = min(num_steps, int(args.max_steps))
    log(f"running {num_steps} denoise step(s)")

    # ---- block-0 modulated-input capture ---------------------------------
    block0 = model.transformer_blocks[0]
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inp_args, inp_kwargs):
        # block.forward(hidden_states, audio_hidden_states, encoder_hidden_states,
        #               audio_encoder_hidden_states, temb, ...)
        hs = inp_kwargs.get("hidden_states")
        if hs is None:
            hs = inp_args[0]
        temb = inp_kwargs.get("temb")
        if temb is None:
            temb = inp_args[4]
        captured["hidden_states"] = hs.detach()
        captured["temb"] = temb.detach()

    handle = block0.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def modulated_input() -> torch.Tensor:
        # Replicate transformer_ltx2.py:628-634 for the video self-attn branch.
        hs = captured["hidden_states"]
        temb = captured["temb"]
        ada = block0.get_mod_params(block0.scale_shift_table, temb, hs.shape[0])
        shift_msa, scale_msa = ada[0], ada[1]
        norm = block0.norm1(hs)
        return norm * (1 + scale_msa) + shift_msa

    signals: list[torch.Tensor] = []
    noise_preds: list[torch.Tensor] = []

    with torch.no_grad():
        for step in range(num_steps):
            sigma_t = float(sigmas[step])
            sigma_next = float(sigmas[step + 1])
            # Per-token timestep, pre-scaled by timestep_scale_multiplier, exactly
            # like the segmented frontend (which receives scheduler timestep ==
            # sigma * timestep_scale_multiplier and passes it straight through).
            # The host frontend / pipeline pass a per-BATCH scalar timestep (shape
            # (batch,)) for both video and audio -- ``time_embed`` produces one temb
            # row per batch element which broadcasts across all tokens (so the same
            # value is valid for the 6144 video tokens and the 126 audio tokens).
            # The value is the scheduler timestep == sigma * timestep_scale_multiplier.
            ts_value = sigma_t * ts_mult
            timestep = torch.full((batch_size,), ts_value, dtype=dtype)
            sigma = torch.full((batch_size,), ts_value, dtype=dtype)

            out = model(
                hidden_states=latents,
                audio_hidden_states=audio_latents,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep=timestep,
                audio_timestep=timestep,
                sigma=sigma,
                audio_sigma=sigma,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
                video_coords=video_coords,
                audio_coords=audio_coords,
                fps=24.0,
                return_dict=False,
            )
            noise_pred_video = out[0]
            noise_pred_audio = out[1]

            mod_inp = modulated_input()
            signals.append(mod_inp.float().cpu())
            noise_preds.append(noise_pred_video.float().cpu())

            # Flow-matching Euler update: x <- x - (sigma_t - sigma_next) * v
            dt = sigma_t - sigma_next
            latents = latents - dt * noise_pred_video.to(dtype)
            audio_latents = audio_latents - dt * noise_pred_audio.to(dtype)
            log(
                f"step {step}: sigma {sigma_t:.4f} -> {sigma_next:.4f} "
                f"(timestep={ts_value:.2f}) done"
            )

    handle.remove()

    # ---- correlate -------------------------------------------------------
    signal_arr: list[float] = []
    delta_arr: list[float] = []
    for i in range(1, len(noise_preds)):
        signal_arr.append(rel_l1(signals[i], signals[i - 1]))
        delta_arr.append(rel_l1(noise_preds[i], noise_preds[i - 1]))

    log("=" * 60)
    log(f"completed {len(noise_preds)} forward(s); {len(signal_arr)} consecutive pair(s)")
    log(f"signal (block-0 modulated-input rel-L1): {[round(x, 6) for x in signal_arr]}")
    log(f"true_delta (noise_pred rel-L1):          {[round(x, 6) for x in delta_arr]}")
    if signal_arr:
        finite = [x for x in signal_arr if math.isfinite(x)]
        if finite:
            log(f"signal min={min(finite):.6f} max={max(finite):.6f}")
    r = pearson(signal_arr, delta_arr)
    log(f"PEARSON(signal, true_delta) = {r:.6f}  over {len(signal_arr)} pair(s)")
    if len(signal_arr) < 2:
        log("NOTE: <2 pairs -> Pearson is degenerate; run more steps for a real verdict.")
    log("=" * 60)


if __name__ == "__main__":
    main()
