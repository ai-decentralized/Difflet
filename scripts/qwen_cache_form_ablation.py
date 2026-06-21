#!/usr/bin/env python3
"""cclog 84 / Option B: CPU eager cache-form ablation for Qwen-Image TeaCache.

Decisively decomposes Qwen's TeaCache cosine deficit into (signal quality) vs
(cache form), which the workflow flagged as the open question gating Path B.

Method: run the diffusers QwenImageTransformer2DModel on CPU, reimplementing its
forward as 3 separable stages (embed / 60-blocks / postprocess). Under an ORACLE
skip pattern that is IDENTICAL for both cache forms (so the only variable is the
cache form), run a full 50-step denoise three ways:

  (i)  reference   — every step full (no cache)
  (ii) difflet-form   — noise_pred extrapolation: on skip, np = prev_np + cached(np_delta)
                     (the HV cache form, skip = don't run the DiT at all)
  (iii) vllm-form  — block-residual + re-postprocess: on full step cache
                     residual_h = h_out - h_in (image stream); on skip recompute
                     h_in from the CURRENT latent, h_out = h_in + residual_h, then
                     np = proj_out(norm_out(h_out, temb_NOW)) with the CURRENT temb.

Report final-latent + min-trajectory cosine of (ii) and (iii) vs (i) at MATCHED
skips. Also harvest corrected rel-L1(block0 mod input) -> rel-L1(noise_pred) pairs
from the reference run (the Option-A calibration data, which was never persisted).

Decision: GO Path B only if vllm-form >> difflet-form at matched skips. STOP Path B
if they are similar and both well below 0.999 -> the 0.59 signal is the limiter.
"""

from __future__ import annotations

import argparse
import json
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402

MODEL_DIR = Path("/home/ubuntu/.cache/huggingface/hub/qwen-image-real/transformer")
BUNDLE_DIR = ROOT / ".difflet-cache" / "qwen_image_dit_inputs" / "m9_calib_50step"
CCLOG = ROOT / "cclogs" / "m9-teacache"
IMG_SHAPES = [[(1, 64, 64)]]  # 1024^2 -> packed 64x64


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1), dim=1).item())


def _traj_cos(la: list[torch.Tensor], lb: list[torch.Tensor]) -> float:
    return min(_cos(a, b) for a, b in zip(la, lb))


def _rel_l1(cur: torch.Tensor, prev: torch.Tensor) -> float:
    return float(((cur - prev).abs().mean() / (prev.abs().mean() + 1e-8)).item())


# --- 3-stage decomposition of the diffusers QwenImage forward (zero_cond_t=False,
#     guidance_embeds=False), faithful to transformer_qwenimage.py:911-989. ---
def _make_stages(model):
    import diffusers.models.transformers.transformer_qwenimage as M

    compute_text_seq_len_from_mask = M.compute_text_seq_len_from_mask

    def embed(hs, ts, ehs, mask):
        h_in = model.img_in(hs)
        ts = ts.to(h_in.dtype)
        e = model.txt_norm(ehs)
        e = model.txt_in(e)
        text_seq_len, _, mask2 = compute_text_seq_len_from_mask(e, mask)
        temb = model.time_text_embed(ts, h_in, None)
        rope = model.pos_embed(IMG_SHAPES, max_txt_seq_len=text_seq_len, device=h_in.device)
        bak: dict = {}
        if mask2 is not None:
            bs, isl = h_in.shape[:2]
            img_mask = torch.ones((bs, isl), dtype=torch.bool, device=h_in.device)
            jam = torch.cat([mask2, img_mask], dim=1)[:, None, None, :]
            bak["attention_mask"] = jam
        return h_in, e, temb, rope, bak

    def blocks(h_in, e, temb, rope, bak):
        h, ee = h_in, e
        for block in model.transformer_blocks:
            ee, h = block(
                hidden_states=h,
                encoder_hidden_states=ee,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=rope,
                joint_attention_kwargs=bak,
                modulate_index=None,
            )
        return h

    def postprocess(h, temb):
        h = model.norm_out(h, temb)
        return model.proj_out(h)

    def block0_mod(h_in, temb):
        b0 = model.transformer_blocks[0]
        img_mod1 = b0.img_mod(temb).chunk(2, dim=-1)[0]
        img_normed = b0.img_norm1(h_in)
        modulated, _ = b0._modulate(img_normed, img_mod1, None)
        return modulated

    return embed, blocks, postprocess, block0_mod


def _scheduler(sched_id: str):
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        sched_id, subfolder="scheduler", local_files_only=True
    )


def _timesteps(sched, num_steps: int, image_seq_len: int):
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps

    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    c = sched.config
    mu = calculate_shift(
        image_seq_len,
        c.get("base_image_seq_len", 256),
        c.get("max_image_seq_len", 4096),
        c.get("base_shift", 0.5),
        c.get("max_shift", 1.15),
    )
    ts, _ = retrieve_timesteps(sched, num_steps, "cpu", sigmas=sigmas, mu=mu)
    return ts


def _oracle_skips(num_steps: int, warmup: int, cooldown: int, stride: int) -> set[int]:
    """Fixed skip set: within [warmup, N-cooldown), skip 1 of every `stride` steps.
    stride=2 -> ~50% of the window skipped. Identical for both cache forms."""
    skips = set()
    lo, hi = warmup, num_steps - cooldown
    for i in range(lo, hi):
        if (i - lo) % stride == (stride - 1):
            skips.add(i)
    return skips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheduler-id", default="Qwen/Qwen-Image")
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--n-bundles", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2, help="skip 1 of every `stride` steps in window")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 8)
    dtype = torch.bfloat16
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    print(f"[abl] loading model (CPU {dtype})...", flush=True)
    t0 = time.perf_counter()
    model = QwenImageTransformer2DModel.from_pretrained(
        MODEL_DIR, torch_dtype=dtype, local_files_only=True
    ).eval()
    print(f"[abl] loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    embed, blocks, postprocess, block0_mod = _make_stages(model)
    sched = _scheduler(args.scheduler_id)

    bundles = sorted(BUNDLE_DIR.glob("calibration_*.safetensors"))[: args.n_bundles]
    skip_set = _oracle_skips(args.num_steps, args.warmup, args.cooldown, args.stride)
    skip_frac = len(skip_set) / args.num_steps
    print(f"[abl] oracle skip set size={len(skip_set)} ({skip_frac:.0%} of {args.num_steps} steps)", flush=True)

    per_bundle = []
    pairs = []  # corrected rel-L1 calibration pairs (from reference runs)

    for bi, bpath in enumerate(bundles):
        tns = load_safetensors_file(str(bpath), device="cpu")
        hs0 = tns["latents_init"].to(dtype)
        ehs = tns["encoder_hidden_states"].to(dtype)
        mask = tns["encoder_hidden_states_mask"].to(torch.bool)
        ts_all = _timesteps(sched, args.num_steps, int(hs0.shape[1]))

        # ---------- (i) reference + harvest pairs ----------
        ref_traj = []
        lat = hs0.clone()
        sched_ref = _scheduler(args.scheduler_id)
        ts_all = _timesteps(sched_ref, args.num_steps, int(hs0.shape[1]))
        prev_mod = None
        prev_np = None
        tb = time.perf_counter()
        with torch.no_grad():
            for si, t in enumerate(ts_all):
                tt = (t.reshape(1).to(dtype) / 1000.0)
                h_in, e, temb, rope, bak = embed(lat, tt, ehs, mask)
                mod = block0_mod(h_in, temb)
                h_out = blocks(h_in, e, temb, rope, bak)
                np_pred = postprocess(h_out, temb)
                if prev_mod is not None and prev_np is not None:
                    pairs.append({
                        "split": "train", "bundle": bi, "step_index": int(si),
                        "rel_l1_mod": _rel_l1(mod, prev_mod),
                        "rel_l1_noise": _rel_l1(np_pred, prev_np),
                    })
                prev_mod = mod.detach(); prev_np = np_pred.detach()
                lat = sched_ref.step(np_pred, t, lat, return_dict=False)[0]
                ref_traj.append(lat.detach().clone())
        print(f"[abl] b{bi} reference done in {time.perf_counter()-tb:.0f}s", flush=True)

        # ---------- (ii) difflet-form: noise_pred extrapolation ----------
        difflet_traj = []
        lat = hs0.clone()
        sched_n = _scheduler(args.scheduler_id)
        _timesteps(sched_n, args.num_steps, int(hs0.shape[1]))
        prev_np = None; resid_np = None
        with torch.no_grad():
            for si, t in enumerate(ts_all):
                tt = (t.reshape(1).to(dtype) / 1000.0)
                if si in skip_set and prev_np is not None and resid_np is not None:
                    np_pred = prev_np + resid_np
                    prev_np = np_pred
                else:
                    h_in, e, temb, rope, bak = embed(lat, tt, ehs, mask)
                    h_out = blocks(h_in, e, temb, rope, bak)
                    np_pred = postprocess(h_out, temb)
                    if prev_np is not None:
                        resid_np = (np_pred - prev_np).detach()
                    prev_np = np_pred.detach()
                lat = sched_n.step(np_pred, t, lat, return_dict=False)[0]
                difflet_traj.append(lat.detach().clone())

        # ---------- (iii) vllm-form: block-residual + current-temb re-postprocess ----------
        vllm_traj = []
        lat = hs0.clone()
        sched_v = _scheduler(args.scheduler_id)
        _timesteps(sched_v, args.num_steps, int(hs0.shape[1]))
        resid_h = None
        with torch.no_grad():
            for si, t in enumerate(ts_all):
                tt = (t.reshape(1).to(dtype) / 1000.0)
                h_in, e, temb, rope, bak = embed(lat, tt, ehs, mask)  # always (cheap)
                if si in skip_set and resid_h is not None:
                    h_out = h_in + resid_h
                else:
                    h_out = blocks(h_in, e, temb, rope, bak)
                    resid_h = (h_out - h_in).detach()
                np_pred = postprocess(h_out, temb)  # CURRENT temb
                lat = sched_v.step(np_pred, t, lat, return_dict=False)[0]
                vllm_traj.append(lat.detach().clone())

        rec = {
            "bundle": bpath.name,
            "difflet_final_cos": _cos(difflet_traj[-1], ref_traj[-1]),
            "difflet_traj_cos": _traj_cos(difflet_traj, ref_traj),
            "vllm_final_cos": _cos(vllm_traj[-1], ref_traj[-1]),
            "vllm_traj_cos": _traj_cos(vllm_traj, ref_traj),
        }
        per_bundle.append(rec)
        print(f"[abl] b{bi} {bpath.name}: difflet final={rec['difflet_final_cos']:.4f} traj={rec['difflet_traj_cos']:.4f} "
              f"| vllm final={rec['vllm_final_cos']:.4f} traj={rec['vllm_traj_cos']:.4f}", flush=True)

    # aggregate (min across bundles, matching the e2e metric)
    agg = {
        "difflet_final_cos_min": min(r["difflet_final_cos"] for r in per_bundle),
        "difflet_traj_cos_min": min(r["difflet_traj_cos"] for r in per_bundle),
        "vllm_final_cos_min": min(r["vllm_final_cos"] for r in per_bundle),
        "vllm_traj_cos_min": min(r["vllm_traj_cos"] for r in per_bundle),
    }
    # signal correlation from harvested pairs
    xs = torch.tensor([p["rel_l1_mod"] for p in pairs], dtype=torch.float64)
    ys = torch.tensor([p["rel_l1_noise"] for p in pairs], dtype=torch.float64)
    pearson = float(torch.corrcoef(torch.stack([xs, ys]))[0, 1].item())

    out = {
        "schema": "difflet-m9-teacache-cacheform-ablation-v1",
        "model": "qwen_image", "shape_label": "1024x1024",
        "num_steps": args.num_steps, "n_bundles": len(per_bundle),
        "oracle_skip_fraction": skip_frac, "skip_stride": args.stride,
        "warmup": args.warmup, "cooldown": args.cooldown,
        "signal_pearson_corrected": pearson, "n_pairs": len(pairs),
        "per_bundle": per_bundle, "aggregate": agg,
        "hardware_measured": False, "device": "cpu",
    }
    (CCLOG / "qwen_cacheform_ablation.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    pairs_doc = {
        "schema": "difflet-m9-teacache-pairs-v1", "model": "qwen_image",
        "shape_label": "1024x1024", "num_steps": args.num_steps,
        "mod_input_source": "block0_modulated_input_rel_l1", "signal": "relative_l1",
        "hardware_measured": False, "samples": pairs,
    }
    (CCLOG / "pairs_qwen_relL1_corrected.json").write_text(json.dumps(pairs_doc, indent=2, sort_keys=True) + "\n")

    print("\n[abl] ===== RESULT (matched %d%% skip) =====" % round(skip_frac * 100), flush=True)
    print(f"[abl] difflet-form  : final={agg['difflet_final_cos_min']:.4f}  traj={agg['difflet_traj_cos_min']:.4f}", flush=True)
    print(f"[abl] vllm-form  : final={agg['vllm_final_cos_min']:.4f}  traj={agg['vllm_traj_cos_min']:.4f}", flush=True)
    delta = agg["vllm_final_cos_min"] - agg["difflet_final_cos_min"]
    print(f"[abl] vllm - difflet (final cos) = {delta:+.4f}", flush=True)
    print(f"[abl] corrected signal Pearson(rel_l1_mod, rel_l1_noise) = {pearson:.4f} (n={len(pairs)})", flush=True)
    verdict = ("GO Path B (cache form recovers a lot)" if delta > 0.03
               else "STOP Path B (cache form ~neutral -> signal is the limiter)")
    print(f"[abl] VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
