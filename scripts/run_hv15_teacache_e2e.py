#!/usr/bin/env python3
"""cclog 86: HunyuanVideo-1.5 TeaCache — compile + signal gate + e2e A/B (monolithic).

Single process: build the HV-1.5 monolithic app with teacache_fused=True (mounts the
fused probe), compile the DiT + probe NEFFs, load, then drive a 50-step denoise from the
cached 320x512x61 DiT-input bundle (text/image embeds are step-independent → reused; a
fresh flow-match schedule is computed). Reports:
  - SIGNAL GATE: Pearson(rel_l1 block0 mod, rel_l1 noise_pred). HV-1.5's block-0 modulation
    is timestep-only (cclog 86) so this is EXPECTED weak (Qwen-like); we measure it.
  - e2e A/B: baseline vs fixed-cadence (probe-free) vs adaptive (poly+accumulator) — final
    latent cosine + wall-clock speedup. Fixed-cadence is the expected deliverable; HV-1.5 is
    video (smooth trajectory) so its cosine may beat Qwen's single-image ~0.90.

DiT input is 65-ch = [32-ch latent | 33-ch zero cond (t2v)]; out is 32-ch noise on the latent.
"""

from __future__ import annotations

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

MODEL_DIR = "/home/ubuntu/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_t2v/snapshots/f4dbc4a1efa4ac8ea56680cdf79d9f455105e814"
BUNDLE = ROOT / ".nova-cache" / "hunyuan15_dit_inputs" / "real_320x512x61_4step.safetensors"
COMPILED = ROOT / ".nova-cache" / "hv15_teacache_keymask" / "compiled"
CCLOG = ROOT / "cclogs" / "m9-teacache"
NUM_STEPS = 50
HEIGHT, WIDTH, NUM_FRAMES = 320, 512, 61
TARGET_SKIP = 0.4
WARMUP, COOLDOWN = 5, 5


def _cos(a, b):
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1), dim=1).item())


def _traj_cos(la, lb):
    return min(_cos(a, b) for a, b in zip(la, lb))


def _pearson(xs, ys):
    a = torch.tensor(xs, dtype=torch.float64)
    b = torch.tensor(ys, dtype=torch.float64)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())


def main() -> int:
    # cclog 86: key-only attention mask (avoids the symmetric-mask softmax 0/0 NaN
    # on the Neuron kernel). Must be set before the trace module is built/compiled.
    os.environ["NOVA_HUNYUAN15_KEY_MASK_ATTENTION"] = "1"
    # cclog 86: run the token refiner on host (the Neuron flash kernel NaNs at <128 valid
    # keys; the refiner self-attends over the 13/1000-valid mllm stream). The NEFF then
    # receives the already-refined (inner_dim) embeds.
    os.environ["NOVA_HUNYUAN15_HOST_REFINER"] = "1"

    from nova.models.hunyuan_video.application import (
        HunyuanVideo15DiTInputBundle,
        NeuronHunyuanVideoApplication,
    )
    from nova.pipeline.parallel_config import NovaParallelConfig
    from nova.pipeline.teacache import TeaCacheCalibration, TeaCacheController

    tns = load_safetensors_file(str(BUNDLE), device="cpu")
    dtype = torch.bfloat16
    # cclog 86: the monolithic-DiT NaN was the attention softmax dividing by zero
    # on fully-masked query rows (default symmetric mask). Fixed by the key-only
    # mask processor (NOVA_HUNYUAN15_KEY_MASK_ATTENTION below), so we now pass the
    # REAL masks (padding keys correctly ignored; no all-masked query row).
    mask2 = tns["encoder_attention_mask_2"].to(torch.int64)
    embeds = {
        "encoder_hidden_states": tns["encoder_hidden_states"].to(dtype),
        "encoder_attention_mask": tns["encoder_attention_mask"].to(torch.int64),
        "encoder_hidden_states_2": tns["encoder_hidden_states_2"].to(dtype),
        "encoder_attention_mask_2": mask2,
        "image_embeds": tns["image_embeds"].to(dtype),
    }
    latents_init = tns["latents_init"].to(dtype)  # (1,32,16,20,32)
    cond = tns["hidden_states"][:, 32:].to(dtype)  # (1,33,...) all-zero t2v pad
    timestep_r = tns["timestep_r"].to(dtype)

    print("[hv15-tc] building app (1.5 monolithic, teacache_fused)...", flush=True)
    app = NeuronHunyuanVideoApplication(
        model_path=MODEL_DIR,
        parallel=NovaParallelConfig(tp_degree=4, cp_enabled=False),
        dtype=dtype,
        shape={"height": HEIGHT, "width": WIDTH, "num_frames": NUM_FRAMES},
        model_version="1.5",
        transformer_runtime="monolithic",
        text_seq_len=int(embeds["encoder_hidden_states"].shape[1]),
        enable_vae_decoder=False,
        teacache_fused=True,
    )
    print(f"[hv15-tc] components: {[c.name for c in app.components()]}", flush=True)
    print("[hv15-tc] compiling DiT + probe (first time, long)...", flush=True)
    t = time.time()
    app.compile(str(COMPILED))
    print(f"[hv15-tc] compiled in {time.time()-t:.0f}s", flush=True)
    app.load(str(COMPILED), skip_warmup=True)
    print("[hv15-tc] loaded", flush=True)

    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")

    def _timesteps():
        sigmas = np.linspace(1.0, 0.0, NUM_STEPS + 1)[:-1]
        sched.set_timesteps(sigmas=sigmas, device="cpu")
        return sched.timesteps

    def _bundle(latent, t):
        bs = latent.shape[0]
        hs = torch.cat([latent, cond], dim=1).to(dtype)
        tb = t.reshape(1).expand(bs).to(dtype)
        return HunyuanVideo15DiTInputBundle(
            hidden_states=hs,
            timestep=tb,
            encoder_hidden_states=embeds["encoder_hidden_states"],
            encoder_attention_mask=embeds["encoder_attention_mask"],
            timestep_r=timestep_r.reshape(1).expand(bs).to(dtype),
            encoder_hidden_states_2=embeds["encoder_hidden_states_2"],
            encoder_attention_mask_2=embeds["encoder_attention_mask_2"],
            image_embeds=embeds["image_embeds"],
        )

    def _first(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def _run(controller, record=False):
        sched_ts = _timesteps()
        latent = latents_init.clone()
        traj, pairs = [], []
        prev_np = None
        if controller is not None:
            controller.reset()
        t0 = time.time()
        for i, ts in enumerate(sched_ts):
            b = _bundle(latent, ts)
            delta = None
            want_probe = record or (controller is not None and controller.needs_signal())
            if want_probe:
                delta = float(
                    app.teacache_probe.teacache_delta(b.hidden_states, b.timestep, b.timestep_r)
                    .detach().cpu().item()
                )
            if controller is not None and controller.should_skip(i, None, diff_norm=delta):
                noise = controller.skip_noise_pred(None)
            else:
                noise = _first(app.forward_dit(b)).detach()
                if controller is not None:
                    controller.record_full_step(noise, None)
            if record and prev_np is not None and delta is not None:
                rel = float((noise.float() - prev_np).abs().mean() / (prev_np.abs().mean() + 1e-8))
                pairs.append((delta, rel))
            if record:
                prev_np = noise.detach().float()
            latent = _first(sched.step(noise.to(latent.dtype), ts, latent, return_dict=False))
            traj.append(latent.detach().cpu())
        return traj, pairs, time.time() - t0

    # ---- signal gate (record-only baseline) ----
    base_traj, pairs, base_t = _run(None, record=True)
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    pearson = _pearson(xs, ys)
    print(f"[hv15-tc] SIGNAL n={len(pairs)} Pearson={pearson:.4f} "
          f"| rel_l1_mod mean={np.mean(xs):.4f} range[{min(xs):.4f},{max(xs):.4f}]", flush=True)
    print(f"[hv15-tc] baseline {base_t:.2f}s", flush=True)

    shape_label = f"{HEIGHT}x{WIDTH}x{NUM_FRAMES}"
    results = {}

    # ---- fixed cadence (probe-free) ----
    cad = TeaCacheCalibration(
        model="hunyuan_video", shape_label=shape_label, num_steps=NUM_STEPS,
        poly_coef=(0.0,), threshold=0.0, warmup_steps=WARMUP, cooldown_steps=COOLDOWN, cadence=2,
    )
    ctrl = TeaCacheController(cad)
    fc_traj, _, fc_t = _run(ctrl)
    st = ctrl.stats()
    results["fixed_cadence"] = {
        "speedup": base_t / fc_t, "final_cosine": _cos(fc_traj[-1], base_traj[-1]),
        "traj_cosine": _traj_cos(fc_traj, base_traj),
        "skipped": int(st["skipped_steps"]), "full": int(st["full_steps"]),
    }
    print(f"[hv15-tc] fixed-cadence: {results['fixed_cadence']}", flush=True)

    # ---- adaptive (poly + accumulator), only meaningful if signal is OK ----
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    finite = np.isfinite(xs_arr) & np.isfinite(ys_arr)
    if finite.sum() < 8:
        print(f"[hv15-tc] adaptive SKIPPED: only {int(finite.sum())} finite pairs "
              f"(signal/DiT produced non-finite values)", flush=True)
        results["adaptive"] = {"skipped_reason": "non-finite signal", "n_finite": int(finite.sum())}
        out = {
            "schema": "nova-m9-teacache-hv15-e2e-v1",
            "model": "hunyuan_video_1.5", "shape_label": shape_label, "num_steps": NUM_STEPS,
            "signal_pearson": pearson, "n_pairs": len(pairs),
            "results": results, "hardware_measured": True,
        }
        (CCLOG / "hv15_teacache_e2e.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        _fc = results["fixed_cadence"]
        print(f"[hv15-tc] fixed-cadence: {_fc['speedup']:.3f}x @ cos {_fc['final_cosine']:.4f}", flush=True)
        return 0
    xs, ys = list(xs_arr[finite]), list(ys_arr[finite])
    coef = np.polyfit(np.array(xs), np.array(ys), 4)
    poly = np.poly1d(coef)
    target_skips = max(int(round(TARGET_SKIP * NUM_STEPS)), 1)

    def _sim(thr):
        accum, sk = 0.0, 0
        for j, x in enumerate(xs[: NUM_STEPS - 1]):
            if j < WARMUP or j >= (NUM_STEPS - 1) - COOLDOWN:
                accum = 0.0
                continue
            accum += abs(float(poly(x)))
            if accum < thr:
                sk += 1
            else:
                accum = 0.0
        return sk

    best_thr, best_err = 0.1, 1e9
    for thr in np.linspace(0.01, 3.0, 200):
        e = abs(_sim(float(thr)) - target_skips)
        if e < best_err:
            best_err, best_thr = e, float(thr)
    ad = TeaCacheCalibration(
        model="hunyuan_video", shape_label=shape_label, num_steps=NUM_STEPS,
        poly_coef=tuple(float(c) for c in coef[::-1]), threshold=best_thr,
        warmup_steps=WARMUP, cooldown_steps=COOLDOWN, accumulate=True,
        fit_r2=float(pearson ** 2), n_samples=len(pairs),
    )
    ctrl = TeaCacheController(ad)
    ad_traj, _, ad_t = _run(ctrl)
    st = ctrl.stats()
    results["adaptive"] = {
        "speedup": base_t / ad_t, "final_cosine": _cos(ad_traj[-1], base_traj[-1]),
        "traj_cosine": _traj_cos(ad_traj, base_traj), "threshold": best_thr,
        "skipped": int(st["skipped_steps"]), "full": int(st["full_steps"]),
    }
    print(f"[hv15-tc] adaptive: {results['adaptive']}", flush=True)

    out = {
        "schema": "nova-m9-teacache-hv15-e2e-v1",
        "model": "hunyuan_video_1.5", "shape_label": shape_label, "num_steps": NUM_STEPS,
        "signal_pearson": pearson, "n_pairs": len(pairs),
        "results": results, "hardware_measured": True,
    }
    (CCLOG / "hv15_teacache_e2e.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("\n[hv15-tc] ===== RESULT =====", flush=True)
    print(f"[hv15-tc] signal Pearson={pearson:.4f}", flush=True)
    fc = results["fixed_cadence"]; adr = results["adaptive"]
    print(f"[hv15-tc] fixed-cadence: {fc['speedup']:.3f}x @ cos {fc['final_cosine']:.4f} "
          f"(skip {fc['skipped']}/{fc['skipped']+fc['full']})", flush=True)
    print(f"[hv15-tc] adaptive:      {adr['speedup']:.3f}x @ cos {adr['final_cosine']:.4f} "
          f"(skip {adr['skipped']}/{adr['skipped']+adr['full']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
