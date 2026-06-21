"""On-device (Trainium) end-to-end TeaCache validation for LTX-2.

Bypasses text encoding: drives ``LTX2Orchestrator._denoise`` directly with the
cached precomputed-embeds bundle, comparing baseline vs teacache for denoise
wall-clock speedup + output cosine.

Usage (detached, see cclog instructions):
    setsid bash -c 'PYTHONPATH=/home/ubuntu/difflet \
      PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
      DIFFLET_BACKEND=trainium NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
      TORCH_DISABLE_ADDR2LINE=1 NEURON_RT_LOG_LEVEL=ERROR \
      python scripts/run_ltx2_teacache_e2e.py --num-steps 50 > log 2>&1' < /dev/null & disown
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

P = "[ltx2-e2e]"

MODEL_DIR = (
    "/home/ubuntu/.cache/huggingface/hub/models--Lightricks--LTX-2/"
    "snapshots/47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
)
COMPILE_CACHE = "/home/ubuntu/difflet/.difflet-cache/ltx_2_transformer_full"
BUNDLE = "/home/ubuntu/difflet/.difflet-cache/ltx_2_dit_inputs/full_512x768x121_4step.safetensors"
CALIB = "/home/ubuntu/difflet/cclogs/m9-teacache/teacache_calib_ltx_2.json"

HEIGHT, WIDTH, NUM_FRAMES = 512, 768, 121
TEXT_SEQ_LEN = 1024
TP_DEGREE = 4


def _preflight() -> int:
    """Verify the compiled single-mode transformer artifact actually exists.

    The content-addressed layout is <cache>/ltx_2/<hash>/transformer/{model.pt,
    neuron_config.json}. If model.pt is missing, on-device load is impossible and
    from_pretrained would otherwise trigger a multi-hour recompile.
    """
    root = Path(COMPILE_CACHE) / "ltx_2"
    if not root.is_dir():
        print(f"{P} preflight FAIL: no ltx_2 cache dir under {COMPILE_CACHE}", file=sys.stderr)
        return 2
    hashes = [d for d in root.iterdir() if d.is_dir()]
    ok = False
    for h in hashes:
        tdir = h / "transformer"
        has_pt = (tdir / "model.pt").is_file()
        has_cfg = (tdir / "neuron_config.json").is_file()
        print(f"{P} preflight artifact {tdir}: model.pt={has_pt} neuron_config.json={has_cfg}")
        ok = ok or (has_pt and has_cfg)
    if not ok:
        print(
            f"{P} preflight FAIL: no complete compiled transformer artifact "
            "(model.pt missing). Refusing to recompile.",
            file=sys.stderr,
        )
        return 2
    print(f"{P} preflight PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--smoke", action="store_true", help="4-step smoke only")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    rc = _preflight()
    if rc != 0 or args.preflight_only:
        return rc

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    from difflet import DiffletParallelConfig, DiffletPipeline
    from difflet.models.ltx_2.pipeline import LTX2DiTInputBundle

    dtype = torch.bfloat16

    print(f"{P} building app via DiffletPipeline.from_pretrained (skip_compile=True)", flush=True)
    pipe = DiffletPipeline.from_pretrained(
        MODEL_DIR,
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=TP_DEGREE),
        dtype=dtype,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        compile_cache_dir=COMPILE_CACHE,
        skip_compile=True,
        load=True,
        skip_warmup=False,
        application_kwargs={
            "transformer_mode": "single",
            "enable_host_pipeline": True,
            "enable_decode_components": False,
            "host_device": "cpu",
            "text_seq_len": TEXT_SEQ_LEN,
            "frame_rate": 24.0,
        },
    )
    print(f"{P} resolved compiled_path = {pipe.compiled_path}", flush=True)

    app = pipe.app
    orch = app.pipeline
    # Ensure the orchestrator drives the loaded single-mode transformer (the app
    # delegate exposes teacache_mod_input + forward_dit).
    orch.transformer = app
    print(f"{P} orchestrator transformer = {type(orch.transformer).__name__}", flush=True)

    # ---- load cached bundle ----
    raw = load_file(BUNDLE)
    for k, v in raw.items():
        print(f"{P} bundle {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    def make_bundle() -> LTX2DiTInputBundle:
        bs = raw["latents_init"].shape[0]
        # timestep/sigma are recomputed per-step inside _denoise; supply placeholders.
        ts = torch.zeros(bs, dtype=dtype)
        return LTX2DiTInputBundle(
            hidden_states=raw["latents_init"].clone().to(dtype),
            audio_hidden_states=raw["audio_latents_init"].clone().to(dtype),
            encoder_hidden_states=raw["encoder_hidden_states"].clone().to(dtype),
            audio_encoder_hidden_states=raw["audio_encoder_hidden_states"].clone().to(dtype),
            timestep=ts,
            sigma=ts.clone(),
            encoder_attention_mask=raw["encoder_attention_mask"].clone().to(torch.bool),
            audio_encoder_attention_mask=raw["audio_encoder_attention_mask"].clone().to(torch.bool),
            video_coords=raw["video_coords"].clone().to(torch.float32),
            audio_coords=raw["audio_coords"].clone().to(torch.float32),
        )

    num_steps = 4 if args.smoke else args.num_steps
    timesteps = orch._timesteps(num_steps, device=torch.device("cpu"))
    print(f"{P} timesteps ({num_steps} steps): {timesteps.tolist()}", flush=True)

    denoise_kwargs = dict(
        guidance_scale=1.0,
        audio_guidance_scale=1.0,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        guidance_rescale=0.0,
        audio_guidance_rescale=0.0,
        spatio_temporal_guidance_blocks=None,
        use_cross_timestep=False,
        attention_kwargs=None,
    )

    def run(label: str, calib_path):
        orch.teacache_calibration_path = calib_path
        orch._teacache_controller = None
        bundle = make_bundle()
        t0 = time.monotonic()
        vid, aud = orch._denoise(bundle=bundle, timesteps=timesteps, trajectory=None, **denoise_kwargs)
        dt = time.monotonic() - t0
        stats = orch._teacache_controller.stats() if orch._teacache_controller is not None else None
        print(f"{P} {label}: denoise {dt:.3f}s stats={stats}", flush=True)
        return dt, vid.detach().float().cpu(), aud.detach().float().cpu(), stats

    if args.smoke:
        print(f"{P} SMOKE: bundle-driven _denoise, {num_steps} steps, baseline path", flush=True)
        bt, bvid, baud, _ = run("smoke-baseline", None)
        tt, tvid, taud, stats = run("smoke-teacache", CALIB)
        print(f"{P} SMOKE OK baseline={bt:.3f}s teacache={tt:.3f}s stats={stats}", flush=True)
        return 0

    print(f"{P} === {num_steps}-step comparison ===", flush=True)
    bt, bvid, baud, _ = run("baseline", None)
    tt, tvid, taud, stats = run("teacache", CALIB)

    vid_cos = float(F.cosine_similarity(bvid.flatten().unsqueeze(0), tvid.flatten().unsqueeze(0)))
    aud_cos = float(F.cosine_similarity(baud.flatten().unsqueeze(0), taud.flatten().unsqueeze(0)))
    speedup = bt / tt if tt > 0 else float("nan")

    print(f"{P} RESULT baseline_s={bt:.3f} teacache_s={tt:.3f} speedup={speedup:.3f}x", flush=True)
    print(f"{P} RESULT teacache_stats={stats}", flush=True)
    print(f"{P} RESULT video_cosine={vid_cos:.6f} audio_cosine={aud_cos:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
