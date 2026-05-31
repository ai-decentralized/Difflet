"""On-device (Trainium) end-to-end TeaCache validation for LTX-2 — SEGMENTED mode.

LTX-2 cannot compile as a single NEFF (neuronx-cc aborts: 24.5M instructions >>
5M limit), so the only on-device path is the per-block SEGMENTED runtime: one
compiled block artifact, weights reloaded per block index, host-stitched across
the 48 transformer layers.

This driver bypasses text encoding: it builds the app in ``transformer_mode=
segmented`` (block_load_mode="streaming": one in-process Neuron load of the
block, then a per-index weight reload), points it at the compiled block artifact
via ``set_compiled_model_path``, and drives ``LTX2Orchestrator._denoise``
directly with the cached precomputed-embeds bundle, comparing baseline vs
teacache for denoise wall-clock speedup + output cosine.

CFG / STG / modality are all disabled (guidance_scale=1.0, stg=0, modality=1.0,
guidance_rescale=0) so each step is exactly one transformer eval and the
TeaCache skip path is clean (one host signal -> skip or one full DiT pass).

Usage (detached, see cclog instructions):
    setsid bash -c 'PYTHONPATH=/home/ubuntu/nova \
      PATH=/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin:$PATH \
      NOVA_BACKEND=trainium NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
      TORCH_DISABLE_ADDR2LINE=1 NEURON_RT_LOG_LEVEL=ERROR \
      python scripts/run_ltx2_teacache_e2e_segmented.py --num-steps 50 > log 2>&1' \
      < /dev/null & disown
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

P = "[ltx2-e2e-seg]"

MODEL_DIR = (
    "/home/ubuntu/.cache/huggingface/hub/models--Lightricks--LTX-2/"
    "snapshots/47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
)
COMPILE_CACHE = "/home/ubuntu/nova/.nova-cache/ltx_2_segmented"
BUNDLE = "/home/ubuntu/nova/.nova-cache/ltx_2_dit_inputs/full_512x768x121_4step.safetensors"
CALIB = "/home/ubuntu/nova/cclogs/m9-teacache/teacache_calib_ltx_2.json"

HEIGHT, WIDTH, NUM_FRAMES = 512, 768, 121
AUDIO_NUM_FRAMES = 126
TEXT_SEQ_LEN = 1024
TP_DEGREE = 4


def _preflight() -> int:
    """Verify the compiled segmented block artifact actually exists.

    The content-addressed layout is <cache>/ltx_2/<hash>/transformer_block/
    {model.pt, neuron_config.json}. If model.pt is missing, on-device load is
    impossible and from_pretrained would otherwise trigger a multi-hour
    recompile.
    """
    root = Path(COMPILE_CACHE) / "ltx_2"
    if not root.is_dir():
        print(f"{P} preflight FAIL: no ltx_2 cache dir under {COMPILE_CACHE}", file=sys.stderr)
        return 2
    hashes = [d for d in root.iterdir() if d.is_dir()]
    ok = False
    for h in hashes:
        tdir = h / "transformer_block"
        has_pt = (tdir / "model.pt").is_file()
        has_cfg = (tdir / "neuron_config.json").is_file()
        print(f"{P} preflight artifact {tdir}: model.pt={has_pt} neuron_config.json={has_cfg}")
        ok = ok or (has_pt and has_cfg)
    if not ok:
        print(
            f"{P} preflight FAIL: no complete compiled transformer_block artifact "
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
    ap.add_argument(
        "--block-load-mode",
        choices=("streaming", "process"),
        default="streaming",
        help=(
            "streaming = one in-process Neuron block load + per-index weight "
            "reload (fast, the only viable mode for a 50-step e2e); process = "
            "spawn a fresh Neuron process per block forward (slow)."
        ),
    )
    args = ap.parse_args()

    rc = _preflight()
    if rc != 0 or args.preflight_only:
        return rc

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    from nova import NovaParallelConfig, NovaPipeline
    from nova.models.ltx_2.pipeline import LTX2DiTInputBundle

    dtype = torch.bfloat16

    # In streaming mode the driver process itself loads the block (one Neuron
    # init), so load=True. In process mode the driver stays Neuron-free and each
    # block forward spawns its own process, so load=False.
    load_in_process = args.block_load_mode == "streaming"

    # IMPORTANT: application_kwargs is hashed into the compile-cache key. To
    # resolve the compiled block artifact produced by
    # ltx_2_segmented_block_compile_probe.py (hash f2b170335694cbc7), these three
    # keys must match byte-for-byte. enable_host_pipeline / block_load_mode /
    # host_device are deliberately NOT passed here: the orchestrator loads the
    # scheduler from model_path on its own, and teacache_mod_input /
    # _prepare_frontend use the segmented runtime's own host CPU transformer copy
    # (not the diffusers host pipeline). We set block_load_mode post-construction.
    print(
        f"{P} building app via NovaPipeline.from_pretrained "
        f"(skip_compile=True, block_load_mode={args.block_load_mode}, load={load_in_process})",
        flush=True,
    )
    pipe = NovaPipeline.from_pretrained(
        MODEL_DIR,
        model_type="ltx_2",
        parallel=NovaParallelConfig(tp_degree=TP_DEGREE),
        dtype=dtype,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        compile_cache_dir=COMPILE_CACHE,
        skip_compile=True,
        load=load_in_process,
        skip_warmup=True,
        application_kwargs={
            "transformer_mode": "segmented",
            "audio_num_frames": AUDIO_NUM_FRAMES,
            "text_seq_len": TEXT_SEQ_LEN,
        },
    )
    print(f"{P} resolved compiled_path = {pipe.compiled_path}", flush=True)

    app = pipe.app
    orch = app.pipeline
    # Ensure the orchestrator drives the app delegate (exposes teacache_mod_input
    # + forward_dit, which dispatch to the segmented backend).
    orch.transformer = app

    seg = app.transformer  # LTX2SegmentedTransformerApplication
    seg.block_load_mode = args.block_load_mode
    set_compiled_model_path = getattr(seg, "set_compiled_model_path", None)
    if set_compiled_model_path is not None:
        set_compiled_model_path(str(pipe.compiled_path))
    print(
        f"{P} segmented transformer={type(seg).__name__} "
        f"block_load_mode={seg.block_load_mode} "
        f"compiled_model_path={getattr(seg, '_compiled_model_path', None)}",
        flush=True,
    )
    # Confirm the teacache host signal resolves on the app delegate.
    assert hasattr(orch.transformer, "teacache_mod_input"), "missing teacache_mod_input"
    print(f"{P} orchestrator transformer = {type(orch.transformer).__name__}", flush=True)

    # ---- load cached bundle ----
    raw = load_file(BUNDLE)
    for k, v in raw.items():
        print(f"{P} bundle {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    def make_bundle() -> LTX2DiTInputBundle:
        bs = raw["latents_init"].shape[0]
        ts = torch.zeros(bs, dtype=dtype)  # recomputed per-step inside _denoise
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
        # Re-derive timesteps each run: _timesteps() calls scheduler.set_timesteps,
        # which resets the shared video scheduler's internal step_index/sigmas.
        # _denoise reads self.scheduler.sigmas by step_index and steps it in place,
        # so without this reset the second run continues from the first run's final
        # index and overflows (IndexError: index N out of bounds).
        run_timesteps = orch._timesteps(num_steps, device=torch.device("cpu"))
        t0 = time.monotonic()
        vid, aud = orch._denoise(
            bundle=bundle, timesteps=run_timesteps, trajectory=None, **denoise_kwargs
        )
        dt = time.monotonic() - t0
        stats = orch._teacache_controller.stats() if orch._teacache_controller is not None else None
        print(f"{P} {label}: denoise {dt:.3f}s stats={stats}", flush=True)
        return dt, vid.detach().float().cpu(), aud.detach().float().cpu(), stats

    if args.smoke:
        print(f"{P} SMOKE: bundle-driven _denoise, {num_steps} steps", flush=True)
        bt, bvid, baud, _ = run("smoke-baseline", None)
        tt, tvid, taud, stats = run("smoke-teacache", CALIB)
        vid_cos = float(F.cosine_similarity(bvid.flatten().unsqueeze(0), tvid.flatten().unsqueeze(0)))
        aud_cos = float(F.cosine_similarity(baud.flatten().unsqueeze(0), taud.flatten().unsqueeze(0)))
        print(
            f"{P} SMOKE OK baseline={bt:.3f}s teacache={tt:.3f}s stats={stats} "
            f"video_cosine={vid_cos:.6f} audio_cosine={aud_cos:.6f}",
            flush=True,
        )
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
