"""Real-denoise-loop per-step latency — the metric measured *consistently with H100*.

The H100 reference (``benchmark/adapters/diffusers_ref.py``) takes per-step as the
wall time between consecutive ``callback_on_step_end`` invocations of the diffusers
denoise loop (cuda-synced; the first step is naturally excluded because no delta
covers it). That is a *real generate* — true latents, real scheduler/guidance
interleaving — not an isolated transformer forward on synthetic inputs.

This script measures the SAME quantity on Trainium: it builds the difflet pipeline
exactly as the CLI orchestrator does (tp=4, compiled NEFF, ``skip_compile=True``),
wraps the per-step DiT call so a synchronized ``perf_counter`` is recorded after
each step, runs ONE real generate, and reports the inter-step deltas (range(1,N) —
step 0 excluded, identical to the H100 method). The Neuron DiT call returns host
tensors, so it is synchronous: the post-call timestamp is already a device-sync
point, the trn2 analog of ``torch.cuda.synchronize()``.

This REPLACES the isolated synthetic-input timer (``benchmark/step_latency.py``)
for cross-device comparison: that timer used a DIFFERENT method per model (n=20
in-process forward for Wan/Qwen/Hunyuan, an n=1 parity script for LTX-2, the tqdm
denoise-loop rate for FLUX), so its numbers were not measured the same way as each
other or as H100. Here every model uses one method, the H100 method. (The legacy
timer also looks up pre-hash compiled-dir names that current orchestrators no
longer write, so it cannot find a fresh artifact at all.) The staged models
(wan, qwen_image, hunyuan_video) are driven through their own CLI stage code --
see the "Staged models" section below.

    source .venv/bin/activate
    python -m benchmark.step_realloop --model flux_1_dev   # ltx_2 / wan_2_1 / qwen_image / hunyuan_video
    python -m benchmark.step_realloop --model flux_1_dev --config tp2cp2

Patches benchmark/<device>/<slug>[_<config>].json (step_latency, throughput) + re-renders md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmark import report
from benchmark.harness import Stats
from benchmark.models import add_config_arg, json_path, report_path, resolve


def _install_timer(cls, method_name: str, stamps: list[float]):
    """Wrap ``cls.method_name`` to append a synchronized perf_counter after each
    call. Returns a restore() thunk. The wrapped call is the per-step DiT eval."""
    orig = getattr(cls, method_name)

    def timed(self, *a, **k):
        r = orig(self, *a, **k)
        stamps.append(time.perf_counter())  # Neuron forward returns host tensors -> synced
        return r

    setattr(cls, method_name, timed)
    return lambda: setattr(cls, method_name, orig)


def _build_flux(cfg, cache):
    """Mirror difflet/cli/orchestrators/flux.py exactly."""
    import torch
    from difflet.pipeline.difflet_pipeline import DiffletPipeline
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    parallel = DiffletParallelConfig(tp_degree=cfg.tp, cp_degree=cfg.cp, cp_mode=cfg.cp_mode, cfg_parallel_enabled=cfg.cfg_parallel, sp_enabled=cfg.sp)
    pipe = DiffletPipeline.from_pretrained(
        cfg.model_id, model_type="flux", parallel=parallel, dtype=torch.bfloat16,
        height=cfg.height, width=cfg.width, compile_cache_dir=str(cache),
        revision=cfg.revision, skip_compile=True)
    dit = pipe.app.pipe.transformer            # NeuronFluxBackboneApplication
    gen_kwargs = dict(
        prompt=cfg.prompt, num_inference_steps=cfg.steps,
        height=cfg.height or 1024, width=cfg.width or 1024,
        guidance_scale=cfg.guidance_scale or 3.5,
        generator=torch.Generator().manual_seed(cfg.seed),
        output_type="pt")  # tensor out so we can finite-check; denoise loop unaffected
    def out_finite(res):
        import torch as _t
        img = res.images if hasattr(res, "images") else res[0]
        t = img if isinstance(img, _t.Tensor) else None
        return None if t is None else bool(_t.isfinite(t).all())
    return (lambda: pipe(**gen_kwargs)), type(dit), "__call__", out_finite


def _build_ltx2(cfg, cache):
    """Mirror difflet/cli/orchestrators/ltx_2.py exactly (host pipeline + decode)."""
    import torch
    from difflet.pipeline.difflet_pipeline import DiffletPipeline
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    parallel = DiffletParallelConfig(tp_degree=cfg.tp, cp_degree=cfg.cp, cp_mode=cfg.cp_mode, cfg_parallel_enabled=cfg.cfg_parallel, sp_enabled=cfg.sp)
    pipe = DiffletPipeline.from_pretrained(
        cfg.model_id, model_type="ltx_2", parallel=parallel, dtype=torch.bfloat16,
        height=cfg.height, width=cfg.width, num_frames=cfg.num_frames,
        compile_cache_dir=str(cache), revision=cfg.revision, skip_compile=True,
        application_kwargs={"enable_host_pipeline": True, "enable_decode_components": True})
    app = pipe.app                              # NeuronLTX2Application; loop calls forward_dit/step
    gen_kwargs = dict(
        prompt=cfg.prompt, num_inference_steps=cfg.steps,
        guidance_scale=cfg.guidance_scale or 1.0,
        generator=torch.Generator().manual_seed(cfg.seed), output_type="pt")
    def out_finite(res):
        import torch as _t
        fr = res.frames if hasattr(res, "frames") else res[0]
        return bool(_t.isfinite(fr).all()) if isinstance(fr, _t.Tensor) else None
    # per-step boundary: NeuronLTX2Application.forward_dit (one batched DiT call / step)
    return (lambda: pipe(**gen_kwargs)), type(app), "forward_dit", out_finite


# --------------------------------------------------------------------------- #
# Staged models (wan / qwen_image / hunyuan_video)
#
# These are NOT in-process DiffletPipeline models: the CLI runs their text
# encoder / transformer / VAE as separate ``difflet.cli.stage`` subprocesses
# with hash-named artifact dirs (orchestrators/base.py hashed_stage_dir), so the
# pipeline-style loader above would look for an artifact that does not exist
# (and the legacy ``benchmark/step_latency.py`` looks up pre-hash directory
# names that current orchestrators no longer write). Instead we drive the
# orchestrator's OWN stage code: pre-stages (text encoders) run as subprocesses
# exactly as ``difflet generate`` runs them -- they must not share a process
# with the DiT (mixed Neuron world sizes in one process SIGSEGV at init) -- and
# the DiT stage runs in-process via ``_run_stage_internal`` with the backbone
# application's ``__call__`` wrapped per step. Same method as
# scripts/parallel_phase_sweep.py's worker_wan / worker_qwen.
# --------------------------------------------------------------------------- #

def _staged_namespace(cfg, cache, work_dir, output):
    """The parent ``difflet generate`` argparse namespace the orchestrator expects."""
    import argparse as _ap
    return _ap.Namespace(
        model_id=cfg.model_id, revision=cfg.revision,
        tp_degree=cfg.tp, cp_degree=cfg.cp, cp_mode=cfg.cp_mode,
        cfg_parallel=cfg.cfg_parallel, sp_enabled=cfg.sp,
        attention_impl=cfg.attention_impl,
        height=cfg.height, width=cfg.width, num_frames=cfg.num_frames,
        steps=cfg.steps, guidance_scale=cfg.guidance_scale, seed=cfg.seed,
        prompt=cfg.prompt, output=str(output), shapes=None,
        cache_dir=str(cache), work_dir=str(work_dir), keep_work_dir=True,
        requests_dir=None, worker_index=0, dp_schedule="round_robin",
        dp_degree=1, host_vae=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )


def _stage_ns(model_id, stage, argv):
    from difflet.cli.stage import _build_stage_parser
    return _build_stage_parser().parse_args(
        ["--orchestrator", model_id, "--stage", stage] + argv)


def _run_pre_stage(model_id, stage, *, num_cores, virtual_core_size, argv, produces):
    """Run a text-encoder stage as ``difflet generate`` does (subprocess with the
    stage's own Neuron env), unless its output already sits in the work dir."""
    from difflet.cli import runner
    if all(p.exists() for p in produces):
        print(f"[realloop] pre-stage {stage}: reusing {[p.name for p in produces]}", flush=True)
        return
    print(f"[realloop] pre-stage {stage}: running as a subprocess "
          f"(num_cores={num_cores}, virtual_core_size={virtual_core_size})", flush=True)
    runner.run_stage(model_id, stage, num_cores=num_cores,
                     virtual_core_size=virtual_core_size, cli_args=argv)


def _latents_finite(path):
    def out_finite(_res):
        import torch as _t
        try:
            x = _t.load(path, map_location="cpu")
            x = x[0] if isinstance(x, (list, tuple)) else x
            return bool(_t.isfinite(x).all())
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[realloop] finite-check skipped: {exc}", flush=True)
            return None
    return out_finite


def _work_dir(cfg, cache):
    d = cache / "work" / f"bench_{cfg.config_slug}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_wan(cfg, cache):
    """Wan transformer stage (UMT5 text-encode + denoise loop in ONE stage, as
    difflet/cli/orchestrators/wan.py runs it); NeuronWanBackboneApplication
    is the per-step DiT call (batch 2 under CFG-parallel)."""
    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication
    from difflet.cli.orchestrators.wan import WanOrchestrator

    work = _work_dir(cfg, cache)
    orch = WanOrchestrator(_staged_namespace(cfg, cache, work, work / "out.mp4"))
    argv = orch._shared_cli_args(stage_mode="generate", work_dir=str(work))
    ns = _stage_ns(cfg.model_id, "transformer", argv)

    def run():
        orch._run_stage_internal("transformer", ns)   # loads + one real generate
        return None
    return run, NeuronWanBackboneApplication, "__call__", _latents_finite(work / "latents.pt")


def _build_qwen(cfg, cache):
    """Qwen-Image: ``text`` stage (Qwen2.5-VL encoder) as a subprocess, then the
    ``generate`` (DiT) stage in-process; NeuronQwenImageTransformerApplication
    is the per-step DiT call."""
    from difflet.backends.trainium.qwen_image.transformer import (
        NeuronQwenImageTransformerApplication,
    )
    from difflet.cli.orchestrators import qwen_image as qo

    work = _work_dir(cfg, cache)
    orch = qo.QwenImageOrchestrator(_staged_namespace(cfg, cache, work, work / "out.png"))
    argv = orch._shared_cli_args(stage_mode="generate", work_dir=str(work))
    full_cores = cfg.tp * cfg.cp
    _run_pre_stage(qo._HF_MODEL_ID, "text", num_cores=full_cores,
                   virtual_core_size=qo._VIRTUAL_CORE_SIZE, argv=argv,
                   produces=[work / "text.pt"])
    ns = _stage_ns(qo._HF_MODEL_ID, "generate", argv)

    def run():
        orch._run_stage_internal("generate", ns)
        return None
    return run, NeuronQwenImageTransformerApplication, "__call__", _latents_finite(work / "latents.pt")


def _build_hunyuan(cfg, cache):
    """HunyuanVideo: ``clip`` (1 core) and ``llama`` stages as subprocesses,
    then the ``generate`` (DiT + VAE) stage in-process with the stage's
    NEURON_RT_VIRTUAL_CORE_SIZE; NeuronHunyuanVideoBackboneApplication is the
    per-step DiT call (guidance-distilled: one call per step)."""
    import os
    from difflet.backends.trainium.hunyuan_video.backbone import (
        NeuronHunyuanVideoBackboneApplication,
    )
    from difflet.cli.orchestrators import hunyuan_video as ho

    work = _work_dir(cfg, cache)
    out = work / "out.pt"   # non-.mp4 -> the stage saves the frames tensor
    orch = ho.HunyuanVideoOrchestrator(_staged_namespace(cfg, cache, work, out))
    argv = orch._shared_cli_args(stage_mode="generate", work_dir=str(work))
    full_cores = cfg.tp * cfg.cp
    _run_pre_stage(ho._HF_MODEL_ID, "clip", num_cores=1,
                   virtual_core_size=ho._VIRTUAL_CORE_SIZE, argv=argv,
                   produces=[work / "clip.pt"])
    _run_pre_stage(ho._HF_MODEL_ID, "llama", num_cores=full_cores,
                   virtual_core_size=ho._VIRTUAL_CORE_SIZE, argv=argv,
                   produces=[work / "llama.pt"])
    # difflet.cli.runner sets this for every HunyuanVideo stage subprocess; the
    # in-process DiT stage must see the same runtime topology as its artifact.
    if ho._VIRTUAL_CORE_SIZE is not None:
        os.environ.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", str(ho._VIRTUAL_CORE_SIZE))
    ns = _stage_ns(ho._HF_MODEL_ID, "generate", argv)

    def run():
        orch._run_stage_internal("generate", ns)
        return None
    return run, NeuronHunyuanVideoBackboneApplication, "__call__", _latents_finite(out)


_BUILDERS = {
    "flux_1_dev": _build_flux,
    "ltx_2": _build_ltx2,
    "wan_2_1": _build_wan,
    "wan_2_2": _build_wan,
    "qwen_image": _build_qwen,
    "hunyuan_video": _build_hunyuan,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(_BUILDERS))
    add_config_arg(p)
    p.add_argument("--generates", type=int, default=1,
                   help="run N generates back to back with the model resident; per-step "
                        "comes from the LAST one and every generate's wall is recorded as "
                        "resident_generate_s (the serving steady state, no reload). Default 1 "
                        "= the campaign rule (first generate of a fresh process).")
    args = p.parse_args()
    cfg = resolve(args.model, args.config)
    # --attention-impl is carried by the DIFFLET_ATTENTION_IMPL env var (the CLI
    # and difflet.cli.stage set it); it selects the compile-cache identity of
    # the in-process pipeline loaders and the kernel the in-process DiT stage
    # traces, so it must be set here the same way before anything is built.
    from difflet.ops.attention_config import attention_implementation
    with attention_implementation(cfg.attention_impl):
        return _main(args, cfg)


def _main(args, cfg) -> int:
    slug = cfg.config_slug   # result-file stem: <model>[_<config>]
    cache = Path("~/.cache/difflet").expanduser()

    print(f"[realloop] {slug}: building pipeline ({' '.join(cfg.parallel_flags())}, "
          f"compiled NEFF, skip_compile)...", flush=True)
    t0 = time.perf_counter()
    run, cls, method, out_finite = _BUILDERS[args.model](cfg, cache)
    load_s = time.perf_counter() - t0
    print(f"[realloop] {slug}: pipeline ready in {load_s:.1f}s; "
          f"timing {cls.__name__}.{method} per step", flush=True)

    stamps: list[float] = []
    restore = _install_timer(cls, method, stamps)
    resident: list[float] = []   # wall of each generate with the model resident
    try:
        for i in range(max(1, args.generates)):
            stamps.clear()
            t1 = time.perf_counter()
            result = run()
            gen_s = time.perf_counter() - t1
            resident.append(gen_s)
            if args.generates > 1:
                print(f"[realloop] {slug}: generate {i + 1}/{args.generates} "
                      f"{gen_s:.1f}s (model resident)", flush=True)
    finally:
        restore()

    n_calls = len(stamps)
    # inter-step deltas, step 0 excluded — identical to diffusers_ref.py (H100)
    deltas = [stamps[i] - stamps[i - 1] for i in range(1, n_calls)]
    if not deltas:
        print(f"[realloop] {slug}: ERROR only {n_calls} DiT call(s) timed; "
              "cannot form per-step deltas", file=sys.stderr)
        return 2
    if n_calls != cfg.steps:
        print(f"[realloop] {slug}: NOTE {n_calls} DiT calls for {cfg.steps} steps "
              f"({n_calls/cfg.steps:.2g}x/step — CFG/segmented); per-step = inter-call delta",
              flush=True)

    st = Stats.from_samples(deltas)
    finite = out_finite(result)
    print(f"[realloop] {slug}: per-step {st.mean*1000:.1f} ms "
          f"(median {st.median*1000:.1f}, p90 {st.p90*1000:.1f}, n={st.n}); "
          f"generate {gen_s:.1f}s, finite={finite}", flush=True)

    jp = Path(json_path(slug))
    d = json.loads(jp.read_text())
    d["step_latency"] = st.__dict__
    if st.mean > 0:
        d["throughput"] = {"DiT steps/s": round(1.0 / st.mean, 3)}
    d["config_slug"] = slug
    d["model_slug"] = args.model
    d["config"] = cfg.config
    # >1 when the loop calls the DiT more than once per scheduler step (two
    # sequential CFG branches at guidance > 1 without cfg-parallel); the
    # per-step figure above is then the inter-CALL delta and a step costs
    # dit_calls_per_step x that.
    d["dit_calls_per_step"] = round(n_calls / cfg.steps, 3)
    if args.generates > 1:
        # steady state with the model resident (no process start, no reload)
        d["resident_generate_s"] = [round(x, 3) for x in resident]
        d["notes"] = [n for n in d.get("notes", []) if not n.startswith("resident generate")]
        d["notes"].append(
            f"resident generate = {resident[-1]:.1f} s (generate {len(resident)} of "
            f"{len(resident)} in one process, model loaded once; walls {[round(x, 1) for x in resident]}) "
            "-- the serving steady state; per-step above is from the last generate.")
    # drop the prior (inconsistent-method) per-step note, append the H100-consistent one
    notes = [n for n in d.get("notes", []) if not n.lstrip().startswith("per-step =")]
    notes.insert(0,
        f"per-step = {st.mean*1000:.1f} ms/DiT-step (median {st.median*1000:.1f}, "
        f"p90 {st.p90*1000:.1f}, n={st.n}) — measured the SAME way as H100: inter-step "
        f"deltas of a real {cfg.steps}-step generate (wrapping {cls.__name__}.{method}, "
        f"synced, step 0 excluded), NOT the old isolated synthetic-input timer. "
        f"{n_calls} DiT calls timed; warm generate {gen_s:.0f}s; output finite={finite}.")
    d["notes"] = notes
    jp.write_text(json.dumps(d, indent=2))
    Path(report_path(slug)).write_text(report.render(d))
    print(f"[realloop] patched {report_path(slug)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
