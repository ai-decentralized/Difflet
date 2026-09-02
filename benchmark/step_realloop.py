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
other or as H100. Here every model uses one method, the H100 method.

    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python -m benchmark.step_realloop --model flux_1_dev   # or ltx_2

Patches benchmark/<device>/<slug>.json (step_latency, throughput) + re-renders md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmark import report
from benchmark.harness import Stats
from benchmark.models import MATRIX, json_path, report_path


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
    return pipe, type(dit), "__call__", gen_kwargs, out_finite


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
    return pipe, type(app), "forward_dit", gen_kwargs, out_finite


_BUILDERS = {"flux_1_dev": _build_flux, "ltx_2": _build_ltx2}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(_BUILDERS))
    args = p.parse_args()
    cfg = MATRIX[args.model]
    cache = Path("~/.cache/difflet").expanduser()

    print(f"[realloop] {args.model}: building pipeline (tp={cfg.tp}, "
          f"compiled NEFF, skip_compile)...", flush=True)
    t0 = time.perf_counter()
    pipe, cls, method, gen_kwargs, out_finite = _BUILDERS[args.model](cfg, cache)
    load_s = time.perf_counter() - t0
    print(f"[realloop] {args.model}: pipeline ready in {load_s:.1f}s; "
          f"timing {cls.__name__}.{method} per step", flush=True)

    stamps: list[float] = []
    restore = _install_timer(cls, method, stamps)
    try:
        t1 = time.perf_counter()
        result = pipe(**gen_kwargs)
        gen_s = time.perf_counter() - t1
    finally:
        restore()

    n_calls = len(stamps)
    # inter-step deltas, step 0 excluded — identical to diffusers_ref.py (H100)
    deltas = [stamps[i] - stamps[i - 1] for i in range(1, n_calls)]
    if not deltas:
        print(f"[realloop] {args.model}: ERROR only {n_calls} DiT call(s) timed; "
              "cannot form per-step deltas", file=sys.stderr)
        return 2
    if n_calls != cfg.steps:
        print(f"[realloop] {args.model}: NOTE {n_calls} DiT calls for {cfg.steps} steps "
              f"({n_calls/cfg.steps:.2g}x/step — CFG/segmented); per-step = inter-call delta",
              flush=True)

    st = Stats.from_samples(deltas)
    finite = out_finite(result)
    print(f"[realloop] {args.model}: per-step {st.mean*1000:.1f} ms "
          f"(median {st.median*1000:.1f}, p90 {st.p90*1000:.1f}, n={st.n}); "
          f"generate {gen_s:.1f}s, finite={finite}", flush=True)

    jp = Path(json_path(args.model))
    d = json.loads(jp.read_text())
    d["step_latency"] = st.__dict__
    if st.mean > 0:
        d["throughput"] = {"DiT steps/s": round(1.0 / st.mean, 3)}
    d["config_slug"] = args.model
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
    Path(report_path(args.model)).write_text(report.render(d))
    print(f"[realloop] patched {report_path(args.model)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
