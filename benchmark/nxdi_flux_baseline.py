"""Native AWS NxDI FLUX.1-dev baseline, measured with the Difflet campaign's rules.

The NxDI package (``neuronx_distributed_inference``, shipped in the same Neuron
venv) has its own FLUX application -- Difflet's ``modeling_flux`` is a fork of
it -- and AWS's ``examples/generate_flux.py`` drives it as: build configs
(``create_flux_config``), construct ``NeuronFluxApplication`` (which loads the
full diffusers pipeline on the host), ``compile``, ``load``, 5 warm-up
generates, then N timed generates and an "Average generation time". That
average is a *resident-model* number; the campaign's e2e is a fresh process
per generate (weights re-loaded), so this script mirrors the example's setup
and measures each phase the way the campaign does:

    python -m benchmark.nxdi_flux_baseline compile  --workdir W      # fresh W; timed
    python -m benchmark.nxdi_flux_baseline generate --workdir W --out o.png   # one process:
        # load + ONE 28-step generate (no warm-up), seed 42, NeuronFluxBackboneApplication
        # wrapped per step (inter-step deltas, step 0 excluded = the real-loop rule)
    python -m benchmark.nxdi_flux_baseline record   --compile c.log --cold cold.log \
        --warm warm.log --cold-wall S --warm-wall S     # -> benchmark/trn2/flux_1_dev_nxdi.json

``benchmark/trn2/nxdi_flux_baseline.sh`` chains these (drop_caches before the
cold generate). Phases print one ``NXDI_RESULT {json}`` line each.

Not apples-to-apples with Difflet, by design (recorded in the JSON notes):
NxDI's application loads the whole diffusers pipeline on the host every
process (Difflet loads only what the Neuron stages need, with presharded
per-rank checkpoints), and NxDI runs a warm-up forward per component inside
``load()``. Attention kernel (attention_cte), CLIP tp=1 / T5 tp=world /
VAE tp=1 placement and compiler flags are the same lineage.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

MODEL_ID = "black-forest-labs/FLUX.1-dev"
_RESULT = "NXDI_RESULT "


def _snapshot(model_slug: str = "flux_1_dev") -> tuple[str, str | None]:
    """The cached HF diffusers snapshot dir for the campaign's pinned revision."""
    from benchmark.models import MATRIX
    cfg = MATRIX[model_slug]
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snap = hub / f"models--{cfg.model_id.replace('/', '--')}" / "snapshots" / (cfg.revision or "")
    if not snap.is_dir():
        raise SystemExit(f"[nxdi] snapshot not found: {snap} (run difflet download first)")
    return str(snap), cfg.revision


def _build_app(ckpt: str, tp: int, height: int, width: int):
    import torch
    from neuronx_distributed_inference.models.diffusers.flux.application import (
        NeuronFluxApplication, create_flux_config, get_flux_parallelism_config,
    )
    from neuronx_distributed_inference.utils.random import set_random_seed

    set_random_seed(0)   # as examples/generate_flux.py
    world = get_flux_parallelism_config(tp)
    clip_c, t5_c, bb_c, dec_c = create_flux_config(ckpt, world, tp, torch.bfloat16, height, width)
    t0 = time.perf_counter()
    app = NeuronFluxApplication(
        model_path=ckpt, text_encoder_config=clip_c, text_encoder2_config=t5_c,
        backbone_config=bb_c, decoder_config=dec_c, height=height, width=width,
    )
    host_load_s = time.perf_counter() - t0
    return app, host_load_s, world


def _emit(d: dict) -> None:
    print(_RESULT + json.dumps(d), flush=True)


def cmd_compile(a) -> int:
    wd = Path(a.workdir)
    if wd.exists() and any(wd.iterdir()):
        if not a.force:
            raise SystemExit(f"[nxdi] {wd} is not empty; NxDI skips existing component dirs, "
                             "which would not be a real compile timing (use --force)")
    wd.mkdir(parents=True, exist_ok=True)
    ckpt, rev = _snapshot()
    os.environ.setdefault("BASE_COMPILE_WORK_DIR", str(Path(a.scratch).expanduser()) + "/")
    print(f"[nxdi] compile: ckpt={ckpt} tp={a.tp} {a.height}x{a.width} -> {wd}", flush=True)
    app, host_load_s, world = _build_app(ckpt, a.tp, a.height, a.width)
    t0 = time.perf_counter()
    app.compile(str(wd))
    compile_s = time.perf_counter() - t0
    _emit({"phase": "compile", "compile_seconds": round(compile_s, 3),
           "host_pipeline_load_seconds": round(host_load_s, 3), "world_size": world,
           "revision": rev, "workdir": str(wd)})
    return 0


def cmd_generate(a) -> int:
    import torch
    from neuronx_distributed_inference.models.diffusers.flux.modeling_flux import (
        NeuronFluxBackboneApplication,
    )
    t_proc = time.perf_counter()
    ckpt, rev = _snapshot()
    app, host_load_s, world = _build_app(ckpt, a.tp, a.height, a.width)
    t0 = time.perf_counter()
    app.load(a.workdir)          # NxDI runs its own per-component warm-up forward here
    load_s = time.perf_counter() - t0

    stamps: list[float] = []
    orig = NeuronFluxBackboneApplication.forward

    def timed(self, *x, **k):
        r = orig(self, *x, **k)
        stamps.append(time.perf_counter())   # host tensors returned -> synced
        return r

    NeuronFluxBackboneApplication.forward = timed
    try:
        t1 = time.perf_counter()
        out = app(a.prompt, height=a.height, width=a.width, guidance_scale=a.guidance,
                  num_inference_steps=a.steps, output_type="pt",
                  generator=torch.Generator().manual_seed(a.seed))
        gen_s = time.perf_counter() - t1
    finally:
        NeuronFluxBackboneApplication.forward = orig
    img = out.images
    img = img[0] if isinstance(img, (list, tuple)) else img
    finite = bool(torch.isfinite(img).all())
    if a.out:
        from PIL import Image
        arr = (img[0] if img.dim() == 4 else img).float().clamp(0, 1)
        arr = (arr.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(a.out)
    deltas = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    _emit({"phase": "generate", "wall_seconds": round(time.perf_counter() - t_proc, 3),
           "host_pipeline_load_seconds": round(host_load_s, 3),
           "load_seconds": round(load_s, 3), "generate_seconds": round(gen_s, 3),
           "n_dit_calls": len(stamps), "step_seconds": [round(d, 6) for d in deltas],
           "finite": finite, "output": a.out, "world_size": world, "revision": rev})
    return 0


def _parse(log: str) -> dict:
    text = Path(log).read_text(errors="ignore")
    lines = [l for l in text.splitlines() if l.startswith(_RESULT)]
    if not lines:
        raise SystemExit(f"[nxdi] no {_RESULT.strip()} line in {log}")
    return json.loads(lines[-1][len(_RESULT):])


def cmd_record(a) -> int:
    from benchmark import report
    from benchmark.adapters.trainium import TrainiumAdapter
    from benchmark.harness import Stats
    from benchmark.models import DEVICE, MATRIX, json_path, report_path, results_dir

    cfg = MATRIX["flux_1_dev"]
    comp, cold, warm = _parse(a.compile), _parse(a.cold), _parse(a.warm)
    ad = TrainiumAdapter()
    st = Stats.from_samples(warm["step_seconds"])
    slug = "flux_1_dev_nxdi"
    cold_wall = a.cold_wall if a.cold_wall is not None else cold["wall_seconds"]
    warm_wall = a.warm_wall if a.warm_wall is not None else warm["wall_seconds"]
    d = {
        "model_id": cfg.model_id, "model_type": "flux", "backend": "nxdi",
        "device": ad.device_info(), "dtype": "bf16",
        "parallel": {"tp_degree": a.tp, "cp_degree": 1, "cp_mode": "gather_kv",
                     "cfg_parallel_enabled": False, "sp_enabled": False},
        "shape": {"height": cfg.height, "width": cfg.width, "num_frames": None},
        "steps": cfg.steps,
        "compile_seconds": comp["compile_seconds"], "compile_breakdown": {
            "host_pipeline_load_s": comp["host_pipeline_load_seconds"],
            "wall_total_s": comp["compile_seconds"]},
        "e2e_cold_seconds": round(cold_wall, 3),
        "load_seconds": cold["load_seconds"],
        "e2e_breakdown": {"host_pipeline_load_s": cold["host_pipeline_load_seconds"],
                          "weights_load_total_s": cold["load_seconds"],
                          "generate_s": cold["generate_seconds"], "wall_total_s": cold_wall},
        "e2e_warm": Stats.from_samples([warm_wall]).__dict__,
        "e2e_warm_breakdown": {"host_pipeline_load_s": warm["host_pipeline_load_seconds"],
                               "weights_load_total_s": warm["load_seconds"],
                               "generate_s": warm["generate_seconds"], "wall_total_s": warm_wall},
        "step_latency": st.__dict__, "step_basis": "real-loop (NxDI backbone forward, synced)",
        "throughput": {"DiT steps/s": round(1.0 / st.mean, 3)} if st.mean > 0 else {},
        "output": {"finite": warm["finite"], "note": f"saved {Path(warm['output']).name}"},
        "status": "ok", "config_slug": slug, "model_slug": "flux_1_dev", "config": "tp4",
        "device_slug": DEVICE, "revision": cfg.revision, "seed": cfg.seed,
        "prompt": cfg.prompt, "guidance_scale": cfg.guidance_scale, "output_kind": "image",
        "config_label": (f"NATIVE NxDI baseline (neuronx_distributed_inference "
                         f"{_nxdi_version()}), examples/generate_flux.py setup: "
                         f"backbone tp={a.tp}, world={cold['world_size']}, bf16, CLIP tp=1, "
                         "T5 tp=world, VAE decoder tp=1"),
        "toolchain": ad.toolchain(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "notes": [
            "Native NxDI baseline for the difflet FLUX tp4 row, same weights/shape/steps/"
            "seed/prompt/guidance. Measured with the campaign rules: compile = timed "
            "NeuronFluxApplication.compile into a fresh workdir; e2e cold = a fresh "
            "process after dropping the OS page cache, load + ONE generate, no warm-up; "
            "e2e warm = the next identical process; per-step = inter-step deltas of the "
            f"backbone forward inside that real generate, step 0 excluded (n={st.n}).",
            "NOT apples-to-apples with difflet: NxDI's NeuronFluxApplication loads the full "
            "diffusers pipeline on the host in every process "
            f"(host_pipeline_load {cold['host_pipeline_load_seconds']:.0f} s cold / "
            f"{warm['host_pipeline_load_seconds']:.0f} s warm, inside e2e) and load() runs "
            "a warm-up forward per component; difflet loads only the Neuron stages from "
            "presharded per-rank checkpoints. The DiT per-step is the comparable number.",
            f"NxDI's own metric (examples/generate_flux.py 'Average generation time', "
            f"resident model, after 5 warm-ups) corresponds to generate_s = "
            f"{warm['generate_seconds']:.1f} s here (1 generate, no warm-up).",
        ],
    }
    Path(results_dir()).mkdir(parents=True, exist_ok=True)
    Path(json_path(slug)).write_text(json.dumps(d, indent=2))
    Path(report_path(slug)).write_text(report.render(d))
    print(f"[nxdi] recorded {json_path(slug)}: compile {comp['compile_seconds']:.0f}s, "
          f"cold {cold_wall:.0f}s, warm {warm_wall:.0f}s, per-step {st.mean*1000:.1f} ms "
          f"(n={st.n}), finite={warm['finite']}", flush=True)
    return 0


def _nxdi_version() -> str:
    try:
        import importlib.metadata as m
        return m.version("neuronx-distributed-inference")
    except Exception:
        return "?"


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--workdir", required=True, help="NxDI compiled_model_path")
        sp.add_argument("--tp", type=int, default=4)
        sp.add_argument("--height", type=int, default=1024)
        sp.add_argument("--width", type=int, default=1024)

    c = sub.add_parser("compile"); common(c)
    c.add_argument("--scratch", default="/tmp/nxd_model/nxdi_flux")
    c.add_argument("--force", action="store_true")
    g = sub.add_parser("generate"); common(g)
    from benchmark.models import MATRIX
    m = MATRIX["flux_1_dev"]
    g.add_argument("--steps", type=int, default=m.steps)
    g.add_argument("--guidance", type=float, default=m.guidance_scale)
    g.add_argument("--seed", type=int, default=m.seed)
    g.add_argument("--prompt", default=m.prompt)
    g.add_argument("--out", default=None)
    r = sub.add_parser("record")
    r.add_argument("--compile", required=True); r.add_argument("--cold", required=True)
    r.add_argument("--warm", required=True)
    r.add_argument("--cold-wall", type=float, default=None,
                   help="process wall of the cold generate measured by the driver")
    r.add_argument("--warm-wall", type=float, default=None)
    r.add_argument("--tp", type=int, default=4)
    a = p.parse_args()
    return {"compile": cmd_compile, "generate": cmd_generate, "record": cmd_record}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
