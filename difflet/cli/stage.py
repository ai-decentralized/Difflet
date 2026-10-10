"""Internal subprocess dispatcher — not a user-facing entry point.

Called by runner.run_stage() as:
    python -m difflet.cli.stage --orchestrator <hf-id> --stage <stage> [forwarded args]
or, when DIFFLET_BACKEND names a non-AoT torchrun-MPMD backend (neuron), once per
NeuronCore:
    python -m torch.distributed.run --standalone --nproc-per-node <N> \
        -m difflet.cli.stage --orchestrator <hf-id | pkg.module:Class> --stage <stage> ...
An exception leaves the process with a non-zero exit code; under torchrun that is
what makes the launcher stop the other ranks, which may be blocked in a collective.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from difflet.backends.registry import available_backends
from difflet.pipeline.parallel_config import CP_MODES

# Execution modes of non-AoT backends: the same values as
# difflet.backends.neuron.compile.EXEC_MODES, repeated so the CLI parsers never
# import a backend module (or torch). test_neuron_launch_c10.py pins the two together.
EXEC_MODE_CHOICES: tuple[str, ...] = ("eager", "compile")

_ORCHESTRATOR_MAP: dict[str, str] = {
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "difflet.cli.orchestrators.wan.WanOrchestrator",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "difflet.cli.orchestrators.wan.WanOrchestrator",
    "hunyuanvideo-community/HunyuanVideo": (
        "difflet.cli.orchestrators.hunyuan_video.HunyuanVideoOrchestrator"
    ),
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": (
        "difflet.cli.orchestrators.hunyuan_video_15.HunyuanVideo15Orchestrator"
    ),
    "Qwen/Qwen-Image": "difflet.cli.orchestrators.qwen_image.QwenImageOrchestrator",
    "MiniMaxAI/MiniMax-H3": "difflet.cli.orchestrators.minimax_h3.MiniMaxH3Orchestrator",
}


def _load_orchestrator_class(name: str):
    if ":" in name:
        # "pkg.module:Class" names an orchestrator outside the model map, such as
        # the test-only toy application (tests/unit/backends/_neuron_toy.py).
        module_path, _, cls_name = name.partition(":")
        if not module_path or not cls_name:
            sys.exit(
                f"[difflet.cli.stage] invalid orchestrator reference: {name!r} "
                "(expected 'pkg.module:Class')"
            )
        return getattr(importlib.import_module(module_path), cls_name)
    if name not in _ORCHESTRATOR_MAP:
        sys.exit(f"[difflet.cli.stage] unknown orchestrator: {name!r}")
    cls_path = _ORCHESTRATOR_MAP[name]
    module_path, cls_name = cls_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _build_stage_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="difflet.cli.stage", add_help=False)
    p.add_argument("--orchestrator", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--stage-mode", default="generate", choices=["compile", "generate"])
    p.add_argument("--model-id", dest="model_id", default=None)
    p.add_argument("--model-path", dest="model_path", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--tp-degree", type=int, default=None)
    p.add_argument("--cp-degree", type=int, default=1)
    p.add_argument("--cp-mode", choices=list(CP_MODES), default="gather_kv")
    p.add_argument("--cfg-parallel", dest="cfg_parallel", action="store_true")
    p.add_argument("--sp", dest="sp_enabled", action="store_true")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--shapes", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--compiled-dir", default=None)
    p.add_argument("--vae-tp-degree", type=int, default=None)
    p.add_argument("--adaln-precompute", dest="adaln_precompute", action="store_true")
    p.add_argument("--teacache-cadence", type=int, default=None)
    p.add_argument("--teacache-online-delta", type=float, default=None)
    p.add_argument("--teacache-speedup", type=float, default=None)
    p.add_argument("--teacache-calibration", default=None)
    # DP worker-mode flags (parse_known_args silently drops what isn't declared
    # here — the --sp bug class; keep in sync with main._add_generate_flags).
    p.add_argument("--requests-dir", default=None)
    p.add_argument("--worker-index", type=int, default=None)
    p.add_argument("--dp-schedule", default="round_robin")
    p.add_argument("--keep-work-dir", action="store_true")
    # Backend selection. main() exports both to the environment before the
    # orchestrator loads, so difflet.ops, DiffletPipeline and the application
    # resolve the same backend and execution mode in every rank.
    p.add_argument("--backend", choices=available_backends(), default=None)
    p.add_argument("--exec-mode", dest="exec_mode", choices=EXEC_MODE_CHOICES, default=None)
    return p


def _export_backend_selection(args: argparse.Namespace) -> None:
    if args.backend is not None:
        os.environ["DIFFLET_BACKEND"] = args.backend
    if args.exec_mode is not None:
        os.environ["DIFFLET_EXEC_MODE"] = args.exec_mode


def main(argv: list[str] | None = None) -> int:
    parser = _build_stage_parser()
    args, _ = parser.parse_known_args(argv)
    _export_backend_selection(args)
    cls = _load_orchestrator_class(args.orchestrator)
    orchestrator = cls(args)
    if args.stage_mode == "generate":
        # Overlap this stage's ~6.7s one-time NeuronCore bring-up with its load.
        # Gated to generate: compile runs on the host compiler and must not spin
        # up the device. prewarm_neuron_runtime returns early for non-AoT
        # backends, whose ranks must bind their core before the runtime starts.
        from difflet.cli.prewarm import prewarm_neuron_runtime
        num_cores = int(os.environ.get("NEURON_RT_NUM_CORES") or 0) or (
            (args.tp_degree or 1) * (args.cp_degree or 1)
        )
        prewarm_neuron_runtime(num_cores)
    orchestrator._run_stage_internal(args.stage, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
