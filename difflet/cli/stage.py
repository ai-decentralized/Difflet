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

from difflet.cli.main import EXEC_MODE_CHOICES  # noqa: F401 - re-exported

def orchestrator_ref(model_id: str) -> str | None:
    """The staged orchestrator ("pkg.module:Class") the registry records for ``model_id``."""
    from difflet.registry import registered_models

    for entry in registered_models():
        if model_id in entry.cli_model_ids:
            return entry.cli_orchestrator
    return None


def _load_orchestrator_class(name: str):
    """``name`` is a CLI model id (its registry entry's cli_orchestrator) or "pkg.module:Class"
    (an orchestrator outside the registry, such as tests/unit/backends/_neuron_toy.py's)."""
    ref = name if ":" in name else orchestrator_ref(name)
    if ref is None:
        sys.exit(f"[difflet.cli.stage] unknown orchestrator: {name!r}")
    module_path, _, cls_name = ref.partition(":")
    if not module_path or not cls_name:
        sys.exit(f"[difflet.cli.stage] invalid orchestrator reference: {ref!r} "
                 "(expected 'pkg.module:Class')")
    return getattr(importlib.import_module(module_path), cls_name)


def _build_stage_parser() -> argparse.ArgumentParser:
    from difflet.cli.main import add_generate_options

    # allow_abbrev=False and parse_args (main): an option the stage does not declare is an
    # error, never read as another ("--dp" as "--dp-schedule") or silently dropped.
    p = argparse.ArgumentParser(prog="difflet.cli.stage", add_help=False, allow_abbrev=False)
    p.add_argument("--orchestrator", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--stage-mode", default="generate", choices=["compile", "generate"])
    p.add_argument("--model-id", dest="model_id", default=None)
    # The generate options, declared once (difflet.cli.main.add_generate_options).
    add_generate_options(p)
    p.set_defaults(cp_degree=1)  # the driver resolved CP; a stage launched without it runs CP 1
    # Stage-only: values the driver resolves, never typed by a user.
    p.add_argument("--model-path", dest="model_path", default=None)
    p.add_argument("--compiled-dir", default=None)
    p.add_argument("--vae-tp-degree", type=int, default=None)
    p.add_argument("--adaln-precompute", dest="adaln_precompute", action="store_true")
    # The decode the driver resolved (HunyuanVideo on a torchrun backend).
    p.add_argument("--vae-placement", dest="vae_placement", default=None,
                   choices=["generate", "launch", "host"])
    return p


def _export_backend_selection(args: argparse.Namespace) -> None:
    if args.backend is not None:
        os.environ["DIFFLET_BACKEND"] = args.backend
    if args.exec_mode is not None:
        os.environ["DIFFLET_EXEC_MODE"] = args.exec_mode


def main(argv: list[str] | None = None) -> int:
    parser = _build_stage_parser()
    args = parser.parse_args(argv)
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
