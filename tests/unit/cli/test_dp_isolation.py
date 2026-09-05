"""DP isolation invariants (spec §Testing 2): the dp axis must never reach a
worker's compiled graph, and dp>1 must not perturb the compile-cache key."""
import re
from pathlib import Path

import difflet
from difflet.pipeline.parallel_config import DiffletParallelConfig

REPO = Path(difflet.__file__).resolve().parent


def test_worker_cli_args_never_forward_dp_or_mode():
    import argparse

    from difflet.cli.dp.router import worker_cli_args

    args = argparse.Namespace(
        model_id="m", tp_degree=4, cp_degree=2, cp_mode="ring", cfg_parallel=True,
        sp_enabled=False, height=None, width=None, num_frames=None, steps=None,
        guidance_scale=None, seed=42, cache_dir=None, revision=None,
        keep_work_dir=False, dp=4, mode="throughput",
    )
    argv = worker_cli_args(args)
    assert "--dp" not in argv and "--mode" not in argv


def test_dp1_cache_key_identical_to_no_dp():
    with_dp = DiffletParallelConfig(tp_degree=4, dp_degree=1).to_cache_dict()
    assert "dp_degree" not in with_dp
    assert with_dp == DiffletParallelConfig(tp_degree=4).to_cache_dict()


def test_no_cli_code_constructs_dp_parallel_config():
    """Workers must always build dp_degree=1 configs (the dataclass default).

    Matches the *binding* forms -- ``dp_degree=`` as a keyword argument or an
    assignment -- rather than any mention of the name. Reading
    ``parallel.dp_degree`` or emitting it as a JSON key (cli/plan.py does both,
    to report a configuration it was handed) leaves the invariant intact; only
    setting it would put the dp axis into a worker's compiled graph.
    """
    binding = re.compile(r"(?<![.\w])dp_degree\s*=")
    offenders = []
    for path in sorted((REPO / "cli").rglob("*.py")):
        if binding.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], f"CLI code must never set dp_degree: {offenders}"


def test_mesh_dp1_builds_no_dp_group():
    from difflet.pipeline.parallel_mesh import MeshSpec

    spec = MeshSpec(dp=1, cfg=2, cp=1, tp=4)
    # dp axis groups are all singletons at dp=1 — nothing to communicate over.
    assert all(len(g) == 1 for g in spec.axis_groups("dp"))
