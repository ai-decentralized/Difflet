"""The DP router forwards TeaCache flags to its workers (a worker that lost them would silently run
without TeaCache)."""

from __future__ import annotations

import argparse

from difflet.cli.dp.router import worker_cli_args


def test_teacache_flags_reach_the_workers():
    args = argparse.Namespace(model_id="m", tp_degree=2, cp_degree=None, cp_mode=None, height=None, width=None,
                              num_frames=None, steps=None, guidance_scale=None, seed=None, cache_dir=None,
                              teacache_cadence=2, teacache_online_delta=None, teacache_speedup=1.5,
                              teacache_calibration="/c.json")
    argv = worker_cli_args(args)
    assert argv[argv.index("--teacache-cadence") + 1] == "2"
    assert argv[argv.index("--teacache-speedup") + 1] == "1.5"
    assert argv[argv.index("--teacache-calibration") + 1] == "/c.json"
    assert "--teacache-online-delta" not in argv
