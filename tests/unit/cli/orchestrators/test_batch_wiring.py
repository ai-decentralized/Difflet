"""CPU-testable batch wiring: _shared_cli_args must forward the DP worker flags
(stage parser silently drops what isn't forwarded — the --sp bug class), and
in-process orchestrators must loop claimed requests."""
import argparse


def _args(**kw):
    defaults = dict(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers", tp_degree=None, cp_degree=None,
        cp_mode="gather_kv", cfg_parallel=False, sp_enabled=False, height=None,
        width=None, num_frames=None, steps=None, guidance_scale=None, seed=42,
        cache_dir=None, prompt=None, output=None, requests_dir="/tmp/reqs",
        worker_index=1, dp_schedule="least_loaded", keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
        teacache_calibration=None, work_dir=None, revision=None, force=False,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_wan_shared_args_forward_dp_worker_flags():
    from difflet.cli.orchestrators.wan import WanOrchestrator
    parts = WanOrchestrator(_args())._shared_cli_args(stage_mode="generate")
    text = " ".join(parts)
    assert "--requests-dir /tmp/reqs" in text
    assert "--worker-index 1" in text
    assert "--dp-schedule least_loaded" in text


def test_hunyuan_shared_args_forward_dp_worker_flags():
    from difflet.cli.orchestrators.hunyuan_video import HunyuanVideoOrchestrator
    parts = HunyuanVideoOrchestrator(_args(
        model_id="hunyuanvideo-community/HunyuanVideo"))._shared_cli_args(
        stage_mode="generate")
    text = " ".join(parts)
    assert "--requests-dir /tmp/reqs" in text and "--worker-index 1" in text


def test_qwen_shared_args_forward_dp_worker_flags():
    from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
    parts = QwenImageOrchestrator(_args(model_id="Qwen/Qwen-Image"))._shared_cli_args(
        stage_mode="generate")
    text = " ".join(parts)
    assert "--requests-dir /tmp/reqs" in text and "--worker-index 1" in text


def test_flux_generate_loops_requests(monkeypatch, tmp_path):
    """Flux batch mode: pipeline loads once, pipe() called once per claimed request."""
    from difflet.cli.dp.claims import summarize
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest
    from difflet.cli.orchestrators import flux as flux_mod

    write_manifest(
        [RequestSpec(index=i, prompt=f"p{i}", output=str(tmp_path / f"o{i}.png"), seed=i)
         for i in range(3)],
        tmp_path / "requests",
    )
    calls = []

    class FakeImage:
        def save(self, path):
            calls.append(("save", path))

    class FakePipe:
        def __call__(self, **kw):
            calls.append(("pipe", kw["prompt"]))
            return type("Out", (), {"images": [FakeImage()]})()

    loads = []
    monkeypatch.setattr(
        flux_mod.FluxOrchestrator, "_load_pipeline",
        lambda self: (loads.append(1), FakePipe())[1],
    )
    args = _args(model_id="black-forest-labs/FLUX.1-dev",
                 requests_dir=str(tmp_path / "requests"), worker_index=0,
                 dp_schedule="least_loaded")
    flux_mod.FluxOrchestrator(args).generate()
    assert loads == [1]                                  # loaded once
    assert [c for c in calls if c[0] == "pipe"] == [("pipe", "p0"), ("pipe", "p1"),
                                                    ("pipe", "p2")]
    assert summarize(tmp_path / "requests").done == [0, 1, 2]


def test_ltx2_generate_loops_requests(monkeypatch, tmp_path):
    import torch
    from difflet.cli.dp.claims import summarize
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest
    from difflet.cli.orchestrators import ltx_2 as ltx_mod

    write_manifest(
        [RequestSpec(index=i, prompt=f"p{i}", output=str(tmp_path / f"o{i}.mp4"), seed=i)
         for i in range(2)],
        tmp_path / "requests",
    )
    calls = []

    class FakePipe:
        def __call__(self, **kw):
            calls.append(kw["prompt"])
            return type("Out", (), {"frames": torch.zeros(1, 3, 2, 8, 8)})()

    monkeypatch.setattr(ltx_mod.LTX2Orchestrator, "_load_pipeline",
                        lambda self: FakePipe())
    args = _args(model_id="Lightricks/LTX-2",
                 requests_dir=str(tmp_path / "requests"), worker_index=0,
                 dp_schedule="least_loaded")
    ltx_mod.LTX2Orchestrator(args).generate()
    assert calls == ["p0", "p1"]
    # mp4 export with a .pt tensor fallback on codec failure
    for i in range(2):
        assert (tmp_path / f"o{i}.mp4").exists() or (tmp_path / f"o{i}.pt").exists()
    assert summarize(tmp_path / "requests").done == [0, 1]


def test_wan_shared_args_legacy_unchanged():
    from difflet.cli.orchestrators.wan import WanOrchestrator
    parts = WanOrchestrator(_args(requests_dir=None, worker_index=None,
                                  prompt="cat", output="c.mp4"))._shared_cli_args(
        stage_mode="generate")
    text = " ".join(parts)
    assert "--requests-dir" not in text and "--worker-index" not in text
    assert "--prompt cat" in text
