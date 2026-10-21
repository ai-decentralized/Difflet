import importlib
import json

import pytest

# difflet.cli.__init__ re-exports main() (the function), shadowing the
# submodule attribute — import the module explicitly.
cli_main = importlib.import_module("difflet.cli.main")


def _argv(extra, model="Wan-AI/Wan2.2-T2V-A14B-Diffusers"):
    return ["generate", "--model-id", model, *extra]


def test_parser_accepts_dp_flags_on_generate_and_compile():
    p = cli_main._build_parser()
    args = p.parse_args(_argv(["--dp", "4", "--mode", "throughput",
                               "--dp-schedule", "least_loaded",
                               "--requests", "r.jsonl"]))
    assert args.dp == 4 and args.mode == "throughput"
    args = p.parse_args(["compile", "--model-id", "black-forest-labs/FLUX.1-dev",
                         "--dp", "4", "--mode", "throughput"])
    assert args.dp == 4


def test_single_prompt_still_parses_without_batch_flags():
    p = cli_main._build_parser()
    args = p.parse_args(_argv(["--prompt", "cat", "--output", "c.mp4"]))
    assert args.dp is None and args.requests is None and args.requests_dir is None


def test_generate_requires_prompt_or_requests():
    p = cli_main._build_parser()
    args = p.parse_args(_argv([]))
    args.command = "generate"
    with pytest.raises(SystemExit):
        cli_main._validate_dp(args)


def test_teacache_rejected_in_batch_mode(tmp_path):
    p = cli_main._build_parser()
    req = tmp_path / "r.jsonl"
    req.write_text(json.dumps({"prompt": "x", "output": "x.png"}) + "\n")
    args = p.parse_args(_argv(["--requests", str(req), "--teacache-cadence", "2"]))
    args.command = "generate"
    with pytest.raises(SystemExit):
        cli_main._validate_dp(args)


def test_core_budget_validation(tmp_path):
    p = cli_main._build_parser()
    args = p.parse_args(_argv(["--dp", "4", "--tp-degree", "4",
                               "--prompt", "x", "--output", "x.mp4",
                               "--total-cores", "8"]))
    args.command = "generate"
    with pytest.raises(SystemExit):
        cli_main._validate_dp(args)   # 4*4 = 16 > 8


def test_mode_resolution_applied_to_args(monkeypatch, tmp_path):
    calls = {}

    def fake_router(args, requests, *, replica_cores, worker_argv_prefix=None):
        calls["dp"] = args.dp
        calls["cfg"] = args.cfg_parallel
        calls["cp"] = args.cp_degree
        calls["replica_cores"] = replica_cores
        calls["n"] = len(requests)
        return 0

    monkeypatch.setattr("difflet.cli.dp.router.run_router", fake_router)
    req = tmp_path / "r.jsonl"
    req.write_text(json.dumps({"prompt": "x", "output": str(tmp_path / "x.mp4")}) + "\n")
    with pytest.raises(SystemExit) as exc:
        cli_main.main(_argv(["--mode", "mixed", "--requests", str(req),
                             "--tp-degree", "4"]))
    assert exc.value.code == 0
    # Wan mixed: dp=2 cfg=2 cp=1 tp=4 -> replica_cores = 2*1*4 = 8
    assert calls == {"dp": 2, "cfg": True, "cp": 1, "replica_cores": 8, "n": 1}


def test_parallel_echo_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("difflet.cli.dp.router.run_router",
                        lambda *a, **k: 0)
    req = tmp_path / "r.jsonl"
    req.write_text(json.dumps({"prompt": "x", "output": str(tmp_path / "x.mp4")}) + "\n")
    with pytest.raises(SystemExit):
        cli_main.main(_argv(["--mode", "mixed", "--requests", str(req),
                             "--tp-degree", "4"]))
    assert "parallel: dp=2 cfg=2 cp=1 (mode=mixed)" in capsys.readouterr().out


def test_stage_parser_accepts_new_flags():
    from difflet.cli.stage import _build_stage_parser
    args, _ = _build_stage_parser().parse_known_args([
        "--orchestrator", "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "--stage", "transformer",
        "--requests-dir", "/tmp/reqs", "--worker-index", "1",
        "--dp-schedule", "least_loaded", "--keep-work-dir",
    ])
    assert args.requests_dir == "/tmp/reqs" and args.worker_index == 1
    assert args.dp_schedule == "least_loaded" and args.keep_work_dir


# ------------------------------------------------- CUDA: GPUs, not NeuronCores


def _fake_cuda(monkeypatch, *, available, count=0):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


@pytest.mark.parametrize(
    "extra,gpus",
    [
        ([], 1),
        (["--cfg-parallel"], 2),
        (["--cfg-parallel", "--cp-degree", "2"], 4),
        (["--tp-degree", "2"], 2),
    ],
)
def test_cuda_replica_is_cfg_times_cp_times_tp_gpus(monkeypatch, extra, gpus):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    args = cli_main._build_parser().parse_args(_argv(extra))
    # No registry default tp on CUDA: one GPU per replica unless asked otherwise.
    assert cli_main._replica_cores(args) == gpus


def test_cuda_total_comes_from_device_count(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "16")  # a Neuron variable means nothing here
    _fake_cuda(monkeypatch, available=True, count=2)
    args = cli_main._build_parser().parse_args(_argv([]))
    assert cli_main._resolve_total_cores(args) == (2, "torch.cuda.device_count()")
    args = cli_main._build_parser().parse_args(_argv(["--total-cores", "8"]))
    assert cli_main._resolve_total_cores(args) == (8, "--total-cores")


def test_cuda_total_is_undetected_without_a_device(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    _fake_cuda(monkeypatch, available=False, count=0)
    args = cli_main._build_parser().parse_args(_argv([]))
    assert cli_main._resolve_total_cores(args)[0] is None
    args.dp = 8
    cli_main._validate_capacity(args)  # unknown budget: not checked


def test_cuda_capacity_rejects_more_gpus_than_the_host_has(monkeypatch, capsys):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    _fake_cuda(monkeypatch, available=True, count=2)
    args = cli_main._build_parser().parse_args(
        _argv(["--dp", "2", "--cfg-parallel", "--prompt", "x", "--output", "x.mp4"])
    )
    args.command = "generate"
    with pytest.raises(SystemExit):
        cli_main._validate_dp(args)  # 2 x 2 = 4 GPUs > 2
    err = capsys.readouterr().err
    assert "needs 4 GPUs" in err and "2 are available" in err and "tp=1" in err


def test_cuda_capacity_accepts_dp2_on_two_gpus(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    _fake_cuda(monkeypatch, available=True, count=2)
    args = cli_main._build_parser().parse_args(
        _argv(["--dp", "2", "--prompt", "x", "--output", "x.mp4"], model="Qwen/Qwen-Image")
    )
    args.command = "generate"
    cli_main._validate_dp(args)  # 2 x 1 = 2 GPUs: fits
