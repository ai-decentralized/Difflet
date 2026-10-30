"""HunyuanVideo 1.5's CLI on the neuron backend (CPU, stages stubbed): the stage chain and its layout,
the refusals, and what the encode and transformer stages write."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import pytest
import torch

import difflet.cli.orchestrators.hunyuan_video_15 as hv15


def _args(**over):
    base = dict(model_id=hv15._HF_MODEL_ID, tp_degree=None, cp_degree=None, cp_mode=None, cfg_parallel=False, height=None,
                width=None, num_frames=None, steps=None, seed=42, prompt="a cat", output="/tmp/o.mp4",
                guidance_scale=None, cache_dir=None, revision=None, exec_mode=None, work_dir=None,
                keep_work_dir=False, sp_enabled=False, requests_dir=None, requests=None, dp=None,
                teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
                teacache_calibration=None, stage_mode="generate")
    base.update(over)
    return argparse.Namespace(**base)


class _Calls(list):
    env_seen: list


@pytest.fixture
def launches(monkeypatch):
    calls = _Calls()
    import difflet.cli.orchestrators.torchrun as torchrun_helpers

    monkeypatch.delenv("DIFFLET_NKI_TRACE_CACHE", raising=False)  # the driver sets it: restore after each test
    monkeypatch.setattr(hv15, "torchrun_backend", lambda: "neuron")
    monkeypatch.setattr(torchrun_helpers, "torchrun_backend", lambda: "neuron")  # the refusal checks it too
    def run_stage(model, stage, num_cores, virtual_core_size, cli_args):
        calls.append((stage, num_cores, list(cli_args)))
        env_seen.append(os.environ.get("DIFFLET_NKI_TRACE_CACHE"))

    env_seen = []
    calls.env_seen = env_seen  # what each launch saw
    monkeypatch.setattr(hv15.runner, "run_stage", run_stage)
    return calls


def _flag(cli, name):
    return cli[cli.index(name) + 1] if name in cli else None


def test_default_layout_is_tp2_x_cfg2(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"))).generate()
    assert launches.env_seen == ["1", "1", "1"]  # every stage keeps the attention kernel's trace
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ  # restored after the launches
    assert [(s, n) for s, n, _ in launches] == [("encode", 1), ("transformer", 4), ("vae", 1)]
    cli = launches[1][2]
    assert _flag(cli, "--tp-degree") == "2" and "--cfg-parallel" in cli
    assert _flag(cli, "--steps") == "50" and _flag(cli, "--num-frames") == "121"
    assert not (tmp_path / "w").exists()


def test_explicit_tp4_runs_cfg_sequentially(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), tp_degree=4)).generate()
    assert launches[1][:2] == ("transformer", 4) and "--cfg-parallel" not in launches[1][2]


def test_compile_skips_the_encoder(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"))).compile()
    assert [s for s, _, _ in launches] == ["transformer", "vae"]


def test_sp_and_cp_mode_reach_the_transformer_stage(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), tp_degree=2, sp_enabled=True)).generate()
    assert "--sp" in launches[1][2]
    launches.clear()
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), tp_degree=1, cp_degree=4,
                                          cp_mode="ring")).generate()
    assert _flag(launches[1][2], "--cp-mode") == "ring"


def test_aot_backend_is_not_implemented(monkeypatch):
    monkeypatch.setattr(hv15, "torchrun_backend", lambda: None)
    with pytest.raises(NotImplementedError, match="neuron"):
        hv15.HunyuanVideo15Orchestrator(_args()).generate()


def test_encode_stage_uses_diffusers_encode_prompt(monkeypatch, tmp_path):
    import diffusers

    prompts = []

    class FakePipe:
        def encode_prompt(self, prompt, device, dtype):
            prompts.append(prompt)
            return (torch.ones(1, 4, 8), torch.ones(1, 4), torch.zeros(1, 2, 3), torch.zeros(1, 2))

    monkeypatch.setattr(diffusers.HunyuanVideo15Pipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: FakePipe()))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    args = _args(work_dir=str(tmp_path))
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("encode", args)
    saved = torch.load(tmp_path / "embeds.pt")
    assert prompts == ["a cat", ""]
    assert set(saved) == {"prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2",
                          "negative_prompt_embeds", "negative_prompt_embeds_mask", "negative_prompt_embeds_2",
                          "negative_prompt_embeds_mask_2"}


def test_transformer_stage_saves_latents(monkeypatch, tmp_path):
    import difflet.backends.neuron.hunyuan_video15.pipeline as P
    import difflet.backends.registry as registry

    monkeypatch.setattr(registry, "get_backend", lambda name: SimpleNamespace(prepare_runtime=lambda p: None))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    seen = {}

    class FakePipeline:
        def __init__(self, **kw):
            seen["init"] = kw
            self.dit = SimpleNamespace(phase_seconds={})
            self.guidance_scale = 6.0

        def load(self, skip_warmup=False):
            seen["skip_warmup"] = skip_warmup

        def __call__(self, num_inference_steps, generator, callback, **embeds):
            seen["steps"] = num_inference_steps
            callback(0, None, None)
            return torch.full((1, 4, 2, 2, 3), 7.0)

    monkeypatch.setattr(P, "TorchNeuronHunyuanVideo15Pipeline", FakePipeline)
    torch.save({"prompt_embeds": torch.zeros(1)}, tmp_path / "embeds.pt")
    args = _args(work_dir=str(tmp_path), tp_degree=2, cfg_parallel=True, height=480, width=848,
                 num_frames=121, steps=50, seed=3)
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("transformer", args)
    assert seen["skip_warmup"] is True and seen["steps"] == 50
    assert seen["init"]["parallel"].cfg_parallel_enabled and seen["init"]["parallel"].tp_degree == 2
    assert torch.equal(torch.load(tmp_path / "latents.pt"), torch.full((1, 4, 2, 2, 3), 7.0))


def test_probe_free_teacache_reaches_the_transformer_stage(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), teacache_cadence=2)).generate()
    assert _flag(launches[1][2], "--teacache-cadence") == "2"


def test_adaptive_teacache_flags_reach_the_transformer_stage(launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_args(work_dir=str(tmp_path / "w"), teacache_speedup=1.5,
                                          teacache_calibration="/c.json")).generate()
    cli = launches[1][2]
    assert _flag(cli, "--teacache-speedup") == "1.5" and _flag(cli, "--teacache-calibration") == "/c.json"


def _requests_dir(tmp_path):
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    rdir = tmp_path / "requests"
    rdir.mkdir()
    write_manifest([RequestSpec(index=0, prompt="a dog", output=str(tmp_path / "o0.mp4"), seed=5,
                                negative_prompt="blurry", assigned_worker=0),
                    RequestSpec(index=1, prompt="a boat", output=str(tmp_path / "o1.mp4"), seed=9,
                                assigned_worker=0)], rdir)
    return rdir  # a Path, as the stage parser gives


def test_batch_encode_claims_each_request(monkeypatch, tmp_path):
    import diffusers

    prompts = []

    class FakePipe:
        def encode_prompt(self, prompt, device, dtype):
            prompts.append(prompt)
            return (torch.ones(1, 4, 8), torch.ones(1, 4), torch.zeros(1, 2, 3), torch.zeros(1, 2))

    monkeypatch.setattr(diffusers.HunyuanVideo15Pipeline, "from_pretrained",
                        classmethod(lambda cls, *a, **k: FakePipe()))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    args = _args(work_dir=str(tmp_path), requests_dir=_requests_dir(tmp_path), worker_index=0,
                 dp_schedule="round_robin")
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("encode", args)
    assert prompts == ["a dog", "blurry", "a boat", ""]
    assert (tmp_path / "embeds_req0000.pt").exists() and (tmp_path / "embeds_req0001.pt").exists()


def test_batch_transformer_replays_the_claims_with_each_seed(monkeypatch, tmp_path):
    import difflet.backends.neuron.hunyuan_video15.pipeline as P
    import difflet.backends.registry as registry
    from difflet.cli.dp import claims

    monkeypatch.setattr(registry, "get_backend", lambda name: SimpleNamespace(prepare_runtime=lambda p: None))
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    seeds = []

    class FakePipeline:
        def __init__(self, **kw):
            self.dit = SimpleNamespace(phase_seconds={})
            self.guidance_scale = 6.0

        def load(self, skip_warmup=False):
            pass

        def __call__(self, num_inference_steps, generator, callback, **embeds):
            seeds.append(generator.initial_seed())
            callback(0, None, None)
            return torch.full((1, 4, 2, 2, 3), float(generator.initial_seed()))

    monkeypatch.setattr(P, "TorchNeuronHunyuanVideo15Pipeline", FakePipeline)
    rdir = _requests_dir(tmp_path)
    for index in (0, 1):
        claims.claim_next(rdir, 0, "round_robin")
        torch.save({"prompt_embeds": torch.zeros(1)}, tmp_path / f"embeds_req000{index}.pt")
    args = _args(work_dir=str(tmp_path), requests_dir=rdir, worker_index=0, dp_schedule="round_robin",
                 tp_degree=2, height=480, width=848, num_frames=121, steps=50, seed=42)
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("transformer", args)
    assert seeds == [5, 9]
    assert torch.load(tmp_path / "latents_req0001.pt").flatten()[0].item() == 9.0
