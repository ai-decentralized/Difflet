from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

import difflet.cli.orchestrators.hunyuan_video_15 as hv15


def _hunyuan_video_15_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        tp_degree=4, cp_degree=1,
        height=480, width=848, num_frames=121,
        cache_dir=None, force=False, revision=None,
        prompt="a cat", output="/tmp/out.mp4",
        steps=40, guidance_scale=3.5, seed=42,
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_compile_dispatches_to_run_aot(monkeypatch):
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator

    modes = []
    monkeypatch.setattr(HunyuanVideo15Orchestrator, "_run_aot", lambda self, mode: modes.append(mode))
    HunyuanVideo15Orchestrator(_hunyuan_video_15_args()).compile()
    assert modes == ["compile"]


def test_generate_dispatches_to_run_aot(monkeypatch):
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator

    modes = []
    monkeypatch.setattr(HunyuanVideo15Orchestrator, "_run_aot", lambda self, mode: modes.append(mode))
    HunyuanVideo15Orchestrator(_hunyuan_video_15_args()).generate()
    assert modes == ["generate"]


def test_download_calls_resolve_model_path(monkeypatch):
    from difflet.cli.orchestrators.hunyuan_video_15 import HunyuanVideo15Orchestrator

    calls = []
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path",
        lambda model_id, *, local_files_only, **kw: calls.append((model_id, local_files_only)),
    )
    HunyuanVideo15Orchestrator(_hunyuan_video_15_args()).download()
    assert len(calls) == 1
    model_id, local_files_only = calls[0]
    assert model_id == "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    assert local_files_only is False


# ---------------------------------------------------------------- the Trainium (AOT) path


def _driver_args(**over) -> argparse.Namespace:
    base = dict(model_id=hv15._HF_MODEL_ID, tp_degree=None, cp_degree=None, cp_mode=None, cfg_parallel=False,
                height=None, width=None, num_frames=None, steps=None, seed=42, prompt="a cat",
                output="/tmp/o.mp4", guidance_scale=None, cache_dir=None, revision=None, exec_mode=None,
                work_dir=None, keep_work_dir=False, sp_enabled=False, requests_dir=None, requests=None,
                dp=None, teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
                teacache_calibration=None)
    base.update(over)
    return argparse.Namespace(**base)


def _stage_args(tmp_path, **over) -> argparse.Namespace:
    """What difflet.cli.stage parses for an AOT stage (its defaults: cp 1, cp_mode gather_kv)."""
    base = dict(model_id=hv15._HF_MODEL_ID, stage_mode="generate", work_dir=str(tmp_path / "work"),
                cache_dir=str(tmp_path / "cache"), tp_degree=2, cp_degree=1, cp_mode="gather_kv",
                cfg_parallel=True, sp_enabled=False, height=32, width=32, num_frames=1, steps=2, seed=42,
                prompt="a cat", output=str(tmp_path / "o.pt"), guidance_scale=None, revision=None,
                exec_mode=None, requests_dir=None, worker_index=None, dp_schedule="round_robin",
                teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
                teacache_calibration=None)
    base.update(over)
    os.makedirs(base["work_dir"], exist_ok=True)
    return argparse.Namespace(**base)


@pytest.fixture
def aot_launches(monkeypatch):
    calls = []
    monkeypatch.setattr(hv15, "torchrun_backend", lambda: None)
    monkeypatch.setattr(hv15.runner, "run_stage",
                        lambda model, stage, num_cores, virtual_core_size, cli_args:
                        calls.append((stage, num_cores, virtual_core_size, list(cli_args))))
    return calls


def _flag(cli, name):
    return cli[cli.index(name) + 1] if name in cli else None


def test_aot_generate_runs_three_stages(aot_launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_driver_args(work_dir=str(tmp_path / "w"))).generate()
    assert [(s, n) for s, n, _, _ in aot_launches] == [("encode", 1), ("aot_transformer", 4), ("aot_vae", 1)]
    assert [v for s, _, v, _ in aot_launches if s.startswith("aot_")] == [2, 2]
    assert not (tmp_path / "w").exists()


def test_aot_compile_skips_the_encoder(aot_launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_driver_args(work_dir=str(tmp_path / "w"))).compile()
    assert [s for s, _, _, _ in aot_launches] == ["aot_transformer", "aot_vae"]


def test_aot_default_layout_is_tp2_cfg2(aot_launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_driver_args(work_dir=str(tmp_path / "w"))).generate()
    cli = aot_launches[1][3]
    assert _flag(cli, "--tp-degree") == "2" and "--cfg-parallel" in cli and _flag(cli, "--cp-degree") == "1"


def test_aot_layout_cores(aot_launches, tmp_path):
    hv15.HunyuanVideo15Orchestrator(_driver_args(work_dir=str(tmp_path / "w"), tp_degree=2, cp_degree=2,
                                                 cp_mode="ulysses")).generate()
    assert aot_launches[1][:2] == ("aot_transformer", 4) and "--cfg-parallel" not in aot_launches[1][3]


def test_aot_failure_keeps_the_work_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(hv15, "torchrun_backend", lambda: None)

    def run_stage(model, stage, **kw):
        if stage == "aot_transformer":
            raise RuntimeError("boom")

    monkeypatch.setattr(hv15.runner, "run_stage", run_stage)
    with pytest.raises(RuntimeError, match="boom"):
        hv15.HunyuanVideo15Orchestrator(_driver_args(work_dir=str(tmp_path / "w"))).generate()
    assert (tmp_path / "w").exists()


def test_aot_backend_dispatches_to_run_aot(monkeypatch):
    monkeypatch.setattr(hv15, "torchrun_backend", lambda: None)
    modes = []
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_run_aot", lambda self, mode: modes.append(mode))
    hv15.HunyuanVideo15Orchestrator(_driver_args()).generate()
    assert modes == ["generate"]


def test_native_branch_unchanged(monkeypatch, tmp_path):
    import difflet.cli.orchestrators.torchrun as torchrun_helpers

    calls = []
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.delenv("DIFFLET_NKI_TRACE_CACHE", raising=False)
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_run_aot",
                        lambda self, mode: pytest.fail("the neuron backend must not take the AOT path"))
    monkeypatch.setattr(hv15.runner, "run_stage",
                        lambda model, stage, num_cores, virtual_core_size, cli_args:
                        calls.append((stage, num_cores, virtual_core_size)))
    assert torchrun_helpers.torchrun_backend() == "neuron"
    hv15.HunyuanVideo15Orchestrator(_driver_args(work_dir=str(tmp_path / "w"))).generate()
    assert calls == [("encode", 1, None), ("transformer", 4, None), ("vae", 1, None)]


def test_aot_stage_dispatch_rejects_unknown_stage(tmp_path):
    args = _stage_args(tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_nope", args)


def test_aot_stage_cache_key_fields(tmp_path):
    def key(component, **over):
        args = _stage_args(tmp_path, **over)
        return hv15.HunyuanVideo15Orchestrator(args)._aot_compiled_dir(args, component)

    base = key("transformer")
    for over in (dict(cp_degree=2, cp_mode="ulysses", cfg_parallel=False),
                 dict(sp_enabled=True), dict(cfg_parallel=False), dict(tp_degree=4),
                 dict(height=64), dict(revision="abc")):
        assert key("transformer", **over) != base, over
    assert (key("transformer", cp_degree=2, cp_mode="ulysses", cfg_parallel=False)
            != key("transformer", cp_degree=2, cp_mode="gather_kv", cfg_parallel=False))
    for over in (dict(teacache_cadence=2), dict(teacache_speedup=1.5),
                 dict(teacache_speedup=1.5, teacache_calibration="/c.json"),
                 dict(teacache_online_delta=0.1), dict(steps=7), dict(seed=3), dict(prompt="a dog")):
        assert key("transformer", **over) == base, over
    inputs = hv15.HunyuanVideo15Orchestrator(_stage_args(tmp_path))._aot_stage_cache_inputs(
        _stage_args(tmp_path), "transformer")
    assert {"model", "revision", "tp", "cp", "cp_mode", "sp", "cfg", "dtype", "height", "width", "num_frames",
            "toolchain"} <= set(inputs)
    assert (inputs["text_seq_len"], inputs["text_seq_len_2"], inputs["image_seq_len"]) == (1000, 256, 729)
    vae = key("vae")
    assert vae != base and key("vae", tp_degree=4, cfg_parallel=False, teacache_cadence=2) == vae
    assert key("vae", height=64) != vae
    vae_inputs = hv15.HunyuanVideo15Orchestrator(_stage_args(tmp_path))._aot_stage_cache_inputs(
        _stage_args(tmp_path), "vae")
    assert {"model", "dtype", "shape", "tile", "toolchain"} <= set(vae_inputs)


# ------------------------------------------------- the aot_transformer stage (application stubbed)


def _snapshot(tmp_path, *, vae: bool = False) -> str:
    from diffusers import FlowMatchEulerDiscreteScheduler

    root = tmp_path / "snap"
    if root.exists():
        return str(root)
    FlowMatchEulerDiscreteScheduler(shift=7.0).save_pretrained(root / "scheduler")
    (root / "guider").mkdir(parents=True)
    (root / "guider" / "guider_config.json").write_text(json.dumps(
        {"_class_name": "ClassifierFreeGuidance", "guidance_scale": 6.0, "guidance_rescale": 0.0,
         "use_original_formulation": False, "start": 0.0, "stop": 1.0, "enabled": True}))
    (root / "model_index.json").write_text(json.dumps(
        {"_class_name": "HunyuanVideo15Pipeline", "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]}))
    if vae:
        from diffusers import AutoencoderKLHunyuanVideo15

        torch.manual_seed(0)
        AutoencoderKLHunyuanVideo15(
            in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 16, 16, 16, 16),
            layers_per_block=1, spatial_compression_ratio=16, temporal_compression_ratio=4,
            downsample_match_channel=True, upsample_match_channel=True, scaling_factor=0.9,
        ).save_pretrained(root / "vae")
    return str(root)


class _FakeApp:
    def __init__(self, *, has_probe: bool = True):
        self.has_probe = has_probe
        self.events = []

    def compile(self, path, select=None):
        self.events.append(("compile", path, select))

    def has_compiled_artifacts(self, path, select=None):
        return self.has_probe


class _FakeDiT:
    """The ``HunyuanVideo15DiT`` protocol over a stub application: records each forward's batch."""

    model_config = {"out_channels": 4, "image_embed_dim": 16}
    image_tokens = 3

    def __init__(self, cfg: int, app: _FakeApp):
        self.cfg, self.app = cfg, app
        self.batches = []

    def load(self, skip_warmup=False):
        self.app.events.append(("load", skip_warmup))

    def begin_request(self):
        pass

    def make_teacache_probe(self):
        raise AssertionError("no adaptive TeaCache in this test")

    def __call__(self, *, hidden_states, **kw):
        self.batches.append(int(hidden_states.shape[0]))
        return torch.zeros((hidden_states.shape[0], 4) + tuple(hidden_states.shape[2:]))


@pytest.fixture
def aot_dit(monkeypatch, tmp_path):
    """build_aot_denoiser → the real HunyuanVideo15Denoiser over _FakeDiT; returns what it saw."""
    import difflet.models.hunyuan_video.aot15 as aot15
    from difflet.models.hunyuan_video.pipeline15 import HunyuanVideo15Denoiser

    root = _snapshot(tmp_path)
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: root)
    seen = {"builds": [], "app": _FakeApp()}

    def build(*, model_path, compiled_model_path, parallel, height, width, num_frames, guidance_scale=None,
              teacache_cadence=None, teacache_online_delta_alpha=None, teacache_speedup=None,
              teacache_calibration_path=None):
        seen["builds"].append(dict(compiled_model_path=compiled_model_path, parallel=parallel,
                                   guidance_scale=guidance_scale, teacache_speedup=teacache_speedup,
                                   teacache_cadence=teacache_cadence))
        dit = _FakeDiT(2 if parallel.cfg_parallel_enabled else 1, seen["app"])
        seen["dit"] = dit
        return HunyuanVideo15Denoiser(dit, model_path=model_path, height=height, width=width,
                                      num_frames=num_frames, dtype=torch.float32, guidance_scale=guidance_scale,
                                      teacache_cadence=teacache_cadence)

    monkeypatch.setattr(aot15, "build_aot_denoiser", build)
    return seen


def _compiled(args, component="transformer"):
    from difflet.cli.orchestrators.base import write_stage_manifest

    orch = hv15.HunyuanVideo15Orchestrator(args)
    path = orch._aot_compiled_dir(args, component)
    write_stage_manifest(path, orch._aot_stage_cache_inputs(args, component))
    return path


def _embeds(path):
    def text():
        return (torch.randn(1, 10, 32), torch.ones(1, 10), torch.randn(1, 6, 24), torch.ones(1, 6))

    names = ("prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2")
    embeds = dict(zip(names, text())) | {f"negative_{n}": t for n, t in zip(names, text())}
    torch.save(embeds, path)


def test_aot_compile_builds_the_application_and_writes_the_manifest(monkeypatch, tmp_path):
    import difflet.models.hunyuan_video.aot15 as aot15
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: "/snap")
    app, seen = _FakeApp(), {}

    def build(**kw):
        seen.update(kw)
        return app

    monkeypatch.setattr(aot15, "build_aot_application", build)
    args = _stage_args(tmp_path, stage_mode="compile", sp_enabled=True)
    orch = hv15.HunyuanVideo15Orchestrator(args)
    orch._run_stage_internal("aot_transformer", args)
    compiled = orch._aot_compiled_dir(args, "transformer")
    assert app.events == [("compile", str(compiled), None)]
    assert seen["teacache_fused"] is False and seen["parallel"].sp_enabled and seen["parallel"].tp_degree == 2
    assert seen["parallel"].cfg_parallel_enabled and (seen["height"], seen["num_frames"]) == (32, 1)
    assert has_valid_stage_manifest(compiled, orch._aot_stage_cache_inputs(args, "transformer"))


def test_aot_generate_needs_the_compiled_artifact(aot_dit, tmp_path):
    args = _stage_args(tmp_path)
    with pytest.raises(SystemExit, match="difflet compile"):
        hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)


def test_aot_transformer_writes_latents(aot_dit, tmp_path):
    args = _stage_args(tmp_path)
    compiled = _compiled(args)
    _embeds(tmp_path / "work" / "embeds.pt")
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)
    assert aot_dit["builds"][0]["compiled_model_path"] == str(compiled)
    assert aot_dit["app"].events == [("load", True)]
    assert aot_dit["dit"].batches == [2, 2]  # CFG parallel: both branches in one forward per step
    assert torch.load(tmp_path / "work" / "latents.pt").shape == (1, 4, 1, 2, 2)


def test_aot_lazy_probe_compile(aot_dit, tmp_path, monkeypatch):
    import difflet.models.hunyuan_video.aot15 as aot15

    app = _FakeApp(has_probe=False)
    kwargs = {}

    class _Denoiser:
        guidance_scale = 6.0

        def __init__(self, **kw):
            kwargs.update(kw)
            self.dit = _FakeDiT(2, app)

        def load(self, skip_warmup=False):
            app.events.append(("load", skip_warmup))

        def __call__(self, **kw):
            app.events.append(("denoise",))
            return torch.zeros(1, 4, 1, 2, 2)

    monkeypatch.setattr(aot15, "build_aot_denoiser", lambda **kw: _Denoiser(**kw))
    args = _stage_args(tmp_path, teacache_speedup=1.5, teacache_calibration="/c.json")
    compiled = _compiled(args)
    _embeds(tmp_path / "work" / "embeds.pt")
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)
    assert app.events == [("compile", str(compiled), ["teacache_probe"]), ("load", True), ("denoise",)]
    assert kwargs["teacache_speedup"] == 1.5 and kwargs["teacache_calibration_path"] == "/c.json"

    app.events.clear()
    app.has_probe = True  # compiled once: reused
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)
    assert app.events == [("load", True), ("denoise",)]


def _requests_dir(tmp_path, *requests):
    from difflet.cli.dp import claims
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    rdir = tmp_path / "requests"
    rdir.mkdir()
    specs = [RequestSpec(index=i, prompt=f"p{i}", output=str(tmp_path / f"o{i}.pt"), assigned_worker=0, **r)
             for i, r in enumerate(requests)]
    write_manifest(specs, rdir)
    for spec in specs:
        claims.claim_next(rdir, 0, "round_robin")
        _embeds(tmp_path / "work" / f"embeds_req{spec.index:04d}.pt")
    return str(rdir)  # a str, as difflet.cli.stage parses it


def test_aot_requests_load_once(aot_dit, tmp_path):
    os.makedirs(tmp_path / "work", exist_ok=True)
    rdir = _requests_dir(tmp_path, dict(seed=5), dict(seed=9, steps=3))
    args = _stage_args(tmp_path, requests_dir=rdir, worker_index=0)
    _compiled(args)
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)
    assert len(aot_dit["builds"]) == 1 and aot_dit["app"].events == [("load", True)]
    assert aot_dit["dit"].batches == [2] * (2 + 3)  # each request's own steps
    a, b = (torch.load(tmp_path / "work" / f"latents_req000{i}.pt") for i in (0, 1))
    assert a.shape == b.shape == (1, 4, 1, 2, 2)


def test_aot_request_seed_and_scale(aot_dit, tmp_path, monkeypatch):
    from difflet.models.hunyuan_video.pipeline15 import HunyuanVideo15Denoiser

    seen = []
    original = HunyuanVideo15Denoiser.__call__

    def spy(self, **kw):
        seen.append((kw["generator"].initial_seed(), kw["num_inference_steps"], self.guidance_scale))
        return original(self, **kw)

    monkeypatch.setattr(HunyuanVideo15Denoiser, "__call__", spy)
    os.makedirs(tmp_path / "work", exist_ok=True)
    rdir = _requests_dir(tmp_path, dict(seed=5), dict(seed=9, guidance_scale=3.0))
    args = _stage_args(tmp_path, requests_dir=rdir, worker_index=0, steps=4)
    _compiled(args)
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)
    assert seen == [(5, 4, 6.0), (9, 4, 3.0)]  # the guider's 6.0 unless the request sets its own


def test_aot_request_guidance_toggle_marks_failed(aot_dit, tmp_path):
    from difflet.cli.dp import claims

    os.makedirs(tmp_path / "work", exist_ok=True)
    rdir = _requests_dir(tmp_path, dict(guidance_scale=1.0), dict())
    args = _stage_args(tmp_path, requests_dir=rdir, worker_index=0)
    _compiled(args)
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)
    assert claims.is_failed(rdir, 0) and not claims.is_failed(rdir, 1)
    message = (tmp_path / "requests").glob("*0000*.failed")
    text = "".join(p.read_text() for p in message)
    assert "ValueError" in text and "guidance" in text and "classifier-free" in text
    assert not (tmp_path / "work" / "latents_req0000.pt").exists()
    assert (tmp_path / "work" / "latents_req0001.pt").exists()


def test_aot_guidance_toggle_raises_outside_batch_mode(aot_dit, tmp_path):
    args = _stage_args(tmp_path, guidance_scale=1.0)  # CFG-parallel layout, guidance off: refused
    _compiled(args)
    _embeds(tmp_path / "work" / "embeds.pt")
    with pytest.raises(ValueError):
        hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)


def test_aot_guidance_one_is_single_pass(aot_dit, tmp_path):
    args = _stage_args(tmp_path, cfg_parallel=False, guidance_scale=1.0, steps=3)
    _compiled(args)
    _embeds(tmp_path / "work" / "embeds.pt")
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_transformer", args)
    assert aot_dit["dit"].cfg == 1 and aot_dit["dit"].batches == [1, 1, 1]  # one forward per step
    assert (tmp_path / "work" / "latents.pt").exists()


# ------------------------------------------------------------------- the aot_vae stage


def test_aot_vae_divides_scaling_factor(monkeypatch, tmp_path):
    """The stage's decode on the Trainium decoder (its traced module stubbed by the eager decoder
    with the tiny VAE's weights) matches diffusers' ``vae.decode(latents / scaling_factor)``."""
    from diffusers import AutoencoderKLHunyuanVideo15

    from difflet.backends.trainium.hunyuan_video import vae15

    root = _snapshot(tmp_path, vae=True)
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: root)
    reference = AutoencoderKLHunyuanVideo15.from_pretrained(os.path.join(root, "vae")).to(torch.bfloat16).eval()
    loads = []

    def load(self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False):
        loads.append((compiled_model_path, skip_warmup))
        model = vae15.HunyuanVideo15VAEDecoderModel(self.config).to(self.dtype).eval()
        model.load_state_dict(reference.state_dict(), strict=False)
        self.models[0].model = model

    monkeypatch.setattr(vae15.NeuronHunyuanVideo15VAEDecoderApplication, "load", load)
    # One latent tile exactly (256 px = 16 latents): no padding, so the decode is diffusers' own.
    args = _stage_args(tmp_path, height=256, width=256, num_frames=1, output=str(tmp_path / "out" / "v.pt"))
    compiled = _compiled(args, "vae")
    latents = torch.randn(1, 4, 1, 16, 16, generator=torch.Generator().manual_seed(0))
    torch.save(latents, tmp_path / "work" / "latents.pt")
    hv15.HunyuanVideo15Orchestrator(args)._run_stage_internal("aot_vae", args)
    assert loads == [(str(compiled), True)]
    video = torch.load(tmp_path / "out" / "v.pt")
    with torch.no_grad():
        expected = reference.decode(latents.to(torch.bfloat16) / reference.config.scaling_factor,
                                    return_dict=False)[0].float()
    assert video.dtype == torch.float32 and video.shape == expected.shape == (1, 3, 1, 256, 256)
    assert torch.allclose(video, expected, atol=1e-2), (video - expected).abs().max()
    undivided = reference.decode(latents.to(torch.bfloat16), return_dict=False)[0].float()
    assert not torch.allclose(video, undivided, atol=1e-2)


def test_aot_vae_compile_writes_the_manifest(monkeypatch, tmp_path):
    from difflet.backends.trainium.hunyuan_video import vae15
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    root = _snapshot(tmp_path, vae=True)
    monkeypatch.setattr(hv15.HunyuanVideo15Orchestrator, "_root", lambda self, args: root)
    compiles = []
    monkeypatch.setattr(vae15.NeuronHunyuanVideo15VAEDecoderApplication, "compile",
                        lambda self, path, **kw: compiles.append((path, self.config.neuron_config.world_size)))
    args = _stage_args(tmp_path, stage_mode="compile", height=256, width=256)
    orch = hv15.HunyuanVideo15Orchestrator(args)
    orch._run_stage_internal("aot_vae", args)
    compiled = orch._aot_compiled_dir(args, "vae")
    assert compiles == [(str(compiled), 1)]
    assert has_valid_stage_manifest(compiled, orch._aot_stage_cache_inputs(args, "vae"))


def test_aot_write_output_mp4(monkeypatch, tmp_path):
    import diffusers.utils

    exported = []
    monkeypatch.setattr(diffusers.utils, "export_to_video",
                        lambda frames, path, fps: exported.append((len(frames), path, fps)))
    out = tmp_path / "sub" / "v.mp4"
    hv15._aot_write_output(torch.zeros(1, 3, 5, 32, 32), out)
    assert exported == [(5, str(out), 24)]
    hv15._aot_write_output(torch.ones(1, 3, 5, 32, 32), tmp_path / "v.pt")
    assert torch.equal(torch.load(tmp_path / "v.pt"), torch.ones(1, 3, 5, 32, 32))


# ------------------------------------------------------- AOT stages never import the native backend

_NO_NATIVE = textwrap.dedent("""
    import argparse, os, sys
    from types import SimpleNamespace
    import torch

    import difflet.cli.orchestrators.hunyuan_video_15 as hv15
    import difflet.models.hunyuan_video.aot15 as aot15
    import difflet.models.hunyuan_video.application as application
    from difflet.backends.trainium.hunyuan_video import vae15
    from difflet.cli.orchestrators.base import write_stage_manifest

    tmp = sys.argv[1]
    assert hv15.torchrun_backend() != "neuron"

    class App:
        def compile(self, path, select=None): pass
        def has_compiled_artifacts(self, path, select=None): return True
        def load(self, path, skip_warmup=False): pass

    class Denoiser:
        guidance_scale = None
        def __init__(self): self.dit = SimpleNamespace(app=App())
        def load(self, skip_warmup=False): pass
        def __call__(self, **kw): return torch.zeros(1, 4, 1, 2, 2)

    class Vae:
        def __init__(self, **kw): self.config = SimpleNamespace(scaling_factor=0.5)
        def compile(self, path): pass
        def load(self, path, skip_warmup=False): pass
        def decode(self, latents, return_dict=True): return (torch.zeros(1, 3, 1, 32, 32),)

    aot15.build_aot_application = lambda **kw: App()
    aot15.build_aot_denoiser = lambda **kw: Denoiser()
    application.create_hunyuan_video15_vae_decoder_config = lambda **kw: SimpleNamespace()
    vae15.NeuronHunyuanVideo15VAEDecoderApplication = Vae
    hv15.HunyuanVideo15Orchestrator._root = lambda self, args: tmp
    os.makedirs(os.path.join(tmp, "guider"))
    open(os.path.join(tmp, "guider", "guider_config.json"), "w").write(
        '{"_class_name": "ClassifierFreeGuidance", "guidance_scale": 6.0, "enabled": true}')
    args = argparse.Namespace(
        model_id=hv15._HF_MODEL_ID, stage_mode="compile", work_dir=tmp, cache_dir=tmp, tp_degree=1,
        cp_degree=1, cp_mode="gather_kv", cfg_parallel=False, sp_enabled=False, height=32, width=32,
        num_frames=1, steps=1, seed=0, prompt="p", output=os.path.join(tmp, "o.pt"), guidance_scale=1.0,
        revision=None, exec_mode=None, requests_dir=None, worker_index=None, dp_schedule="round_robin",
        teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None, teacache_calibration=None)
    torch.save({}, os.path.join(tmp, "embeds.pt"))
    orch = hv15.HunyuanVideo15Orchestrator(args)
    for mode in ("compile", "generate"):
        args.stage_mode = mode
        orch._run_stage_internal("aot_transformer", args)
        orch._run_stage_internal("aot_vae", args)
    assert os.path.exists(os.path.join(tmp, "o.pt"))
    native = sorted(m for m in sys.modules if m.startswith("difflet.backends.neuron"))
    assert not native, native
    print("OK")
""")


def test_aot_stages_do_not_import_native(tmp_path):
    env = dict(os.environ, DIFFLET_BACKEND="cpu")
    result = subprocess.run([sys.executable, "-c", _NO_NATIVE, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0 and result.stdout.strip().endswith("OK"), result.stderr[-3000:]
