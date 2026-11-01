"""Incorrect, unmatched and missing evidence cannot become a parity pass."""

import pytest
import torch
from types import SimpleNamespace


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool])
@pytest.mark.parametrize("layout", ["scalar", "zero_stride_singleton", "sliced"])
def test_value_hash_accepts_scalar_and_nonstandard_contiguous_strides(dtype, layout):
    from tests.numerical.gpu_hunyuan_video_parity import _hash

    canonical = torch.tensor([1], dtype=dtype)
    if layout == "scalar":
        actual = canonical[0]
    elif layout == "zero_stride_singleton":
        actual = canonical.as_strided((1,), (0,))
    else:
        actual = torch.tensor([1, 9], dtype=dtype)[::2]
    assert _hash(actual) == _hash(canonical)


def test_numerical_metrics_detect_nonfinite_and_real_error():
    from tests.numerical.gpu_hunyuan_video_parity import tensor_metrics

    ref = torch.tensor([1.0, 2.0, 3.0])
    exact = tensor_metrics(ref, ref)
    assert exact["max_abs"] == 0 and exact["finite"] is True
    assert exact["reference_nonfinite"] == exact["actual_nonfinite"] == 0
    assert tensor_metrics(ref, ref + 1)["relative_l1"] == pytest.approx(0.5)
    assert tensor_metrics(ref, ref + 1)["relative_l2"] > 0
    assert tensor_metrics(ref, torch.tensor([float("nan"), 2, 3]))["finite"] is False


def test_missing_unverified_or_wrong_mode_cannot_pass_performance():
    from tests.numerical.gpu_hunyuan_video_parity import paired_performance

    settings = dict(
        model="hunyuan_video",
        checkpoint="abc",
        shape=[320, 512, 61],
        steps=50,
        guidance=6.0,
        noise_hash="noise",
        timestep_hash="times",
        output_boundary="cpu-fp32-video",
    )
    assert (
        paired_performance([], [], settings=settings, mode="eager")["status"]
        == "baseline_unavailable"
    )
    d = [
        {
            "latency_s": 10.1,
            "denoise_s": 9.1,
            "mode": "eager",
            "execution_verified": True,
            "settings": settings,
        }
        for _ in range(6)
    ]
    o = [{**row, "latency_s": 10.0, "denoise_s": 9.0} for row in d]
    result = paired_performance(d, o, settings=settings, mode="eager")
    assert result["status"] == "pass"
    assert result["generation_ratio"] == pytest.approx(1.01)
    o[0] = {**o[0], "mode": "compiled"}
    with pytest.raises(ValueError, match="mode"):
        paired_performance(d, o, settings=settings, mode="eager")
    o[0] = {**o[0], "mode": "eager", "execution_verified": False}
    assert paired_performance(d, o, settings=settings, mode="eager")["status"] == "inconclusive"


def test_mismatched_noise_or_schedule_is_rejected():
    from tests.numerical.gpu_hunyuan_video_parity import paired_performance

    settings = {"checkpoint": "abc", "noise_hash": "a", "timestep_hash": "b"}
    d = [
        dict(latency_s=1, denoise_s=1, mode="eager", execution_verified=True, settings=settings)
    ] * 6
    for key in ("noise_hash", "timestep_hash", "checkpoint"):
        other = {**d[0], "settings": {**settings, key: "different"}}
        with pytest.raises(ValueError, match="settings"):
            paired_performance(d, [other] * 6, settings=settings, mode="eager")


def test_uncalibrated_quality_and_nonfinite_outputs_cannot_pass():
    from tests.numerical.gpu_hunyuan_video_parity import quality_verdict

    metrics = {"finite": True, "relative_l2": 0.001, "max_abs": 0.01}
    assert quality_verdict(metrics, None) == "uncalibrated"
    limits = {"relative_l2": 0.01, "max_abs": 0.02}
    assert quality_verdict(metrics, limits) == "pass"
    assert quality_verdict({**metrics, "finite": False}, limits) == "fail"


def test_clean_request_keeps_timestep_transfers_outside_the_timing_boundary(monkeypatch):
    from tests.numerical import gpu_hunyuan_video_parity as harness

    active = [False]
    cpu = torch.Tensor.cpu

    def checked_cpu(tensor, *a, **kw):
        assert not active[0], "timestep transfer synchronized the timed request"
        return cpu(tensor, *a, **kw)

    def clock():
        active[0] = not active[0]
        return 0.0 if active[0] else 2.0

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 1000.0

    class Model(torch.nn.Module):
        def forward(self, hidden_states, timestep, guidance):
            return hidden_states

    model = Model()
    scheduler = SimpleNamespace(step=lambda prediction, timestep, latents: (latents,))

    class Application:
        def generate(self, prompt, *, latents, num_inference_steps, guidance_scale):
            for t in torch.tensor([1000.0, 500.0]):
                prediction = model(latents, t[None], guidance=torch.tensor([6000.0]))
                scheduler.step(prediction, t, latents)
            return torch.zeros(1, 3, 9, 32, 32)

    monkeypatch.setattr(torch.Tensor, "cpu", checked_cpu)
    monkeypatch.setattr(harness.time, "perf_counter", clock)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    args = SimpleNamespace(
        num_frames=9,
        height=32,
        width=32,
        steps=2,
        prompt="cat",
        checkpoint_revision="a" * 40,
        vae_tiling=False,
    )
    _, trajectory, row = harness._request(
        Application(), SimpleNamespace(scheduler=scheduler), model, args, seed=0
    )
    assert trajectory == [] and row["latency_s"] == 2.0
    assert row["settings"]["timestep_hash"] == harness._hash(torch.tensor([1000.0, 500.0]))
    assert row["settings"]["dit_timestep_hash"] == harness._hash(torch.tensor([1000.0, 500.0]))
    assert row["settings"]["guidance_hash"] == harness._hash(torch.tensor([6000.0, 6000.0]))


def test_checkpoint_revision_must_be_the_resolved_snapshot_not_a_label(tmp_path):
    from tests.numerical.gpu_hunyuan_video_parity import checkpoint_revision

    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    assert checkpoint_revision(snapshot, "a" * 40) == "a" * 40
    with pytest.raises(ValueError, match="revision"):
        checkpoint_revision(snapshot, "b" * 40)
    with pytest.raises(ValueError, match="snapshot"):
        checkpoint_revision(tmp_path, "a" * 40)


def test_video_pass_cannot_hide_missing_or_divergent_trajectory():
    from tests.numerical.gpu_hunyuan_video_parity import compare_outputs

    video, latent = torch.ones(1, 3, 1, 2, 2), torch.ones(1, 16, 1, 1, 1)
    limits = {"video": {"relative_l2": 0.01}, "trajectory": {"relative_l2": 0.01}}
    with pytest.raises(ValueError, match="trajectory"):
        compare_outputs(video, video, [latent], [], steps=1, limits=limits)
    report = compare_outputs(video, video, [latent], [latent * 2], steps=1, limits=limits)
    assert report["video_quality"] == "pass" and report["quality"] == "fail"
    assert (
        compare_outputs(video, video, [latent], [latent], steps=1, limits=None)["quality"]
        == "uncalibrated"
    )


def test_profile_request_is_short_without_changing_the_timed_request():
    from tests.numerical.gpu_hunyuan_video_parity import profile_request_args

    request = SimpleNamespace(steps=50, profile_steps=2, num_frames=61)
    traced = profile_request_args(request)
    assert traced.steps == 2 and traced.num_frames == request.num_frames
    assert request.steps == 50
    with pytest.raises(ValueError, match="profile"):
        profile_request_args(SimpleNamespace(steps=50, profile_steps=0))


def _main_fixture(monkeypatch, tmp_path, *, changed=None, mode="eager", cuda_events=True):
    import json
    from tests.numerical import gpu_hunyuan_video_parity as harness

    engine = [None]
    settings = dict(
        model="hunyuan_video",
        checkpoint="a" * 40,
        shape=[2, 2, 1],
        steps=1,
        guidance=6.0,
        prompt="cat",
        noise_hash="noise",
        timestep_hash="time",
        dit_timestep_hash="dit-time",
        guidance_hash="guidance",
        vae_tiling=False,
        output_boundary="cpu-fp32-video",
    )

    def load(name, *args):
        engine[0] = name
        return None, None, torch.nn.Identity()

    def request(*args, seed, capture=False):
        effective = dict(settings)
        if engine[0] == "difflet" and changed:
            effective[changed] = "different"
        row = dict(seed=seed, settings=effective, latency_s=1.0, denoise_s=0.9)
        return torch.ones(1, 3, 1, 2, 2), [torch.ones(1, 2, 1, 1, 1)], row

    class Profile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def events(self):
            dit = SimpleNamespace(
                name="phase/DiT", cpu_parent=None, device_type=torch.autograd.DeviceType.CPU
            )
            return (
                [
                    dit,
                    SimpleNamespace(
                        name="fixture-kernel",
                        cpu_parent=dit,
                        device_type=torch.autograd.DeviceType.CUDA,
                    ),
                ]
                if cuda_events
                else [dit]
            )

        def export_chrome_trace(self, path):
            from pathlib import Path

            Path(path).write_text("{}")

    monkeypatch.setattr(harness, "_load", load)
    monkeypatch.setattr(harness, "_request", request)
    monkeypatch.setattr(harness.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(harness.torch.cuda, "get_device_name", lambda: "fixture")
    monkeypatch.setattr(harness.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(harness.torch.profiler, "profile", lambda **kwargs: Profile())
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps(dict(video={"max_abs": 0}, trajectory={"max_abs": 0})))
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    output = tmp_path / "output"
    argv = [
        "--model-path",
        str(snapshot),
        "--checkpoint-revision",
        "a" * 40,
        "--height",
        "2",
        "--width",
        "2",
        "--num-frames",
        "1",
        "--steps",
        "1",
        "--profile-steps",
        "1",
        "--mode",
        mode,
        "--seeds",
        "0",
        "--repeats",
        "1",
        "--calibration-limits",
        str(limits),
        "--output-dir",
        str(output),
    ]
    return harness, argv, output


@pytest.mark.parametrize(
    "changed",
    [
        "noise_hash",
        "timestep_hash",
        "dit_timestep_hash",
        "guidance_hash",
        "checkpoint",
        "prompt",
        "output_boundary",
    ],
)
def test_main_rejects_unmatched_quality_capture_contract(monkeypatch, tmp_path, changed):
    harness, argv, _ = _main_fixture(monkeypatch, tmp_path, changed=changed)
    with pytest.raises(ValueError, match="contract"):
        harness.main(argv)


@pytest.mark.parametrize("mode,cuda_events", [("compiled", True), ("eager", False)])
def test_main_cannot_accept_quality_without_requested_mode_execution(
    monkeypatch, tmp_path, mode, cuda_events
):
    import json

    harness, argv, output = _main_fixture(monkeypatch, tmp_path, mode=mode, cuda_events=cuda_events)
    assert harness.main(argv) == 2
    report = json.loads((output / "report.json").read_text())
    assert report["reference_quality_calibrated_and_passing"] is False


def test_main_binds_saved_quality_tensors_to_executed_contract(monkeypatch, tmp_path):
    harness, argv, output = _main_fixture(monkeypatch, tmp_path)
    assert harness.main(argv) == 0
    data = torch.load(output / "difflet-eager-seed0.pt", weights_only=True)
    assert data["capture_metadata"]["settings"]["noise_hash"] == "noise"
    assert data["capture_metadata"]["seed"] == 0
