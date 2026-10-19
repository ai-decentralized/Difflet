"""scripts/cuda_e2e_flux_vs_diffusers.py imports and parses on a CPU host (the run is GPU-only)."""

import pathlib
import sys
import types

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

import cuda_e2e_flux_vs_diffusers as flux_e2e  # noqa: E402
import cuda_e2e_vs_diffusers as qwen_e2e  # noqa: E402


def test_defaults_are_the_gated_flux_dev_run():
    args = flux_e2e.build_parser().parse_args(["--model-path", "/snap"])
    assert args.model_path == "/snap" and args.model_id == "black-forest-labs/FLUX.1-dev"
    assert args.steps == 28 and args.guidance_scale == 3.5 and args.seed == 0
    assert (args.height, args.width) == (1024, 1024)
    assert args.out == "/tmp/logs/cuda_e2e_flux.json" and args.images_dir == "/tmp/logs"


def test_metrics_are_the_qwen_oracle_s():
    # One metric definition for both e2e gates.
    assert flux_e2e.image_metrics is qwen_e2e.image_metrics
    assert flux_e2e._save is qwen_e2e._save


def test_timings_record_carries_load_generate_and_peak(monkeypatch):
    # F4b: the Flux oracle records per-side timings as the Wan oracle does.
    rec = flux_e2e.timings_record(load_s=12.345, generate_s=16.78, peak_bytes=68_440_000_000)
    assert rec == {"load_s": 12.3, "generate_s": 16.8, "peak_alloc_gb": 68.44}
    rec = flux_e2e.timings_record(load_s=1.0, generate_s=2.0, peak_bytes=None)
    assert rec["peak_alloc_gb"] is None


def test_noise_floor_flag_parses_off_by_default():
    assert flux_e2e.build_parser().parse_args(["--model-path", "/snap"]).noise_floor is False
    assert (
        flux_e2e.build_parser().parse_args(["--model-path", "/snap", "--noise-floor"]).noise_floor
    )


class _FakeFluxPipeline:
    """Returns an image that depends on the bf16-reduction flag, so the perturbed run differs."""

    instances: list = []
    loads: list = []

    def __init__(self):
        self.calls = []
        _FakeFluxPipeline.instances.append(self)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.loads.append((path, kwargs))
        return cls()

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        base = kwargs["latents"].float().abs().clamp(0, 1)[:, :3, :8, :8]
        return types.SimpleNamespace(images=base * (0.5 if flag else 0.9))


@pytest.mark.parametrize("noise_floor", [False, True])
def test_run_diffusers_loads_once_and_measures_the_floor(monkeypatch, noise_floor):
    _FakeFluxPipeline.instances.clear()
    _FakeFluxPipeline.loads.clear()
    monkeypatch.setitem(
        sys.modules, "diffusers", types.SimpleNamespace(FluxPipeline=_FakeFluxPipeline)
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(flux_e2e, "_peak_bytes", lambda: None)
    before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    args = flux_e2e.build_parser().parse_args(["--model-path", "/snap"])
    noise = torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(0))
    img, timings, floor = flux_e2e.run_diffusers(
        args, torch.device("cpu"), noise, noise_floor=noise_floor
    )
    assert len(_FakeFluxPipeline.loads) == 1
    path, kwargs = _FakeFluxPipeline.loads[0]
    assert path == "/snap" and kwargs["torch_dtype"] == torch.bfloat16
    (pipe,) = _FakeFluxPipeline.instances
    assert len(pipe.calls) == (2 if noise_floor else 1)
    for call in pipe.calls:
        assert torch.equal(call["latents"], noise) and call["num_inference_steps"] == 28
    assert img.shape == (3, 8, 8) and set(timings) == {"load_s", "generate_s", "peak_alloc_gb"}
    # The flag is restored after the perturbed run.
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == before
    if not noise_floor:
        assert floor is None
        return
    assert floor["perturbation"] == (
        f"allow_bf16_reduced_precision_reduction={not before} (exact: {before})"
    )
    assert set(floor) == {"perturbation", "metrics", "timings"}
    assert floor["timings"]["peak_alloc_gb"] is None
    assert floor["metrics"]["psnr_db"] < 60  # the perturbed image differs from the exact one


def test_noise_floor_restores_the_flag_when_the_perturbed_run_raises(monkeypatch):
    class _Raising(_FakeFluxPipeline):
        def __call__(self, **kwargs):
            if self.calls:
                raise RuntimeError("boom")
            return super().__call__(**kwargs)

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(FluxPipeline=_Raising))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(flux_e2e, "_peak_bytes", lambda: None)
    before = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    args = flux_e2e.build_parser().parse_args(["--model-path", "/snap"])
    with pytest.raises(RuntimeError, match="boom"):
        flux_e2e.run_diffusers(
            args, torch.device("cpu"), torch.zeros(1, 16, 8, 8), noise_floor=True
        )
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == before
