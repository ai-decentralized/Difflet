"""scripts/cuda_e2e_wan_vs_diffusers.py imports and parses on a CPU host (the run is GPU-only).

The pure parts are tested here: the defaults of the gated run, and the two layouts the
video is compared in — Difflet's ``(B, 3, F, H, W)`` in [-1, 1] and diffusers'
``WanPipeline(output_type="pt")`` ``(B, F, 3, H, W)`` in [0, 1] — both brought to
``(F, H, W, 3)`` in [0, 1].
"""

import os
import pathlib
import subprocess
import sys

import torch

SCRIPTS = pathlib.Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import cuda_e2e_vs_diffusers as qwen_e2e  # noqa: E402
import cuda_e2e_wan_vs_diffusers as wan_e2e  # noqa: E402


def test_defaults_are_the_gated_wan22_run():
    args = wan_e2e.build_parser().parse_args(["--model-path", "/snap"])
    assert args.model_path == "/snap" and args.model_id == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert args.steps == 40 and args.guidance_scale == 4.0 and args.guidance_scale_2 == 3.0
    assert args.seed == 0 and (args.height, args.width, args.num_frames) == (480, 832, 9)
    assert args.out == "/tmp/logs/cuda_e2e_wan.json" and args.images_dir == "/tmp/logs"


def test_metrics_and_png_writer_are_the_qwen_oracle_s():
    assert wan_e2e.image_metrics is qwen_e2e.image_metrics
    assert wan_e2e._save is qwen_e2e._save


def test_both_layouts_reach_the_same_frames_hwc_unit_video():
    torch.manual_seed(0)
    unit = torch.rand(9, 6, 10, 3)  # (F, H, W, 3) in [0, 1]
    difflet = (unit * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, F, H, W) in [-1, 1]
    diffusers = unit.permute(0, 3, 1, 2).unsqueeze(0)  # (1, F, 3, H, W) in [0, 1]
    a = wan_e2e.difflet_unit_video(difflet)
    b = wan_e2e.diffusers_unit_video(diffusers)
    assert a.shape == b.shape == (9, 6, 10, 3)
    assert torch.allclose(a, unit, atol=1e-6) and torch.allclose(b, unit)
    m = wan_e2e.image_metrics(a, b)
    assert m["psnr_db"] > 100 and abs(m["cosine"] - 1.0) < 1e-9


def test_units_are_clamped():
    v = torch.full((1, 3, 1, 2, 2), 3.0)
    assert wan_e2e.difflet_unit_video(v).max() == 1.0
    assert wan_e2e.diffusers_unit_video(-v.permute(0, 2, 1, 3, 4)).min() == 0.0


def test_first_frame_is_chw_for_the_png_writer(tmp_path):
    video = torch.rand(9, 6, 10, 3)
    frame = wan_e2e.first_frame(video)
    assert frame.shape == (3, 6, 10) and torch.equal(frame, video[0].permute(2, 0, 1))
    wan_e2e._save(frame, tmp_path / "f.png")
    assert (tmp_path / "f.png").is_file()


def test_help_runs_on_a_cpu_host():
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "cuda_e2e_wan_vs_diffusers.py"), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DIFFLET_BACKEND": "cpu"},
    )
    assert proc.returncode == 0 and "--num-frames" in proc.stdout, proc.stderr[-2000:]


# ------------------------------------------------- timestep-dtype A/B (F1b task 2)
#
# ``--timestep-dtypes float32,model`` runs Difflet once per variant (``timestep_dtype`` is a
# CudaWanApplication kwarg, reaching it through DiffletPipeline's ``application_kwargs``)
# against ONE diffusers run from ONE noise tensor, drawn by the first Difflet load.

import inspect  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402


def test_timestep_dtypes_default_to_production_and_parse_a_list():
    parse = wan_e2e.build_parser().parse_args
    assert parse(["--model-path", "/s"]).timestep_dtypes == ["float32"]
    assert parse(["--model-path", "/s", "--timestep-dtypes", "float32,model"]).timestep_dtypes == [
        "float32",
        "model",
    ]
    assert parse(
        ["--model-path", "/s", "--timestep-dtypes", " model , float32 "]
    ).timestep_dtypes == [
        "model",
        "float32",
    ]


@pytest.mark.parametrize("bad", ["", "bf16", "float32,float32", "float32,,model", "fp32"])
def test_timestep_dtypes_reject_what_the_app_would_refuse(bad):
    with pytest.raises(SystemExit):
        wan_e2e.build_parser().parse_args(["--model-path", "/s", "--timestep-dtypes", bad])


def test_each_variant_has_its_own_first_frame_png(tmp_path):
    assert (
        wan_e2e.variant_png(tmp_path, "float32") == tmp_path / "wan_e2e_difflet_float32_frame0.png"
    )
    assert wan_e2e.variant_png(tmp_path, "model") == tmp_path / "wan_e2e_difflet_model_frame0.png"


def _timings(tag: float) -> dict:
    return {"load_s": tag, "generate_s": tag, "peak_alloc_gb": tag}


def test_e2e_variants_share_the_reference(tmp_path, monkeypatch):
    torch.manual_seed(0)
    noise = torch.randn(1, 16, 3, 4, 6)
    reference = torch.rand(9, 6, 10, 3)
    calls = []

    def fake_run_difflet(args, *, timestep_dtype, noise=None):
        calls.append(("difflet", timestep_dtype, noise))
        drawn = globals_noise if noise is None else noise
        # The fp32 variant is nearer the reference than Release 1's bf16 cast.
        offset = {"float32": 0.01, "model": 0.05}[timestep_dtype]
        return (reference + offset).clamp(0, 1), drawn, _timings(len(calls))

    def fake_run_diffusers(args, device, noise):
        calls.append(("diffusers", None, noise))
        return reference, _timings(99.0)

    globals_noise = noise
    monkeypatch.setattr(wan_e2e, "run_difflet", fake_run_difflet)
    monkeypatch.setattr(wan_e2e, "run_diffusers", fake_run_diffusers)
    monkeypatch.setattr(wan_e2e, "_device_name", lambda: "fake GPU")
    out = tmp_path / "e2e.json"
    wan_e2e.main(
        [
            "--model-path",
            "/snap",
            "--timestep-dtypes",
            "float32,model",
            "--out",
            str(out),
            "--images-dir",
            str(tmp_path),
        ]
    )

    # One noise tensor: drawn by the first Difflet load, handed to every later variant and to
    # the single diffusers run.
    assert [(side, dtype) for side, dtype, _ in calls] == [
        ("difflet", "float32"),
        ("difflet", "model"),
        ("diffusers", None),
    ]
    assert calls[0][2] is None
    assert calls[1][2] is noise and calls[2][2] is noise

    payload = json.loads(out.read_text())
    assert payload["timestep_dtypes"] == ["float32", "model"]
    assert list(payload["variants"]) == ["float32", "model"]
    for name, variant in payload["variants"].items():
        assert set(variant) == {"metrics", "timings", "image"}
        assert variant["image"] == str(wan_e2e.variant_png(tmp_path, name))
        assert pathlib.Path(variant["image"]).is_file()
    v = payload["variants"]
    assert v["float32"]["metrics"] == wan_e2e.image_metrics(
        (reference + 0.01).clamp(0, 1), reference
    )
    assert v["float32"]["metrics"]["psnr_db"] > v["model"]["metrics"]["psnr_db"]
    assert v["float32"]["timings"] == _timings(1) and v["model"]["timings"] == _timings(2)
    # The gate's existing keys are the first variant's.
    assert payload["metrics"] == v["float32"]["metrics"]
    assert payload["timings"] == {"difflet": _timings(1), "diffusers": _timings(99.0)}
    assert payload["images"]["difflet"] == v["float32"]["image"]
    assert pathlib.Path(payload["images"]["diffusers"]).is_file()
    assert payload["latent_shape"] == list(noise.shape)
    assert payload["video_shape"] == [9, 6, 10, 3] and payload["device"] == "fake GPU"


class _FakeApp:
    def __init__(self, noise):
        self.noise = noise
        self.config = SimpleNamespace(in_channels=16)
        self.pipeline = SimpleNamespace(prepare_latents=self._prepare_latents)
        self.drawn = 0
        self.generate_kwargs = None

    def _prepare_latents(self, **kwargs):
        self.drawn += 1
        return self.noise.clone()

    def generate(self, prompt, **kwargs):
        self.generate_kwargs = kwargs
        return torch.zeros(1, 3, 9, 6, 10)


@pytest.mark.parametrize("given", [False, True])
def test_run_difflet_passes_the_timestep_dtype_and_reuses_given_noise(monkeypatch, given):
    import difflet

    real = difflet.DiffletPipeline.from_pretrained
    torch.manual_seed(0)
    noise = torch.randn(1, 16, 3, 4, 6)
    app = _FakeApp(noise)
    seen = {}

    class FakePipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            inspect.signature(real).bind(*args, **kwargs)  # every kwarg is a real one
            seen.update(kwargs)
            return SimpleNamespace(app=app)

    monkeypatch.setattr(difflet, "DiffletPipeline", FakePipeline)
    for name in ("reset_peak_memory_stats", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 2**30)
    args = wan_e2e.build_parser().parse_args(["--model-path", "/snap"])

    given_noise = torch.randn(1, 16, 3, 4, 6) if given else None
    video, out_noise, timings = wan_e2e.run_difflet(args, timestep_dtype="model", noise=given_noise)

    assert seen["application_kwargs"] == {"timestep_dtype": "model"}
    assert seen["backend"] == "cuda" and seen["model_path_override"] == "/snap"
    expected = given_noise if given else noise
    assert app.drawn == (0 if given else 1)
    assert torch.equal(out_noise, expected)
    latents = app.generate_kwargs["latents"]
    assert torch.equal(latents, expected) and latents is not expected  # a clone goes in
    assert video.shape == (9, 6, 10, 3)
    assert set(timings) == {"load_s", "generate_s", "peak_alloc_gb"}
