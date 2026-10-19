"""scripts/cuda_e2e_flux_vs_diffusers.py imports and parses on a CPU host (the run is GPU-only).

The runs are exercised with fakes for both pipelines: one load per side over seeds, the same
noise on both sides per seed, the bf16-reduction flag restored after every perturbed run, and
the JSON's per-seed, aggregate and first-seed keys (F9c).
"""

import inspect
import json
import pathlib
import sys
import types

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

import cuda_e2e_flux_vs_diffusers as flux_e2e  # noqa: E402
import cuda_e2e_vs_diffusers as qwen_e2e  # noqa: E402
import cuda_e2e_wan_vs_diffusers as wan_e2e  # noqa: E402


def test_defaults_are_the_gated_flux_dev_run():
    args = flux_e2e.build_parser().parse_args(["--model-path", "/snap"])
    assert args.model_path == "/snap" and args.model_id == "black-forest-labs/FLUX.1-dev"
    assert args.steps == 28 and args.guidance_scale == 3.5 and args.seed == 0
    assert (args.height, args.width) == (1024, 1024)
    assert args.out == "/tmp/logs/cuda_e2e_flux.json" and args.images_dir == "/tmp/logs"
    # One seed, no noise floor: the F5 run.
    assert args.seeds is None and args.noise_floor is False
    assert flux_e2e.run_seeds(args) == [0]


def test_metrics_are_the_qwen_oracle_s():
    # One metric definition for both e2e gates.
    assert flux_e2e.image_metrics is qwen_e2e.image_metrics
    assert flux_e2e._save is qwen_e2e._save
    # One definition of the seed helpers for both multi-seed scripts (F9c ruling R1).
    for name in ("parse_seeds", "run_seeds", "aggregate_metrics"):
        assert getattr(flux_e2e, name) is getattr(qwen_e2e, name) is getattr(wan_e2e, name), name


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


FLAG0 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
NOISE_SHAPE = (1, 4, 64)  # (B, (H/16)*(W/16), 64): packed latents at 32 x 32
OFFSET = 0.02  # how far the fake Difflet image sits from the fake diffusers one
DRIFT = 3e-3  # how far the fake diffusers image moves with the flag flipped


def _pattern(k: int) -> torch.Tensor:
    n = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    return torch.sin(n * 0.37 * (k + 1))


def _base(latents: torch.Tensor) -> torch.Tensor:
    """The fake image a noise tensor denoises to, (1, 3, 8, 8) in [0, 1]."""
    return 0.5 + 0.2 * torch.tanh(latents.float().mean()) + 0.1 * _pattern(0)


def _seed_noise(seed: int) -> torch.Tensor:
    return torch.randn(NOISE_SHAPE, generator=torch.Generator().manual_seed(seed)).to(
        torch.bfloat16
    )


def _unit(images: torch.Tensor) -> torch.Tensor:
    return images[0].float().clamp(0, 1)


class _FakeFluxApp:
    """CudaFluxApplication's surface the script uses: pipe.prepare_latents and generate."""

    def __init__(self):
        self.pipe = types.SimpleNamespace(
            prepare_latents=self._prepare_latents,
            transformer=types.SimpleNamespace(config=types.SimpleNamespace(in_channels=64)),
        )
        self.draws = []
        self.generated = []

    def _prepare_latents(self, batch, channels, height, width, dtype, device, generator):
        assert (batch, channels, height, width, dtype) == (1, 16, 32, 32, torch.bfloat16)
        noise = torch.randn(NOISE_SHAPE, generator=generator).to(dtype)
        self.draws.append(noise)
        return noise, None

    def generate(self, prompt, *, latents, output_type, **kwargs):
        assert output_type == "pt" and kwargs["num_inference_steps"] == 28
        self.generated.append(latents.clone())
        images = _base(latents) + OFFSET * _pattern(2)
        latents.add_(1.0)  # a pipeline may step its latents in place: the noise must survive
        return types.SimpleNamespace(images=images)


class _FakeDiffletPipeline:
    loads: list = []
    apps: list = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        inspect.signature(_REAL_FROM_PRETRAINED).bind(*args, **kwargs)  # real kwargs only
        cls.loads.append(kwargs)
        app = _FakeFluxApp()
        cls.apps.append(app)
        return types.SimpleNamespace(app=app)


import difflet as _difflet  # noqa: E402

_REAL_FROM_PRETRAINED = _difflet.DiffletPipeline.from_pretrained


class _FakeFluxPipeline:
    """diffusers.FluxPipeline's surface; its image moves by DRIFT when the bf16-reduction flag
    is flipped from its default, so the perturbed run differs from the exact one."""

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
        flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        self.calls.append({**kwargs, "flag": flag})
        drift = 0.0 if flag == FLAG0 else DRIFT
        return types.SimpleNamespace(images=_base(kwargs["latents"]) + drift * _pattern(1))


@pytest.fixture
def fakes(monkeypatch):
    import difflet

    _FakeDiffletPipeline.loads, _FakeDiffletPipeline.apps = [], []
    _FakeFluxPipeline.loads, _FakeFluxPipeline.instances = [], []
    monkeypatch.setattr(difflet, "DiffletPipeline", _FakeDiffletPipeline)
    monkeypatch.setitem(
        sys.modules, "diffusers", types.SimpleNamespace(FluxPipeline=_FakeFluxPipeline)
    )
    for name in ("reset_peak_memory_stats", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(flux_e2e, "_peak_bytes", lambda: 36_330_000_000)
    monkeypatch.setattr(flux_e2e, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(flux_e2e, "_device_name", lambda: "fake GPU")
    return types.SimpleNamespace(difflet=_FakeDiffletPipeline, diffusers=_FakeFluxPipeline)


def _args(*extra):
    return flux_e2e.build_parser().parse_args(
        ["--model-path", "/snap", "--height", "32", "--width", "32", *extra]
    )


def test_seeds_parse_and_override_the_single_seed():
    parse = flux_e2e.build_parser().parse_args
    args = parse(["--model-path", "/s", "--seeds", "0,1,2"])
    assert args.seeds == [0, 1, 2] and flux_e2e.run_seeds(args) == [0, 1, 2]
    assert flux_e2e.run_seeds(parse(["--model-path", "/s", "--seed", "5"])) == [5]
    args = parse(["--model-path", "/s", "--seed", "5", "--seeds", " 3 , 7 "])
    assert flux_e2e.run_seeds(args) == [3, 7]


@pytest.mark.parametrize("bad", ["", "a", "1,1", "1,,2", "0.5", "-1"])
def test_seeds_reject_bad_lists(bad):
    with pytest.raises(SystemExit):
        flux_e2e.build_parser().parse_args(["--model-path", "/s", "--seeds", bad])


def test_run_difflet_loads_once_and_generates_each_seed(fakes):
    images, noises, timings = flux_e2e.run_difflet(_args(), torch.device("cpu"), [4, 7])
    (load,) = fakes.difflet.loads
    assert load["backend"] == "cuda" and load["model_path_override"] == "/snap"
    assert (load["height"], load["width"]) == (32, 32)
    (app,) = fakes.difflet.apps
    assert list(images) == list(noises) == list(timings) == [4, 7]
    # Each seed's noise is the seeded packed draw (as the single-seed run drew it).
    assert len(app.draws) == 2
    for seed, latents in zip((4, 7), app.generated):
        assert torch.equal(noises[seed], _seed_noise(seed))
        assert torch.equal(latents, noises[seed])
        assert torch.equal(images[seed], _unit(_base(noises[seed]) + OFFSET * _pattern(2)))
        assert timings[seed]["peak_alloc_gb"] == 36.33
        assert set(timings[seed]) == {"load_s", "generate_s", "peak_alloc_gb"}
    assert timings[4]["load_s"] == timings[7]["load_s"]  # one load, shared


@pytest.mark.parametrize("noise_floor", [False, True])
def test_run_diffusers_loads_once_and_runs_each_seed(fakes, noise_floor):
    noises = {seed: _seed_noise(seed) for seed in (0, 1, 2)}
    images, timings, floors = flux_e2e.run_diffusers(
        _args(), torch.device("cpu"), noises, noise_floor=noise_floor
    )
    assert len(fakes.diffusers.loads) == 1
    path, kwargs = fakes.diffusers.loads[0]
    assert path == "/snap" and kwargs["torch_dtype"] == torch.bfloat16
    (pipe,) = fakes.diffusers.instances
    runs = 2 if noise_floor else 1
    assert len(pipe.calls) == 3 * runs
    for i, seed in enumerate((0, 1, 2)):
        for j in range(runs):
            call = pipe.calls[i * runs + j]
            assert torch.equal(call["latents"], noises[seed]) and call["num_inference_steps"] == 28
            # Per seed: the exact run (diffusers as it ships), then the flag flipped.
            assert call["flag"] == (FLAG0 if j == 0 else not FLAG0)
    # The flag is restored after every perturbed run.
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == FLAG0
    assert list(images) == list(timings) == [0, 1, 2]
    for seed in (0, 1, 2):
        assert torch.equal(images[seed], _unit(_base(noises[seed])))
        assert timings[seed]["peak_alloc_gb"] == 36.33
        assert set(timings[seed]) == {"load_s", "generate_s", "peak_alloc_gb"}
    if not noise_floor:
        assert floors == {}
        return
    assert list(floors) == [0, 1, 2]
    for seed, floor in floors.items():
        assert set(floor) == {"perturbation", "metrics", "timings"}
        assert floor["perturbation"] == (
            f"allow_bf16_reduced_precision_reduction={not FLAG0} (exact: {FLAG0})"
        )
        perturbed = _unit(_base(noises[seed]) + DRIFT * _pattern(1))
        expected = flux_e2e.image_metrics(images[seed], perturbed)
        assert floor["metrics"] == pytest.approx(expected, rel=1e-6)
        assert floor["timings"]["peak_alloc_gb"] is None


def test_noise_floor_restores_the_flag_when_the_perturbed_run_raises(fakes, monkeypatch):
    class _Raising(_FakeFluxPipeline):
        def __call__(self, **kwargs):
            if self.calls:
                raise RuntimeError("boom")
            return super().__call__(**kwargs)

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(FluxPipeline=_Raising))
    with pytest.raises(RuntimeError, match="boom"):
        flux_e2e.run_diffusers(_args(), torch.device("cpu"), {0: _seed_noise(0)}, noise_floor=True)
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == FLAG0


def _summary(values):
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def test_seeds_share_loads_and_aggregate(fakes, tmp_path):
    out = tmp_path / "e2e.json"
    flux_e2e.main(
        [
            "--model-path",
            "/snap",
            "--height",
            "32",
            "--width",
            "32",
            "--seeds",
            "0,1,2",
            "--noise-floor",
            "--out",
            str(out),
            "--images-dir",
            str(tmp_path),
        ]
    )
    seeds = [0, 1, 2]
    # One load per side, one noise draw per seed, the same noise on both sides.
    assert len(fakes.difflet.loads) == 1 and len(fakes.diffusers.loads) == 1
    (app,) = fakes.difflet.apps
    (pipe,) = fakes.diffusers.instances
    assert len(app.draws) == 3 and len(pipe.calls) == 6  # exact + perturbed per seed
    for i, seed in enumerate(seeds):
        noise = _seed_noise(seed)
        assert torch.equal(app.generated[i], noise)
        assert torch.equal(pipe.calls[2 * i]["latents"], noise)
        assert torch.equal(pipe.calls[2 * i + 1]["latents"], noise)

    payload = json.loads(out.read_text())
    assert payload["seeds"] == seeds and payload["seed"] == 0
    assert list(payload["per_seed"]) == ["0", "1", "2"]
    for seed in seeds:
        entry = payload["per_seed"][str(seed)]
        noise = _seed_noise(seed)
        exact = _unit(_base(noise))
        expected = flux_e2e.image_metrics(_unit(_base(noise) + OFFSET * _pattern(2)), exact)
        assert entry["metrics"] == pytest.approx(expected, rel=1e-6)
        for side in ("difflet", "diffusers"):
            assert set(entry["timings"][side]) == {"load_s", "generate_s", "peak_alloc_gb"}
            png = flux_e2e.side_png(tmp_path, side, seed=seed)
            assert entry["images"][side] == str(png) and png.is_file()
        floor = entry["noise_floor"]
        perturbed = _unit(_base(noise) + DRIFT * _pattern(1))
        assert floor["metrics"] == pytest.approx(flux_e2e.image_metrics(exact, perturbed), rel=1e-6)
        assert "allow_bf16_reduced_precision_reduction" in floor["perturbation"]

    aggregate = payload["aggregate"]
    for key in ("psnr_db", "cosine"):
        values = [payload["per_seed"][str(s)]["metrics"][key] for s in seeds]
        assert aggregate["difflet"][key] == pytest.approx(_summary(values))
        values = [payload["per_seed"][str(s)]["noise_floor"]["metrics"][key] for s in seeds]
        assert aggregate["noise_floor"][key] == pytest.approx(_summary(values))

    # The pre-F9c keys are the first seed's, so the F4f readers and artifacts' shape hold.
    first = payload["per_seed"]["0"]
    assert payload["metrics"] == first["metrics"]
    assert payload["timings"] == first["timings"]
    assert payload["noise_floor"] == first["noise_floor"]
    for side in ("difflet", "diffusers"):
        legacy = flux_e2e.side_png(tmp_path, side)
        assert payload["images"][side] == str(legacy) and legacy.is_file()
        assert legacy.read_bytes() == flux_e2e.side_png(tmp_path, side, seed=0).read_bytes()
    assert payload["model"] == "flux" and payload["shape"] == [32, 32]
    assert payload["device"] == "fake GPU"


def test_one_seed_without_the_noise_floor_is_the_f4f_run(fakes, tmp_path):
    out = tmp_path / "e2e.json"
    flux_e2e.main(
        [
            "--model-path",
            "/snap",
            "--height",
            "32",
            "--width",
            "32",
            "--seed",
            "3",
            "--out",
            str(out),
            "--images-dir",
            str(tmp_path),
        ]
    )
    (pipe,) = fakes.diffusers.instances
    assert len(pipe.calls) == 1  # no perturbed run
    payload = json.loads(out.read_text())
    assert payload["seeds"] == [3] and payload["seed"] == 3
    assert list(payload["per_seed"]) == ["3"]
    assert payload["per_seed"]["3"]["noise_floor"] is None and payload["noise_floor"] is None
    assert payload["aggregate"]["noise_floor"] is None
    m = payload["per_seed"]["3"]["metrics"]
    assert payload["metrics"] == m
    assert payload["aggregate"]["difflet"]["psnr_db"] == {
        "min": m["psnr_db"],
        "mean": m["psnr_db"],
        "max": m["psnr_db"],
    }
    assert flux_e2e.side_png(tmp_path, "difflet").name == "flux_e2e_difflet.png"
    assert flux_e2e.side_png(tmp_path, "diffusers", seed=3).name == "flux_e2e_diffusers_seed3.png"
    assert flux_e2e.side_png(tmp_path, "diffusers").is_file()
