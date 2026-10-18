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
    # One seed, no noise floor: the F1/F1b run.
    assert args.seeds is None and args.noise_floor is False
    assert wan_e2e.run_seeds(args) == [0]


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
    assert (
        wan_e2e.variant_png(tmp_path, "float32", seed=2)
        == tmp_path / "wan_e2e_difflet_float32_seed2_frame0.png"
    )
    assert wan_e2e.diffusers_png(tmp_path) == tmp_path / "wan_e2e_diffusers_frame0.png"
    assert (
        wan_e2e.diffusers_png(tmp_path, seed=1) == tmp_path / "wan_e2e_diffusers_seed1_frame0.png"
    )
    assert (
        wan_e2e.diffusers_png(tmp_path, seed=1, bf16ts=True)
        == tmp_path / "wan_e2e_diffusers_bf16ts_seed1_frame0.png"
    )


# ------------------------------------------------- seeds and the noise floor (F9 task 1)
#
# ``--seeds 0,1,2`` loads Difflet once per variant and diffusers once; each seed's noise is
# drawn by the first Difflet load from ``torch.Generator().manual_seed(seed)`` and handed to
# every later variant and to diffusers. ``--noise-floor`` adds, per seed, a second diffusers run
# with both experts' ``forward`` wrapped so the timestep is rounded through bf16, compared
# against the exact run: how far diffusers moves from itself under Release 1's cast.

F, H, W = 9, 6, 10
NOISE_SHAPE = (1, 16, 3, 4, 6)
OFFSET = {"float32": 0.01, "model": 0.03}
EXACT_TS = (999, 750)  # one step on each expert: 999 >= the boundary, 750 below it


def _pattern(k: int) -> torch.Tensor:
    n = torch.arange(F * H * W * 3, dtype=torch.float32).reshape(F, H, W, 3)
    return torch.sin(n * 0.37 * (k + 1))


def _base(latents: torch.Tensor) -> torch.Tensor:
    """The fake 'clip' a noise tensor denoises to, (F, H, W, 3) in [0, 1]."""
    return 0.5 + 0.2 * torch.tanh(latents.float().mean()) + 0.1 * _pattern(0)


def _seed_noise(seed: int) -> torch.Tensor:
    return torch.randn(NOISE_SHAPE, generator=torch.Generator().manual_seed(seed))


def _stub_cuda(monkeypatch):
    for name in ("reset_peak_memory_stats", "synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 2**30)


class _FakeApp:
    """CudaWanApplication's surface the script uses: prepare_latents and generate."""

    def __init__(self, timestep_dtype):
        self.timestep_dtype = timestep_dtype
        self.config = SimpleNamespace(in_channels=16)
        self.pipeline = SimpleNamespace(prepare_latents=self._prepare_latents)
        self.draws = []
        self.generated = []

    def _prepare_latents(self, *, batch_size, channels, dtype, generator):
        assert (batch_size, channels, dtype) == (1, 16, torch.float32)
        noise = torch.randn(NOISE_SHAPE, generator=generator)
        self.draws.append(noise)
        return noise

    def generate(self, prompt, *, latents, output_type, **kwargs):
        assert output_type == "pt"
        self.generated.append(latents)
        unit = _base(latents) + OFFSET[self.timestep_dtype] * _pattern(2)
        return (unit * 2 - 1).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, F, H, W) in [-1, 1]


class _FakeDiffletPipeline:
    loads: list = []
    apps: list = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        inspect.signature(_REAL_FROM_PRETRAINED).bind(*args, **kwargs)  # real kwargs only
        cls.loads.append(kwargs)
        app = _FakeApp(kwargs["application_kwargs"]["timestep_dtype"])
        cls.apps.append(app)
        return SimpleNamespace(app=app)


import difflet as _difflet  # noqa: E402

_REAL_FROM_PRETRAINED = _difflet.DiffletPipeline.from_pretrained


class _RecordingTransformer(torch.nn.Module):
    """A WanTransformer3DModel stand-in: records what its forward receives."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=True,
        attention_kwargs=None,
    ):
        self.seen.append(
            {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_hidden_states_image": encoder_hidden_states_image,
                "return_dict": return_dict,
                "attention_kwargs": attention_kwargs,
            }
        )
        return (timestep.float().sum(),)


class _FakeWanPipeline:
    """diffusers.WanPipeline's surface: from_pretrained, .to, both experts called as diffusers
    0.38 calls them (keywords, ``timestep = t.expand(B)`` from an int64 schedule)."""

    loads: list = []
    instances: list = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.loads.append((path, kwargs))
        pipe = cls()
        cls.instances.append(pipe)
        return pipe

    def __init__(self):
        self.transformer = _RecordingTransformer()
        self.transformer_2 = _RecordingTransformer()
        self.calls = []

    def to(self, device):
        return self

    def __call__(self, *, latents, output_type, return_dict, **kwargs):
        assert output_type == "pt" and return_dict is False
        self.calls.append({"latents": latents, **kwargs})
        context = torch.zeros(1, 4, 8)
        total = 0.0
        for expert, t in ((self.transformer, EXACT_TS[0]), (self.transformer_2, EXACT_TS[1])):
            out = expert(
                hidden_states=latents,
                timestep=torch.tensor(t).expand(latents.shape[0]),
                encoder_hidden_states=context,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            total += float(out)
        drift = total - sum(EXACT_TS)  # 0 exact; 3 under the bf16 rounding (1000 + 752)
        unit = _base(latents) + 1e-3 * drift * _pattern(1)
        return (unit.permute(0, 3, 1, 2).unsqueeze(0),)  # (1, F, 3, H, W) in [0, 1]


@pytest.fixture
def fakes(monkeypatch):
    import diffusers
    import difflet

    _FakeDiffletPipeline.loads, _FakeDiffletPipeline.apps = [], []
    _FakeWanPipeline.loads, _FakeWanPipeline.instances = [], []
    monkeypatch.setattr(difflet, "DiffletPipeline", _FakeDiffletPipeline)
    monkeypatch.setattr(diffusers, "WanPipeline", _FakeWanPipeline)
    monkeypatch.setattr(wan_e2e, "_device_name", lambda: "fake GPU")
    monkeypatch.setattr(wan_e2e, "_device", lambda: torch.device("cpu"))
    _stub_cuda(monkeypatch)
    return SimpleNamespace(difflet=_FakeDiffletPipeline, diffusers=_FakeWanPipeline)


def test_seeds_parse_and_override_the_single_seed():
    parse = wan_e2e.build_parser().parse_args
    args = parse(["--model-path", "/s", "--seeds", "0,1,2"])
    assert args.seeds == [0, 1, 2] and wan_e2e.run_seeds(args) == [0, 1, 2]
    assert wan_e2e.run_seeds(parse(["--model-path", "/s", "--seed", "5"])) == [5]
    args = parse(["--model-path", "/s", "--seed", "5", "--seeds", " 3 , 7 "])
    assert wan_e2e.run_seeds(args) == [3, 7]
    assert parse(["--model-path", "/s", "--noise-floor"]).noise_floor is True


@pytest.mark.parametrize("bad", ["", "a", "1,1", "1,,2", "0.5", "-1"])
def test_seeds_reject_bad_lists(bad):
    with pytest.raises(SystemExit):
        wan_e2e.build_parser().parse_args(["--model-path", "/s", "--seeds", bad])


def test_bf16_rounded_timestep_is_release1_s_cast():
    t = torch.tensor([999, 750, 500, 0])
    rounded = wan_e2e.bf16_rounded_timestep(t)
    assert rounded.dtype == torch.float32
    assert rounded.tolist() == [1000.0, 752.0, 500.0, 0.0]
    assert t.tolist() == [999, 750, 500, 0]  # the input is not touched


def test_noise_floor_wrapper_rounds_only_the_timestep():
    first, second = _RecordingTransformer(), _RecordingTransformer()
    pipe = SimpleNamespace(transformer=first, transformer_2=second)
    hidden, context = torch.randn(1, 16, 3, 4, 6), torch.randn(1, 4, 8)
    attention = {"scale": 1.0}

    def call(module, t):
        module(
            hidden_states=hidden,
            timestep=torch.tensor(t).expand(1),
            encoder_hidden_states=context,
            attention_kwargs=attention,
            return_dict=False,
        )
        return module.seen[-1]

    with wan_e2e.perturb_timestep(pipe):
        for module, t, expected in ((first, 999, 1000.0), (second, 750, 752.0)):
            seen = call(module, t)
            assert seen["timestep"].dtype == torch.float32
            assert seen["timestep"].tolist() == [expected]
            # Everything else reaches the expert untouched, the very same objects.
            assert seen["hidden_states"] is hidden
            assert seen["encoder_hidden_states"] is context
            assert seen["attention_kwargs"] is attention
            assert seen["return_dict"] is False
            assert seen["encoder_hidden_states_image"] is None
        # A positional timestep (the forward signature's second argument) is rounded too.
        first(hidden, torch.tensor([999]), context)
        assert first.seen[-1]["timestep"].tolist() == [1000.0]
        assert first.seen[-1]["hidden_states"] is hidden

    # Unwrapped again: the exact run that follows sees the int64 schedule as diffusers sends it.
    assert "forward" not in vars(first) and "forward" not in vars(second)
    for module, t in ((first, 999), (second, 750)):
        seen = call(module, t)
        assert seen["timestep"].dtype == torch.int64 and seen["timestep"].tolist() == [t]


def test_noise_floor_wrapper_tolerates_a_single_expert_and_unwraps_on_error():
    only = _RecordingTransformer()
    pipe = SimpleNamespace(transformer=only, transformer_2=None)
    with pytest.raises(RuntimeError, match="boom"):
        with wan_e2e.perturb_timestep(pipe):
            assert "forward" in vars(only)
            raise RuntimeError("boom")
    assert "forward" not in vars(only)
    # The same module under both names is wrapped once (rounded once) and unwrapped cleanly.
    same = _RecordingTransformer()
    with wan_e2e.perturb_timestep(SimpleNamespace(transformer=same, transformer_2=same)):
        same(torch.zeros(1), torch.tensor([750]), torch.zeros(1))
    assert same.seen[-1]["timestep"].tolist() == [752.0] and "forward" not in vars(same)


def test_noise_floor_on_a_real_wan_transformer_is_the_rounded_timestep():
    # diffusers' own WanTransformer3DModel (tiny, CPU): the wrapped call with UniPC's int64 999
    # is bit-for-bit the call with a float 1000.0 timestep, and unwrapped it is 999 again.
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=32,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=32,
    ).eval()
    hidden, context = torch.randn(1, 4, 1, 4, 4), torch.randn(1, 3, 16)

    def call(t):
        return model(
            hidden_states=hidden, timestep=t, encoder_hidden_states=context, return_dict=False
        )[0]

    with torch.no_grad():
        exact = call(torch.tensor([999]))
        rounded = call(torch.tensor([1000.0]))
        with wan_e2e.perturb_timestep(SimpleNamespace(transformer=model, transformer_2=None)):
            wrapped = call(torch.tensor([999]))
        after = call(torch.tensor([999]))
    assert torch.equal(wrapped, rounded) and torch.equal(after, exact)
    assert not torch.equal(exact, rounded)


def test_noise_floor_wrapper_restores_a_hooked_forward():
    # A diffusers hook (HookRegistry) installs an instance-level forward; the wrapper goes on top
    # of it and puts it back afterwards.
    module = _RecordingTransformer()
    hooked = module.forward
    module.forward = hooked
    with wan_e2e.perturb_timestep(SimpleNamespace(transformer=module)):
        module(torch.zeros(1), torch.tensor([999]), torch.zeros(1))
        assert module.seen[-1]["timestep"].tolist() == [1000.0]
    assert vars(module)["forward"] is hooked
    module(torch.zeros(1), torch.tensor([999]), torch.zeros(1))
    assert module.seen[-1]["timestep"].tolist() == [999]


@pytest.mark.parametrize("noise_floor", [False, True])
def test_run_diffusers_loads_once_and_runs_each_seed(fakes, noise_floor):
    args = wan_e2e.build_parser().parse_args(["--model-path", "/snap"])
    noises = {seed: _seed_noise(seed) for seed in (0, 1, 2)}
    exact, exact_t, perturbed, perturbed_t = wan_e2e.run_diffusers(
        args, torch.device("cpu"), noises, noise_floor=noise_floor
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
            assert torch.equal(call["latents"], noises[seed])
            assert call["negative_prompt"] is None and call["num_inference_steps"] == 40
            assert (call["guidance_scale"], call["guidance_scale_2"]) == (4.0, 3.0)
    # Per seed: the exact run (diffusers as it ships), then the perturbed one.
    for expert, t, rounded in ((pipe.transformer, 999, 1000.0), (pipe.transformer_2, 750, 752.0)):
        stamps = [(s["timestep"].dtype, s["timestep"].tolist()) for s in expert.seen]
        one_seed = [(torch.int64, [t])] + ([(torch.float32, [rounded])] if noise_floor else [])
        assert stamps == one_seed * 3
        assert "forward" not in vars(expert)
    assert list(exact) == [0, 1, 2] and list(exact_t) == [0, 1, 2]
    for seed, video in exact.items():
        assert torch.allclose(video, _base(noises[seed]).clamp(0, 1))
        assert set(exact_t[seed]) == {"load_s", "generate_s", "peak_alloc_gb"}
    if noise_floor:
        assert list(perturbed) == [0, 1, 2] and list(perturbed_t) == [0, 1, 2]
        for seed, video in perturbed.items():
            assert not torch.equal(video, exact[seed])
    else:
        assert perturbed == {} and perturbed_t == {}


@pytest.mark.parametrize("given", [False, True])
def test_run_difflet_loads_once_and_generates_each_seed(fakes, given):
    args = wan_e2e.build_parser().parse_args(["--model-path", "/snap"])
    given_noises = {seed: torch.randn(NOISE_SHAPE) for seed in (4, 7)} if given else None
    videos, noises, timings = wan_e2e.run_difflet(
        args, timestep_dtype="model", seeds=[4, 7], noises=given_noises
    )
    (load,) = fakes.difflet.loads
    assert load["application_kwargs"] == {"timestep_dtype": "model"}
    assert load["backend"] == "cuda" and load["model_path_override"] == "/snap"
    (app,) = fakes.difflet.apps
    assert list(videos) == list(noises) == list(timings) == [4, 7]
    if given:
        assert app.draws == []
        assert all(noises[s] is given_noises[s] for s in (4, 7))
    else:
        # The noise of each seed is the seeded host draw, made explicit.
        assert len(app.draws) == 2
        for seed in (4, 7):
            assert torch.equal(noises[seed], _seed_noise(seed))
    for seed, latents in zip((4, 7), app.generated):
        assert torch.equal(latents, noises[seed]) and latents is not noises[seed]  # a clone
        assert videos[seed].shape == (F, H, W, 3)
        assert set(timings[seed]) == {"load_s", "generate_s", "peak_alloc_gb"}
    assert timings[4]["load_s"] == timings[7]["load_s"]  # one load, shared


def _summary(values):
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def test_seeds_share_loads_and_aggregate(fakes, tmp_path):
    out = tmp_path / "e2e.json"
    wan_e2e.main(
        [
            "--model-path",
            "/snap",
            "--seeds",
            "0,1,2",
            "--noise-floor",
            "--timestep-dtypes",
            "float32,model",
            "--out",
            str(out),
            "--images-dir",
            str(tmp_path),
        ]
    )
    seeds = [0, 1, 2]
    # One Difflet load per variant, one diffusers load, one noise draw per seed.
    assert [k["application_kwargs"]["timestep_dtype"] for k in fakes.difflet.loads] == [
        "float32",
        "model",
    ]
    assert len(fakes.diffusers.loads) == 1
    first_app, second_app = fakes.difflet.apps
    assert len(first_app.draws) == 3 and second_app.draws == []
    (pipe,) = fakes.diffusers.instances
    assert len(pipe.calls) == 6  # exact + perturbed per seed
    for i, seed in enumerate(seeds):
        noise = _seed_noise(seed)
        assert torch.equal(first_app.generated[i], noise)
        assert torch.equal(second_app.generated[i], noise)
        assert torch.equal(pipe.calls[2 * i]["latents"], noise)
        assert torch.equal(pipe.calls[2 * i + 1]["latents"], noise)

    payload = json.loads(out.read_text())
    assert payload["seeds"] == seeds and payload["seed"] == 0
    assert list(payload["per_seed"]) == ["0", "1", "2"]
    for seed in seeds:
        entry = payload["per_seed"][str(seed)]
        noise = _seed_noise(seed)
        exact = _base(noise).clamp(0, 1)
        for name in ("float32", "model"):
            variant = entry["variants"][name]
            difflet_video = (_base(noise) + OFFSET[name] * _pattern(2)).clamp(0, 1)
            expected = wan_e2e.image_metrics(difflet_video, exact)
            assert variant["metrics"] == pytest.approx(expected, rel=1e-6)
            assert set(variant["timings"]) == {"load_s", "generate_s", "peak_alloc_gb"}
            assert pathlib.Path(wan_e2e.variant_png(tmp_path, name, seed=seed)).is_file()
        floor = entry["noise_floor"]
        perturbed = (_base(noise) + 3e-3 * _pattern(1)).clamp(0, 1)
        expected = wan_e2e.image_metrics(perturbed, exact)
        assert floor["metrics"] == pytest.approx(expected, rel=1e-6)
        assert set(floor["timings"]) == {"load_s", "generate_s", "peak_alloc_gb"}
        assert wan_e2e.diffusers_png(tmp_path, seed=seed).is_file()
        assert wan_e2e.diffusers_png(tmp_path, seed=seed, bf16ts=True).is_file()

    aggregate = payload["aggregate"]
    for name in ("float32", "model"):
        for key in ("psnr_db", "cosine"):
            values = [payload["per_seed"][str(s)]["variants"][name]["metrics"][key] for s in seeds]
            assert aggregate["variants"][name][key] == pytest.approx(_summary(values))
    for key in ("psnr_db", "cosine"):
        values = [payload["per_seed"][str(s)]["noise_floor"]["metrics"][key] for s in seeds]
        assert aggregate["noise_floor"][key] == pytest.approx(_summary(values))

    # The legacy keys are the first seed's, so pre-F9 readers and the gate's fallbacks work.
    first = payload["per_seed"]["0"]
    assert payload["timestep_dtypes"] == ["float32", "model"]
    assert payload["metrics"] == first["variants"]["float32"]["metrics"]
    for name in ("float32", "model"):
        legacy = payload["variants"][name]
        assert legacy["metrics"] == first["variants"][name]["metrics"]
        assert legacy["timings"] == first["variants"][name]["timings"]
        assert legacy["image"] == str(wan_e2e.variant_png(tmp_path, name))
        assert pathlib.Path(legacy["image"]).is_file()
    assert payload["timings"]["difflet"] == first["variants"]["float32"]["timings"]
    assert set(payload["timings"]["diffusers"]) == {"load_s", "generate_s", "peak_alloc_gb"}
    assert payload["images"]["difflet"] == str(wan_e2e.variant_png(tmp_path, "float32"))
    assert payload["images"]["diffusers"] == str(wan_e2e.diffusers_png(tmp_path))
    assert pathlib.Path(payload["images"]["diffusers"]).is_file()
    assert payload["noise_floor"] == first["noise_floor"]
    assert payload["latent_shape"] == list(NOISE_SHAPE)
    assert payload["video_shape"] == [F, H, W, 3] and payload["device"] == "fake GPU"


def test_one_seed_without_the_noise_floor_is_the_f1b_run(fakes, tmp_path):
    out = tmp_path / "e2e.json"
    wan_e2e.main(
        ["--model-path", "/snap", "--seed", "3", "--out", str(out), "--images-dir", str(tmp_path)]
    )
    assert len(fakes.difflet.loads) == 1 and len(fakes.diffusers.loads) == 1
    (pipe,) = fakes.diffusers.instances
    assert len(pipe.calls) == 1  # no perturbed run
    payload = json.loads(out.read_text())
    assert payload["seeds"] == [3] and payload["seed"] == 3
    assert payload["per_seed"]["3"]["noise_floor"] is None
    assert payload["noise_floor"] is None and payload["aggregate"]["noise_floor"] is None
    m = payload["per_seed"]["3"]["variants"]["float32"]["metrics"]
    assert payload["aggregate"]["variants"]["float32"]["psnr_db"] == {
        "min": m["psnr_db"],
        "mean": m["psnr_db"],
        "max": m["psnr_db"],
    }
    assert payload["metrics"] == payload["variants"]["float32"]["metrics"] == m
    assert not wan_e2e.diffusers_png(tmp_path, seed=3, bf16ts=True).exists()
    assert wan_e2e.diffusers_png(tmp_path).is_file()
