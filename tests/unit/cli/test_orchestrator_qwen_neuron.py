"""The Qwen-Image CLI on the neuron backend (torchrun MPMD, task D).

With DIFFLET_BACKEND=neuron, ``difflet compile/generate --model-id Qwen/Qwen-Image`` is one
torchrun launch of the generate stage (the text stage folded into it: rank 0 encodes on its
host) plus, for generate, a host decode in the driver. The ranks build
TorchNeuronQwenImageApplication through DiffletPipeline.from_pretrained, here the CPU fake of
tests/unit/cli/_fake_neuron_pipeline.py; only rank 0 writes. No test touches a NeuronCore.
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from difflet.cli.orchestrators import qwen_image as qwen_mod
from difflet.cli.orchestrators.qwen_image import QwenImageOrchestrator
from tests.unit.cli import _fake_neuron_pipeline as fake_pipeline

_QWEN_ID = "Qwen/Qwen-Image"
#: The shipped default, read before the fixture below replaces it.
_SHIPPED_DEVICE_VAE_DEFAULT = qwen_mod._DEVICE_VAE_DEFAULT


@pytest.fixture(autouse=True)
def _host_decode_unless_asked(monkeypatch):
    """The tests here check the launch plan, flags, DP queues and caches around the host decode,
    which the CPU fakes can run (a vae stage would need the real VAE weights). The decode
    placement itself, the shipped default included, is pinned by the --device-vae tests below,
    which set ``_DEVICE_VAE_DEFAULT`` themselves."""
    monkeypatch.setattr(qwen_mod, "_DEVICE_VAE_DEFAULT", False)
_REV = "75e0b4be04f60ec59a75f475837eced720f823b6"
_CACHE_ENV = ("DIFFLET_COMPILE_CACHE", "TORCH_NEURONX_NEFF_CACHE_DIR",
              "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "TORCH_NEURONX_HLO_CACHE_DIR")
_CACHE_NOTE = "compiles into --cache-dir: DIFFLET_COMPILE_CACHE="
_NEURON_APP_KWARGS = {
    "text_seq_len": 1024, "batch_size": 1, "enable_text_encoder": True,
    "enable_transformer": True, "enable_vae_decoder": False, "exec_mode": None,
    "encoder_threads": 12,
}


def _qwen_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id=_QWEN_ID, tp_degree=4, cp_degree=1, cp_mode="gather_kv", cfg_parallel=False,
        sp_enabled=False, height=1024, width=1024, num_frames=None, shapes=None,
        cache_dir=None, force=False, revision=None, host_vae=False, device_vae=False,
        prompt="a cat", output="/tmp/qwen.png",
        steps=4, guidance_scale=4.0, seed=42, stage_mode="generate",
        work_dir=None, keep_work_dir=False,
        teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None,
        compiled_dir=None, vae_tp_degree=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _unset(monkeypatch, *names):
    """Unset ``names`` now and restore them after the test, even when the code under test
    assigns os.environ itself (delenv of an absent name records nothing to undo)."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def _backend(monkeypatch, backend):
    _unset(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE")
    if backend is not None:
        monkeypatch.setenv("DIFFLET_BACKEND", backend)


def _record_stages(monkeypatch) -> tuple[list, list]:
    """run_stage and the host decode, recorded."""
    stages, decoded = [], []
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda orch, stage, **kw: stages.append((stage, kw["num_cores"])))
    monkeypatch.setattr(qwen_mod, "_decode_latents_host",
                        lambda *a, **k: decoded.append((a, k)))
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: "/fake/model")
    return stages, decoded


# ----------------------------------------------------------------- the stage plan


@pytest.mark.parametrize("keep", [False, True])
def test_neuron_generate_is_one_launch_and_a_host_decode(monkeypatch, tmp_path, keep):
    _backend(monkeypatch, "neuron")
    stages, decoded = _record_stages(monkeypatch)
    work = tmp_path / "work"
    args = _qwen_args(revision=_REV, work_dir=str(work), keep_work_dir=keep, height=768,
                      width=1360, output=str(tmp_path / "out.png"))

    QwenImageOrchestrator(args).generate()

    # The text stage is folded into the generate launch; tp 4 = every core of a trn2.3xlarge.
    assert stages == [("generate", 4)]
    assert decoded == [((str(work / "latents.pt"), "/fake/model", str(tmp_path / "out.png")),
                        {"height": 768, "width": 1360})]
    assert work.exists() is keep


def test_neuron_compile_is_the_generate_launch_only(monkeypatch, tmp_path):
    _backend(monkeypatch, "neuron")
    stages, decoded = _record_stages(monkeypatch)
    QwenImageOrchestrator(_qwen_args(tp_degree=2, cp_degree=2, cp_mode="ulysses")).compile()
    assert stages == [("generate", 4)] and decoded == []  # the host VAE compiles nothing


def test_neuron_run_compiles_then_generates(monkeypatch, tmp_path):
    _backend(monkeypatch, "neuron")
    stages, decoded = _record_stages(monkeypatch)
    downloads = []
    monkeypatch.setattr(QwenImageOrchestrator, "download", lambda self: downloads.append(1))
    QwenImageOrchestrator(_qwen_args(work_dir=str(tmp_path / "w"))).run()
    assert downloads == [1] and [s for s, _ in stages] == ["generate", "generate"]
    assert len(decoded) == 1


def test_neuron_generate_preserves_the_work_dir_when_the_decode_fails(monkeypatch, tmp_path,
                                                                      capsys):
    _backend(monkeypatch, "neuron")
    _record_stages(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(qwen_mod, "_decode_latents_host", boom)
    work = tmp_path / "work"
    with pytest.raises(RuntimeError, match="decode failed"):
        QwenImageOrchestrator(_qwen_args(work_dir=str(work))).generate()
    assert work.exists() and "work dir preserved" in capsys.readouterr().err


@pytest.mark.parametrize("backend", [None, "trainium", "tpu"])
@pytest.mark.parametrize("mode", ["compile", "generate"])
def test_aot_backends_keep_the_three_stages(monkeypatch, tmp_path, backend, mode):
    _backend(monkeypatch, backend)
    stages, decoded = _record_stages(monkeypatch)
    getattr(QwenImageOrchestrator(_qwen_args(work_dir=str(tmp_path / "w"))), mode)()
    assert stages == [("text", 4), ("generate", 4), ("vae", 1)] and decoded == []


def test_neuron_has_no_text_stage(monkeypatch):
    _backend(monkeypatch, "neuron")
    args = _qwen_args()
    with pytest.raises(RuntimeError, match="no 'text' stage"):
        QwenImageOrchestrator(args)._run_stage_internal("text", args)


# ----------------------------------------------------------------- device VAE (--device-vae)


@pytest.mark.parametrize("mode", ["compile", "generate"])
def test_neuron_device_vae_is_a_vae_stage_after_the_generate_launch(monkeypatch, tmp_path, mode):
    # A launch of its own on one core, as Wan's: the decoder never shares a core with the DiT.
    _backend(monkeypatch, "neuron")
    stages, decoded = _record_stages(monkeypatch)
    args = _qwen_args(device_vae=True, work_dir=str(tmp_path / "w"))
    getattr(QwenImageOrchestrator(args), mode)()
    assert stages == [("generate", 4), ("vae", 1)] and decoded == []


@pytest.mark.parametrize("device_default", [False, True])
@pytest.mark.parametrize("host_vae, device_vae", [(False, False), (True, False), (False, True)])
def test_vae_placement_flags_and_the_default(monkeypatch, tmp_path, device_default, host_vae,
                                             device_vae):
    # The vae stage on one core is the default (flipped after the device measurements);
    # --host-vae is the opt-out whatever the default is.
    assert _SHIPPED_DEVICE_VAE_DEFAULT is True
    monkeypatch.setattr(qwen_mod, "_DEVICE_VAE_DEFAULT", device_default)
    _backend(monkeypatch, "neuron")
    stages, decoded = _record_stages(monkeypatch)
    args = _qwen_args(host_vae=host_vae, device_vae=device_vae, work_dir=str(tmp_path / "w"))
    QwenImageOrchestrator(args).generate()
    on_device = not host_vae and (device_vae or device_default)
    assert [s for s, _ in stages] == (["generate", "vae"] if on_device else ["generate"])
    assert len(decoded) == (0 if on_device else 1)


@pytest.mark.parametrize("mode", ["compile", "generate", "run"])
def test_host_vae_and_device_vae_are_exclusive(monkeypatch, tmp_path, capsys, mode):
    _backend(monkeypatch, "neuron")
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    monkeypatch.setattr(QwenImageOrchestrator, "download",
                        lambda self: pytest.fail("rejected before the download"))
    args = _qwen_args(host_vae=True, device_vae=True, work_dir=str(tmp_path / "w"))
    with pytest.raises(SystemExit) as exc:
        getattr(QwenImageOrchestrator(args), mode)()
    assert exc.value.code == 2
    assert "--host-vae and --device-vae are mutually exclusive" in capsys.readouterr().err


def test_device_vae_flag_is_parsed_and_forwarded_to_dp_workers():
    from difflet.cli.dp.router import worker_cli_args

    cli_main = importlib.import_module("difflet.cli.main")
    parser = cli_main._build_parser()
    for command in ("compile", "generate", "run"):
        args = parser.parse_args([command, "--model-id", _QWEN_ID, "--device-vae"])
        assert args.device_vae is True and args.host_vae is False
    args = parser.parse_args(["generate", "--model-id", _QWEN_ID, "--device-vae"])
    args.dp = 2
    assert "--device-vae" in worker_cli_args(args)


def _cpu_vae_stage(monkeypatch):
    """The vae stage's decoder on the CPU: the real ``NeuronWanVaeDecoderApplication`` and
    ``_DeviceQwenVaeDecoder``, with the device the stage asks for ("neuron") replaced by the
    CPU. Returns the applications the stage built."""
    import difflet.backends.neuron.wan.vae as vae_mod
    import difflet.models.qwen_image.neuron_application as na_mod

    real_app, real_adapter = vae_mod.NeuronWanVaeDecoderApplication, na_mod._DeviceQwenVaeDecoder
    built = []

    def app(*, device, **kwargs):
        assert device == "neuron"
        built.append(real_app(device="cpu", **kwargs))
        return built[-1]

    def adapter(app, *, device):
        assert device == "neuron"
        return real_adapter(app, device="cpu")

    monkeypatch.setattr(vae_mod, "NeuronWanVaeDecoderApplication", app)
    monkeypatch.setattr(na_mod, "_DeviceQwenVaeDecoder", adapter)
    return built


@pytest.fixture
def clean_neuron_mesh(monkeypatch, tmp_path):
    """No mesh, and a neuron runtime this process has not prepared, with its cache locations
    unset and DIFFLET_COMPILE_CACHE at ``tmp_path / "compile_cache"``: the vae stage prepares
    the runtime itself (``run_vae_stage``), which sets them; monkeypatch restores them."""
    from difflet.backends.neuron import runtime
    from difflet.backends.neuron.ops_impl import parallel_mesh

    monkeypatch.setattr(runtime, "_runtime_prepared", False)
    # No device here; the AoT venv's torch_neuronx has no native-runtime query at all.
    monkeypatch.setattr(runtime, "_neuron_runtime_initialized", lambda: False)
    for name in (*_CACHE_ENV, "NKI_ENABLE_TRACE_CACHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "compile_cache"))
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


def test_neuron_vae_stage_decodes_what_the_host_decode_does(monkeypatch, tmp_path,
                                                            clean_neuron_mesh):
    """The vae stage (the Wan decoder through the application's adapter, eager, here on the CPU,
    with --vae-dtype fp32: the default is bf16) against the host decode (diffusers'
    AutoencoderKLQwenImage, fp32) on the same
    packed latents of a non-square tiny image: the same unpack and latents_mean/std, the same
    [1, 3, H, W] image in [-1, 1], equal to fp32 rounding."""
    import numpy as np
    import torch
    from PIL import Image

    _backend(monkeypatch, "neuron")
    model = tmp_path / "model"
    _tiny_vae_snapshot(model)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(model))
    built = _cpu_vae_stage(monkeypatch)
    height, width = 64, 96
    work = tmp_path / "work"
    work.mkdir()
    packed = torch.randn(1, (height // 16) * (width // 16), 16,
                         generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    torch.save(packed, work / "latents.pt")
    args = _qwen_args(height=height, width=width, work_dir=str(work), vae_dtype="fp32",
                      output=str(tmp_path / "device.png"), cache_dir=str(tmp_path / "cache"))
    QwenImageOrchestrator(args)._run_stage_internal("vae", args)
    qwen_mod._decode_latents_host(str(work / "latents.pt"), str(model),
                                  str(tmp_path / "host.png"), height=height, width=width)

    [app] = built
    assert app.latent_shape == (1, 4, 1, height // 8, width // 8) and app.dtype is torch.float32
    assert app.load_report["missing"] == [] and app.warmup_shapes is None  # generate: no warm-up
    device_png = np.asarray(Image.open(tmp_path / "device.png"))
    host_png = np.asarray(Image.open(tmp_path / "host.png"))
    assert device_png.shape == (height, width, 3)
    assert np.abs(device_png.astype(int) - host_png.astype(int)).max() <= 1
    # The tensors themselves: the writer's .pt fallback keeps them.
    monkeypatch.setattr(qwen_mod, "_save_image_pil",
                        lambda image, path: (_ for _ in ()).throw(RuntimeError("no image")))
    QwenImageOrchestrator(args)._run_stage_internal("vae", args)
    qwen_mod._decode_latents_host(str(work / "latents.pt"), str(model),
                                  str(tmp_path / "host.png"), height=height, width=width)
    device = torch.load(tmp_path / "device.pt")
    host = torch.load(tmp_path / "host.pt")
    assert device.shape == host.shape == (1, 3, height, width)
    assert device.dtype == host.dtype == torch.float32
    torch.testing.assert_close(device, host, atol=1e-5, rtol=1e-5)


def test_neuron_vae_stage_compile_warms_up_and_writes_the_manifest(monkeypatch, tmp_path,
                                                                   clean_neuron_mesh):
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    _backend(monkeypatch, "neuron")
    model = tmp_path / "model"
    _tiny_vae_snapshot(model)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(model))
    built = _cpu_vae_stage(monkeypatch)
    args = _qwen_args(height=32, width=48, stage_mode="compile", work_dir=str(tmp_path / "w"),
                      cache_dir=str(tmp_path / "cache"))
    orch = QwenImageOrchestrator(args)
    orch._run_stage_internal("vae", args)
    [app] = built
    assert app.warmup_shapes == [(1, 4, 1, 4, 6)]  # the warm-up decode at the request shape
    # The stage prepared the neuron runtime itself: its NEFF cache is DIFFLET_COMPILE_CACHE's,
    # not torch-neuronx's /tmp/neff_cache (review C3).
    assert os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(
        tmp_path / "compile_cache" / "neuron" / "neff")
    assert has_valid_stage_manifest(orch._stage_compiled_dir("vae", args),
                                    orch._stage_cache_inputs("vae", args))
    assert not (tmp_path / "w").exists()


# ------------------------------------------- --vae-dtype (bf16 by default, fp32 on request)


def test_neuron_vae_stage_decodes_in_bf16_by_default(monkeypatch, tmp_path, clean_neuron_mesh):
    """The vae stage builds the decoder in bf16 (bf16 weights) by default and with --vae-dtype
    bf16, in fp32 with --vae-dtype fp32. The image comes back fp32 either way, the bf16 one
    within bf16's error of the fp32 stage's decode of the same latents (40 dB here on a random
    decoder; the device gate on the real one passed at 56.7 dB)."""
    import math

    import torch

    _backend(monkeypatch, "neuron")
    model = tmp_path / "model"
    _tiny_vae_snapshot(model)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(model))
    monkeypatch.setattr(qwen_mod, "_save_image_pil",
                        lambda image, path: (_ for _ in ()).throw(RuntimeError("keep the .pt")))
    built = _cpu_vae_stage(monkeypatch)
    height, width = 64, 96
    work = tmp_path / "work"
    work.mkdir()
    packed = torch.randn(1, (height // 16) * (width // 16), 16,
                         generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    torch.save(packed, work / "latents.pt")
    images = {}
    for flag in ("fp32", "bf16", None):
        from difflet.backends.neuron.ops_impl import parallel_mesh

        parallel_mesh.destroy_parallel_mesh()
        args = _qwen_args(height=height, width=width, work_dir=str(work), vae_dtype=flag,
                          output=str(tmp_path / f"{flag}.png"), cache_dir=str(tmp_path / "c"))
        QwenImageOrchestrator(args)._run_stage_internal("vae", args)
        images[flag] = torch.load(tmp_path / f"{flag}.pt")
    fp32_app, bf16_app, default_app = built
    assert fp32_app.dtype is torch.float32 and bf16_app.dtype is torch.bfloat16
    assert default_app.dtype is torch.bfloat16
    assert {p.dtype for p in bf16_app.module.parameters()} == {torch.bfloat16}
    torch.testing.assert_close(images[None], images["bf16"], atol=0, rtol=0)
    assert images["fp32"].dtype == images["bf16"].dtype == torch.float32
    mse = float(((images["bf16"].double() - images["fp32"].double()) ** 2).mean())
    assert 0 < mse and 10 * math.log10(4.0 / mse) >= 40.0


def test_vae_dtype_is_forwarded_only_when_given_and_keys_only_a_non_default_vae_stage(
        monkeypatch):
    """--vae-dtype reaches the stage processes only when given; the vae stage key gains
    ``vae_dtype`` only for the non-default precision (fp32), so a default stage, given
    --vae-dtype bf16 or not, keeps one key, as Wan's neuron vae stage does."""
    plain_args = _qwen_args(cache_dir="/c")
    assert "--vae-dtype" not in QwenImageOrchestrator(plain_args)._shared_cli_args("generate")
    fp32_args = _qwen_args(cache_dir="/c", vae_dtype="fp32")
    parts = QwenImageOrchestrator(fp32_args)._shared_cli_args("generate")
    assert parts[parts.index("--vae-dtype") + 1] == "fp32"

    def keyed(backend, args):
        _backend(monkeypatch, backend)
        orch = QwenImageOrchestrator(args)
        return {s: (orch._stage_cache_inputs(s, args), orch._stage_compiled_dir(s, args))
                for s in ("text", "generate", "vae")}

    for backend in (None, "trainium", "tpu"):  # the AoT paths do not read the flag
        assert keyed(backend, fp32_args) == keyed(backend, plain_args)
    neuron, fp32 = keyed("neuron", plain_args), keyed("neuron", fp32_args)
    assert keyed("neuron", _qwen_args(cache_dir="/c", vae_dtype="bf16")) == neuron
    assert fp32["vae"][0] == {**neuron["vae"][0], "vae_dtype": "float32"}
    assert fp32["vae"][1] != neuron["vae"][1]
    assert (fp32["text"], fp32["generate"]) == (neuron["text"], neuron["generate"])


@pytest.mark.parametrize("mode", ["compile", "generate", "run"])
def test_vae_dtype_with_host_vae_is_refused(monkeypatch, tmp_path, capsys, mode):
    _backend(monkeypatch, "neuron")
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    monkeypatch.setattr(QwenImageOrchestrator, "download",
                        lambda self: pytest.fail("rejected before the download"))
    args = _qwen_args(host_vae=True, vae_dtype="bf16", work_dir=str(tmp_path / "w"))
    with pytest.raises(SystemExit) as exc:
        getattr(QwenImageOrchestrator(args), mode)()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--vae-dtype sets the device decoder; --host-vae decodes on the host" in err


# ------------------------------------ --vae-exec-mode compile (phase 7's opt-in)


@pytest.mark.parametrize("vae_dtype", ["fp32", None])
def test_neuron_vae_stage_compiled_by_region_decodes_as_the_eager_stage(
        monkeypatch, tmp_path, clean_neuron_mesh, vae_dtype):
    """--vae-exec-mode compile: the vae stage builds the decoder compiled region by region (on
    the CPU: aot_eager) and writes the image the eager stage writes, to fp32 rounding (fp32) or
    within bf16 rounding (bf16: the first latent frame's convolutions sum fewer zero terms)."""
    import math

    import torch

    _backend(monkeypatch, "neuron")
    model = tmp_path / "model"
    _tiny_vae_snapshot(model)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(model))
    monkeypatch.setattr(qwen_mod, "_save_image_pil",
                        lambda image, path: (_ for _ in ()).throw(RuntimeError("keep the .pt")))
    built = _cpu_vae_stage(monkeypatch)
    height, width = 64, 96
    work = tmp_path / "work"
    work.mkdir()
    packed = torch.randn(1, (height // 16) * (width // 16), 16,
                         generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    torch.save(packed, work / "latents.pt")
    images = {}
    for exec_mode in (None, "compile"):
        from difflet.backends.neuron.ops_impl import parallel_mesh

        parallel_mesh.destroy_parallel_mesh()
        args = _qwen_args(height=height, width=width, work_dir=str(work), vae_dtype=vae_dtype,
                          vae_exec_mode=exec_mode, output=str(tmp_path / f"{exec_mode}.png"),
                          cache_dir=str(tmp_path / "c"))
        QwenImageOrchestrator(args)._run_stage_internal("vae", args)
        images[exec_mode] = torch.load(tmp_path / f"{exec_mode}.pt")
    eager_app, compiled_app = built
    assert (eager_app.exec_mode, compiled_app.exec_mode) == ("eager", "compile")
    eager, compiled = images[None], images["compile"]
    assert compiled.dtype == eager.dtype == torch.float32 and compiled.shape == eager.shape
    if vae_dtype == "fp32":
        torch.testing.assert_close(compiled, eager, atol=2e-5, rtol=1e-4)
    else:
        mse = float(((compiled.double() - eager.double()) ** 2).mean())
        assert mse == 0 or 10 * math.log10(4.0 / mse) >= 50.0


def test_vae_exec_mode_is_forwarded_only_when_given_and_keys_the_vae_stage(monkeypatch):
    plain_args = _qwen_args(cache_dir="/c")
    assert "--vae-exec-mode" not in QwenImageOrchestrator(plain_args)._shared_cli_args("generate")
    compiled_args = _qwen_args(cache_dir="/c", vae_exec_mode="compile")
    parts = QwenImageOrchestrator(compiled_args)._shared_cli_args("generate")
    assert parts[parts.index("--vae-exec-mode") + 1] == "compile"

    def keyed(backend, args):
        _backend(monkeypatch, backend)
        orch = QwenImageOrchestrator(args)
        return {s: (orch._stage_cache_inputs(s, args), orch._stage_compiled_dir(s, args))
                for s in ("text", "generate", "vae")}

    for backend in (None, "trainium", "tpu"):  # the AoT paths do not read the flag
        assert keyed(backend, compiled_args) == keyed(backend, plain_args)
    neuron, compiled = keyed("neuron", plain_args), keyed("neuron", compiled_args)
    assert keyed("neuron", _qwen_args(cache_dir="/c", vae_exec_mode="eager")) == neuron
    assert compiled["vae"][0] == {**neuron["vae"][0], "vae_exec_mode": "compile"}
    assert compiled["vae"][1] != neuron["vae"][1]
    assert (compiled["text"], compiled["generate"]) == (neuron["text"], neuron["generate"])


@pytest.mark.parametrize("mode", ["compile", "generate", "run"])
def test_vae_exec_mode_with_host_vae_is_refused(monkeypatch, tmp_path, capsys, mode):
    _backend(monkeypatch, "neuron")
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    monkeypatch.setattr(QwenImageOrchestrator, "download",
                        lambda self: pytest.fail("rejected before the download"))
    args = _qwen_args(host_vae=True, vae_exec_mode="compile", work_dir=str(tmp_path / "w"))
    with pytest.raises(SystemExit) as exc:
        getattr(QwenImageOrchestrator(args), mode)()
    assert exc.value.code == 2
    assert "--vae-exec-mode sets the device decoder" in capsys.readouterr().err


# ----------------------------------------------------------------- driver-side rejection

_REJECTED = [
    (dict(teacache_cadence=2), "--teacache-cadence"),
    (dict(teacache_online_delta=0.6), "--teacache-online-delta"),
    (dict(teacache_speedup=1.5, teacache_calibration="c.json"),
     "--teacache-speedup, --teacache-calibration"),
    (dict(shapes="1024x1024,768x1360"), "--shapes with more than one shape"),
    (dict(sp_enabled=True, cfg_parallel=True, shapes="1024x1024, 512x512 ,",
          teacache_cadence=3), "--teacache-cadence, --shapes with more than one shape"),
]


@pytest.mark.parametrize("overrides, named", _REJECTED)
@pytest.mark.parametrize("mode", ["compile", "generate", "run"])
def test_neuron_driver_rejects_unsupported_options_before_any_launch(monkeypatch, tmp_path,
                                                                     capsys, mode, overrides,
                                                                     named):
    _unset(monkeypatch, *_CACHE_ENV)
    _backend(monkeypatch, "neuron")
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    monkeypatch.setattr(qwen_mod, "_decode_latents_host",
                        lambda *a, **k: pytest.fail("nothing to decode"))
    monkeypatch.setattr(QwenImageOrchestrator, "download",
                        lambda self: pytest.fail("rejected before the download"))
    args = _qwen_args(cache_dir=str(tmp_path / "cache"), work_dir=str(tmp_path / "work"),
                      **overrides)

    with pytest.raises(SystemExit) as exc:
        getattr(QwenImageOrchestrator(args), mode)()

    assert exc.value.code == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.strip() == (
        f"Error: the neuron backend does not support {named} (it runs tensor, sequence, "
        "context and CFG parallelism at one static shape, and DP replicas, no TeaCache); "
        "nothing was launched.")
    assert "DIFFLET_COMPILE_CACHE" not in os.environ
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("overrides", [
    {}, dict(tp_degree=2, steps=50, guidance_scale=5.0, seed=7),
    dict(tp_degree=2, cp_degree=2, cp_mode="ring"), dict(tp_degree=1, cp_degree=4,
                                                         cp_mode="ulysses"),
    dict(shapes="1024x1024"), dict(height=768, width=1360, shapes="768x1360"),
    dict(host_vae=True), dict(sp_enabled=True), dict(sp_enabled=True, tp_degree=2),
])
def test_neuron_driver_lets_supported_options_through(monkeypatch, tmp_path, capsys, overrides):
    _backend(monkeypatch, "neuron")
    stages, decoded = _record_stages(monkeypatch)
    QwenImageOrchestrator(_qwen_args(work_dir=str(tmp_path / "w"), **overrides)).generate()
    assert [s for s, _ in stages] == ["generate"] and len(decoded) == 1
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("mode", ["compile", "generate", "run"])
def test_neuron_driver_refuses_sp_with_cp_before_any_launch(monkeypatch, tmp_path, capsys, mode):
    # Both shard the image sequence; every rank's NeuronQwenImageConfig would refuse the pair.
    _backend(monkeypatch, "neuron")
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    monkeypatch.setattr(QwenImageOrchestrator, "download",
                        lambda self: pytest.fail("rejected before the download"))
    args = _qwen_args(sp_enabled=True, tp_degree=2, cp_degree=2, work_dir=str(tmp_path / "w"))
    with pytest.raises(SystemExit) as exc:
        getattr(QwenImageOrchestrator(args), mode)()
    assert exc.value.code == 2
    assert capsys.readouterr().err.strip() == (
        "Error: --sp and --cp-degree > 1 are mutually exclusive for Qwen-Image (both shard the "
        "image sequence); nothing was launched.")


def test_sp_reaches_the_ranks_and_keys_its_own_artifact(neuron_stage, tmp_path):
    from difflet.cli import stage as stage_mod

    args = _qwen_args(sp_enabled=True, stage_mode="compile", cache_dir=str(tmp_path))
    orch = QwenImageOrchestrator(args)
    shared = orch._shared_cli_args(stage_mode="generate")
    assert "--sp" in shared
    stage_args, _ = stage_mod._build_stage_parser().parse_known_args(
        ["--orchestrator", _QWEN_ID, "--stage", "generate", *shared])
    assert stage_args.sp_enabled is True
    orch._stage_generate(args)
    assert neuron_stage.CALLS[0]["kwargs"]["parallel"].sp_enabled is True
    dense = _qwen_args(stage_mode="compile", cache_dir=str(tmp_path))
    assert orch._stage_compiled_dir("generate", args) != orch._stage_compiled_dir("generate", dense)


@pytest.mark.parametrize("backend", [None, "trainium", "tpu"])
@pytest.mark.parametrize("overrides", [o for o, _ in _REJECTED])
def test_aot_backends_skip_the_torchrun_option_check(monkeypatch, tmp_path, capsys, backend,
                                                     overrides):
    # Trainium/TPU support these (or reject them where they always did); the hook stays out.
    _backend(monkeypatch, backend)
    stages, _ = _record_stages(monkeypatch)
    monkeypatch.setattr(qwen_mod, "_require_request_shape_in_set", lambda args: None)
    QwenImageOrchestrator(_qwen_args(work_dir=str(tmp_path / "w"), **overrides)).generate()
    assert [s for s, _ in stages] == ["text", "generate", "vae"]
    assert capsys.readouterr().err == ""


def test_neuron_request_shape_outside_the_single_shape_fails_before_the_launch(monkeypatch,
                                                                               tmp_path):
    _backend(monkeypatch, "neuron")
    stages, _ = _record_stages(monkeypatch)
    args = _qwen_args(shapes="768x1360", work_dir=str(tmp_path / "w"))
    with pytest.raises(SystemExit, match="not in the compiled bucket set"):
        QwenImageOrchestrator(args).generate()
    assert stages == []


# ----------------------------------------------------------------- stage cache key


def test_cache_inputs_unchanged_for_aot_backends_and_distinct_for_neuron(monkeypatch):
    args = _qwen_args(cache_dir="/c")
    orch = QwenImageOrchestrator(args)
    stages = ("text", "generate", "vae")

    def keyed(backend, exec_mode=None):
        _backend(monkeypatch, backend)
        if exec_mode is not None:
            monkeypatch.setenv("DIFFLET_EXEC_MODE", exec_mode)
        return ({s: orch._stage_cache_inputs(s, args) for s in stages},
                {s: orch._stage_compiled_dir(s, args) for s in stages})

    base, base_dirs = keyed(None)
    assert set(base["generate"]) == {"component", "model_id", "tp", "cp", "cp_mode", "sp",
                                     "dtype", "text_seq_len", "shapes", "toolchain"}
    for backend in ("trainium", "tpu"):  # AoT: byte-identical keys and dirs
        assert keyed(backend) == (base, base_dirs)
    neuron, neuron_dirs = keyed("neuron")
    for stage in stages:
        assert neuron[stage] == {**base[stage], "backend": "neuron", "exec_mode": "compile"}
        assert neuron_dirs[stage] != base_dirs[stage]
        assert neuron_dirs[stage].parent == base_dirs[stage].parent
    # An omitted --exec-mode keys like the default (compile); eager differs.
    assert keyed("neuron", "compile") == (neuron, neuron_dirs)
    eager, eager_dirs = keyed("neuron", "eager")
    assert eager["generate"] == {**neuron["generate"], "exec_mode": "eager"}
    assert eager_dirs["generate"] != neuron_dirs["generate"]
    # CP degree and mode, SP and CFG parallelism, each alone: test_neuron_stage_cache_keys.py.


# ----------------------------------------------------------------- the torchrun command line


def _record_launches(monkeypatch) -> list[tuple[list[str], dict]]:
    """The real run_stage up to the launch: each torchrun command line and environment."""
    launches: list = []
    monkeypatch.setattr(qwen_mod.runner, "_run_torchrun",
                        lambda cmd, env: launches.append((list(cmd), dict(env))))
    monkeypatch.setattr(qwen_mod, "_decode_latents_host", lambda *a, **k: None)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: "/fake/model")
    return launches


@pytest.mark.parametrize("mode", ["compile", "generate"])
def test_neuron_torchrun_command_line(monkeypatch, tmp_path, capsys, mode):
    from difflet.cli import stage as stage_mod

    _unset(monkeypatch, *_CACHE_ENV)
    _backend(monkeypatch, "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "eager")
    monkeypatch.chdir(tmp_path)
    launches = _record_launches(monkeypatch)
    work = tmp_path / "work"
    args = _qwen_args(tp_degree=2, cp_degree=2, cp_mode="ulysses", cache_dir="cache",
                      work_dir=str(work), revision=_REV, steps=20, guidance_scale=3.5, seed=7,
                      prompt="a fox", output=str(tmp_path / "fox.png"), shapes="1024x1024")
    getattr(QwenImageOrchestrator(args), mode)()

    [(cmd, env)] = launches
    head = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node",
            "4", "-m", "difflet.cli.stage", "--orchestrator", _QWEN_ID, "--stage", "generate"]
    assert cmd[:len(head)] == head
    stage_args, _ = stage_mod._build_stage_parser().parse_known_args(cmd[len(head) - 4:])
    assert (stage_args.stage, stage_args.stage_mode) == ("generate", mode)
    assert (stage_args.tp_degree, stage_args.cp_degree, stage_args.cp_mode) == (2, 2, "ulysses")
    assert (stage_args.height, stage_args.width, stage_args.shapes) == (1024, 1024, "1024x1024")
    assert (stage_args.steps, stage_args.guidance_scale, stage_args.seed) == (20, 3.5, 7)
    assert (stage_args.prompt, stage_args.revision, stage_args.cache_dir) == ("a fox", _REV,
                                                                              "cache")
    assert stage_args.work_dir == (str(work) if mode == "generate" else
                                   str(Path.home() / ".cache" / "difflet" / "work" / "qwen-image"))
    # The backend and exec mode reach the ranks through the environment; --cache-dir roots the
    # compile cache; torchrun owns the rank variables and core binding.
    root = str(tmp_path / "cache")
    assert (env["DIFFLET_BACKEND"], env["DIFFLET_EXEC_MODE"]) == ("neuron", "eager")
    assert env["DIFFLET_COMPILE_CACHE"] == root
    assert env["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "cache" / "neuron" / "neff")
    assert "RANK" not in env and "WORLD_SIZE" not in env
    assert capsys.readouterr().out.count(_CACHE_NOTE + root) == 1


def test_neuron_exported_compile_cache_wins_over_cache_dir(monkeypatch, tmp_path, capsys):
    _unset(monkeypatch, *_CACHE_ENV)
    explicit = str(tmp_path / "explicit")
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", explicit)
    _backend(monkeypatch, "neuron")
    launches = _record_launches(monkeypatch)
    args = _qwen_args(cache_dir=str(tmp_path / "cache"), work_dir=str(tmp_path / "work"))
    QwenImageOrchestrator(args).generate()
    [(_, env)] = launches
    assert env["DIFFLET_COMPILE_CACHE"] == explicit
    assert _CACHE_NOTE not in capsys.readouterr().out


@pytest.mark.parametrize("backend", [None, "trainium", "tpu"])
def test_aot_backends_leave_the_compile_cache_alone(monkeypatch, tmp_path, capsys, backend):
    _unset(monkeypatch, *_CACHE_ENV)
    _backend(monkeypatch, backend)
    stages, _ = _record_stages(monkeypatch)
    QwenImageOrchestrator(_qwen_args(cache_dir=str(tmp_path / "c"),
                                     work_dir=str(tmp_path / "w"))).generate()
    assert len(stages) == 3 and "DIFFLET_COMPILE_CACHE" not in os.environ
    assert _CACHE_NOTE not in capsys.readouterr().out


# ----------------------------------------------------------------- the generate stage (ranks)


@pytest.fixture
def neuron_stage(monkeypatch):
    """One process, DIFFLET_BACKEND=neuron, no process group (rank 0): from_pretrained is the
    CPU fake. (The Trainium application module imports neuronx_distributed, which this venv
    lacks: the neuron path importing it would fail every test here.)"""
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    _backend(monkeypatch, "neuron")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(fake_pipeline, "CALLS", [])
    monkeypatch.setattr(DiffletPipeline, "from_pretrained",
                        classmethod(fake_pipeline.fake_from_pretrained))
    return fake_pipeline


def _fake_latents(seed: int):
    import torch

    return torch.randn(fake_pipeline.LATENT_SHAPE, generator=torch.Generator().manual_seed(seed))


def test_stage_generate_neuron_routes_through_pipeline(neuron_stage, monkeypatch, tmp_path,
                                                       capsys):
    import torch

    monkeypatch.setattr(QwenImageOrchestrator, "_require_stage_artifact",
                        lambda *a: pytest.fail("non-AoT: there is no stage artifact to require"))
    cache = tmp_path / "cache"
    args = _qwen_args(revision=_REV, work_dir=str(tmp_path), cache_dir=str(cache),
                      prompt="a fox", steps=50, guidance_scale=5.0, seed=42,
                      tp_degree=2, cp_degree=2, cp_mode="ring")
    QwenImageOrchestrator(args)._stage_generate(args)

    [call] = neuron_stage.CALLS
    kw = call["kwargs"]
    assert call["model_id"] == _QWEN_ID
    assert (kw["backend"], kw["revision"], kw["local_files_only"]) == ("neuron", _REV, True)
    assert kw["application_kwargs"] == _NEURON_APP_KWARGS
    assert (kw["height"], kw["width"], kw["shapes"], kw["load"]) == (1024, 1024, None, False)
    assert kw["dtype"] is torch.bfloat16 and kw["compile_cache_dir"] == str(cache)
    assert "model_path_override" not in kw
    p = kw["parallel"]
    assert (p.tp_degree, p.cp_degree, p.cp_mode, p.cfg_parallel_enabled, p.sp_enabled) == (
        2, 2, "ring", False, False)
    app = call["pipeline"].app
    # The warm-up runs in load(), before the step loop (the Trainium stage skips it: AoT NEFFs).
    assert [c["skip_warmup"] for c in app.load_calls] == [False]
    assert app.calls == [{"prompt": "a fox", "num_inference_steps": 50, "guidance_scale": 5.0,
                          "output_type": "latent", "seed": 42}]
    assert torch.equal(torch.load(tmp_path / "latents.pt"), _fake_latents(42))
    out = capsys.readouterr().out
    assert out.count(f"-> {tmp_path / 'latents.pt'}") == 1
    assert "[qwen_image] neuron rank 0/1: exec_mode=compile compiled_blocks=40" in out
    assert not cache.exists()  # generate writes no stage manifest and no AoT artifact


def test_stage_generate_neuron_compile_warms_up_and_writes_the_manifest(neuron_stage, tmp_path):
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    args = _qwen_args(stage_mode="compile", work_dir=str(tmp_path / "w"),
                      cache_dir=str(tmp_path / "cache"))
    orch = QwenImageOrchestrator(args)
    orch._stage_generate(args)

    [call] = neuron_stage.CALLS
    app = call["pipeline"].app
    assert [c["skip_warmup"] for c in app.load_calls] == [False] and app.calls == []
    compiled_dir = orch._stage_compiled_dir("generate", args)
    inputs = orch._stage_cache_inputs("generate", args)
    assert has_valid_stage_manifest(compiled_dir, inputs)
    assert (inputs["backend"], inputs["exec_mode"]) == ("neuron", "compile")
    assert not (tmp_path / "w").exists()


@pytest.mark.parametrize("mode", ["generate", "compile"])
def test_stage_generate_neuron_writes_files_on_rank0_only(neuron_stage, monkeypatch, tmp_path,
                                                          capsys, mode):
    from difflet.backends.neuron.core import distributed

    monkeypatch.setattr(distributed, "is_rank0", lambda: False)
    work, cache = tmp_path / "work", tmp_path / "cache"
    work.mkdir()
    args = _qwen_args(stage_mode=mode, work_dir=str(work), cache_dir=str(cache))
    QwenImageOrchestrator(args)._stage_generate(args)

    [call] = neuron_stage.CALLS
    assert len(call["pipeline"].app.calls) == (1 if mode == "generate" else 0)  # every rank runs
    assert list(work.iterdir()) == [] and not cache.exists()
    assert "packed latents" not in capsys.readouterr().out


def _queue(tmp_path, prompts, *, workers=1):
    """A DP request queue (the router's manifest), round robin over ``workers``."""
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    requests = tmp_path / "requests"
    write_manifest([RequestSpec(index=i, prompt=prompt, output=str(tmp_path / f"o{i}.png"),
                                seed=10 + i, assigned_worker=i % workers)
                    for i, prompt in enumerate(prompts)], requests)
    return requests


def test_stage_generate_neuron_runs_a_request_batch(neuron_stage, tmp_path):
    # A DP worker's queue: one load, then every request this worker claims, in order, each
    # with its own prompt and seed and its own latents file.
    requests = _queue(tmp_path, ["a fox", "a cat"])
    args = _qwen_args(work_dir=str(tmp_path), requests_dir=str(requests), worker_index=0,
                      dp_schedule="round_robin")
    QwenImageOrchestrator(args)._stage_generate(args)

    [call] = neuron_stage.CALLS
    app = call["pipeline"].app
    assert [(c["prompt"], c["seed"]) for c in app.calls] == [("a fox", 10), ("a cat", 11)]
    assert len(app.load_calls) == 1  # loaded once for the whole queue
    assert (tmp_path / "latents_req0000.pt").exists() and (tmp_path / "latents_req0001.pt").exists()
    assert sorted(p.name for p in requests.glob("*.claim")) == ["req_0000.claim", "req_0001.claim"]
    assert list(requests.glob("*.done")) == []  # done once the image is written (the decode)


def test_dp_replica_ranks_run_the_same_requests_rank0_claims_and_writes(tmp_path):
    """A two-rank replica of a DP worker on gloo (the real stage, the fake pipeline): rank 0
    alone claims, every rank runs every request of the queue in order, rank 0 alone saves."""
    from tests.unit.backends._neuron_gloo import run_ranks

    requests = _queue(tmp_path, ["p0", "p1", "p2"])
    results = run_ranks(fake_pipeline.qwen_replica_worker, str(tmp_path), str(requests),
                        world_size=2)
    for result in results:
        assert result["prompts"] == ["p0", "p1", "p2"] and result["seeds"] == [10, 11, 12]
        assert result["error"] is None and result["loads"] == 1
    assert results[0]["saved"] == ["latents_req0000.pt", "latents_req0001.pt",
                                   "latents_req0002.pt"]
    assert results[1]["saved"] == []  # one writer
    claims = sorted(requests.glob("*.claim"))
    assert [p.name for p in claims] == ["req_0000.claim", "req_0001.claim", "req_0002.claim"]
    assert {p.read_text() for p in claims} == {"0"}  # claimed once, by worker 0's rank 0


def test_dp_replica_stops_together_on_a_failed_request(tmp_path):
    from tests.unit.backends._neuron_gloo import run_ranks

    requests = _queue(tmp_path, ["p0", fake_pipeline.FAIL_PROMPT, "p2"])
    results = run_ranks(fake_pipeline.qwen_replica_worker, str(tmp_path), str(requests),
                        world_size=2)
    for result in results:
        assert result["prompts"] == ["p0", fake_pipeline.FAIL_PROMPT]
        assert "fake failure" in result["error"]  # every rank stops, none waits in a collective
    assert results[0]["saved"] == ["latents_req0000.pt"] and results[1]["saved"] == []
    assert sorted(p.name for p in requests.glob("*.failed")) == ["req_0001.failed"]
    assert not (requests / "req_0002.claim").exists()  # left for the router's summary


@pytest.mark.parametrize("overrides, parallel, app_extra", [
    (dict(cfg_parallel=True), (True, False), {}),
    (dict(sp_enabled=True), (False, True), {}),
    (dict(teacache_cadence=2), (False, False), {"teacache_cadence": 2}),
    (dict(teacache_online_delta=0.6), (False, False), {"teacache_online_delta_alpha": 0.6}),
    (dict(teacache_speedup=1.5, teacache_calibration="c.json"), (False, False),
     {"teacache_speedup": 1.5, "teacache_calibration_path": "c.json"}),
])
def test_stage_generate_neuron_hands_unsupported_options_on(neuron_stage, tmp_path, overrides,
                                                            parallel, app_extra):
    # Not dropped silently: the backend rejects cfg/sp, the application TeaCache.
    args = _qwen_args(stage_mode="compile", cache_dir=str(tmp_path), **overrides)
    QwenImageOrchestrator(args)._stage_generate(args)
    kw = neuron_stage.CALLS[0]["kwargs"]
    p = kw["parallel"]
    assert (p.cfg_parallel_enabled, p.sp_enabled) == parallel
    assert kw["application_kwargs"] == {**_NEURON_APP_KWARGS, **app_extra}


def test_stage_generate_neuron_binds_a_local_model_path(neuron_stage, tmp_path):
    model = tmp_path / "snapshot"
    model.mkdir()
    args = _qwen_args(stage_mode="compile", cache_dir=str(tmp_path / "cache"),
                      model_path=str(model), shapes="1024x1024")
    orch = QwenImageOrchestrator(args)
    orch._stage_generate(args)
    kw = neuron_stage.CALLS[0]["kwargs"]
    assert kw["model_path_override"] == str(model.resolve())
    assert kw["resolved_source_id"] == "local"
    assert kw["compiled_path_override"] == str(orch._stage_compiled_dir("generate", args))
    assert [tuple(s) for s in kw["shapes"]] == [(1024, 1024, None)]


# ----------------------------------------------------------------- DP workers: the decode


@pytest.mark.parametrize("device_vae", [False, True])
def test_dp_worker_decodes_each_claimed_request_and_marks_it_done(monkeypatch, tmp_path,
                                                                  device_vae):
    """A DP worker's driver after its generate launch: the host decode runs once per request
    the replica claimed and did not fail, from its own latents file to its own output, and
    marks it done. With --device-vae the vae stage does that instead (its
    request_scope(final=True))."""
    from difflet.cli.dp import claims

    _backend(monkeypatch, "neuron")
    stages, decoded = _record_stages(monkeypatch)
    requests = _queue(tmp_path, ["p0", "p1", "p2", "p3"], workers=2)
    work = tmp_path / "worker_0"

    def generate_launch(orch, stage, **kw):
        stages.append((stage, kw["num_cores"]))
        if stage == "generate":  # what worker 0's replica leaves: 0 and 2 claimed, 2 failed
            for index in (0, 2):
                claims._try_claim(requests, index, 0)
            claims.mark_failed(requests, 2, "boom")

    monkeypatch.setattr(qwen_mod.runner, "run_stage", generate_launch)
    args = _qwen_args(work_dir=str(work), requests_dir=str(requests), worker_index=0,
                      dp_schedule="round_robin", device_vae=device_vae, keep_work_dir=True)
    QwenImageOrchestrator(args).generate()

    if device_vae:
        assert stages == [("generate", 4), ("vae", 1)] and decoded == []
        return
    assert stages == [("generate", 4)]
    assert decoded == [((str(work / "latents_req0000.pt"), "/fake/model",
                         str(tmp_path / "o0.png")), {"height": 1024, "width": 1024})]
    assert sorted(p.name for p in requests.glob("*.done")) == ["req_0000.done"]
    assert claims.summarize(requests).failed.keys() == {2}


def test_dp_worker_marks_a_failed_decode_and_decodes_the_next(monkeypatch, tmp_path):
    from difflet.cli.dp import claims

    _backend(monkeypatch, "neuron")
    _record_stages(monkeypatch)
    requests = _queue(tmp_path, ["p0", "p1"])
    for index in (0, 1):
        claims._try_claim(requests, index, 0)
    seen = []

    def decode(latents_path, model_dir, output_path, *, height, width):
        seen.append(Path(output_path).name)
        if output_path.endswith("o0.png"):
            raise RuntimeError("decode failed")

    monkeypatch.setattr(qwen_mod, "_decode_latents_host", decode)
    args = _qwen_args(work_dir=str(tmp_path / "w"), requests_dir=str(requests), worker_index=0,
                      dp_schedule="round_robin")
    QwenImageOrchestrator(args).generate()
    assert seen == ["o0.png", "o1.png"]
    summary = claims.summarize(requests)
    assert summary.done == [1] and "decode failed" in summary.failed[0]


def test_dp_workers_get_the_qwen_flags_from_the_router():
    from difflet.cli import stage as stage_mod
    from difflet.cli.dp.router import worker_cli_args

    cli_main = importlib.import_module("difflet.cli.main")
    args = cli_main._build_parser().parse_args([
        "generate", "--model-id", _QWEN_ID, "--backend", "neuron", "--exec-mode", "eager",
        "--dp", "2", "--tp-degree", "2", "--sp", "--device-vae", "--steps", "8",
        "--prompt", "a fox", "--output", "/tmp/o.png",
    ])
    forwarded = worker_cli_args(args)
    worker = cli_main._build_parser().parse_args(["generate", *forwarded])
    assert (worker.tp_degree, worker.sp_enabled, worker.device_vae) == (2, True, True)
    assert (worker.exec_mode, worker.steps) == ("eager", 8)
    stage_args, _ = stage_mod._build_stage_parser().parse_known_args(
        ["--orchestrator", _QWEN_ID, "--stage", "generate", *QwenImageOrchestrator(
            _qwen_args(sp_enabled=True, requests_dir="/rd", worker_index=1,
                       dp_schedule="round_robin"))._shared_cli_args("generate", "/w")])
    assert (stage_args.requests_dir, stage_args.worker_index) == ("/rd", 1)


# ----------------------------------------------------------------- the DiT's HBM budget


def _header_only_dit(root: Path, numel: int) -> Path:
    """A transformer directory whose safetensors header claims ``numel`` elements (no data: the
    driver's estimate reads only headers)."""
    import json
    import struct

    header = json.dumps({"w": {"dtype": "BF16", "shape": [numel], "data_offsets": [0, 0]}})
    (root / "transformer").mkdir(parents=True)
    with open(root / "transformer" / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header.encode())
    return root


_REAL_DIT_NUMEL = 20_430_401_088  # Qwen-Image's transformer: 38.05 GiB in bf16


@pytest.mark.parametrize("mode", ["compile", "generate", "run"])
@pytest.mark.parametrize("worker", [False, True])
def test_a_dit_shard_that_cannot_fit_is_refused_before_any_launch(monkeypatch, tmp_path, capsys,
                                                                  mode, worker):
    # --dp 4 --tp-degree 1 (each worker), or a plain tp1: 38 GiB per core against 24 GiB.
    _backend(monkeypatch, "neuron")
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    model = _header_only_dit(tmp_path / "model", _REAL_DIT_NUMEL)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(model))
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    monkeypatch.setattr(QwenImageOrchestrator, "download", lambda self: None)
    extra = (dict(requests_dir=str(tmp_path / "rd"), worker_index=3, dp_schedule="round_robin")
             if worker else {})
    args = _qwen_args(tp_degree=1, work_dir=str(tmp_path / "w"), **extra)
    with pytest.raises(SystemExit) as exc:
        getattr(QwenImageOrchestrator(args), mode)()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "the Qwen-Image DiT needs at least 38.1 GiB of bf16 weights per core at tp=1" in err
    assert "use --tp-degree 2 or more; nothing was launched." in err


def test_run_with_dp4_tp1_is_refused_before_the_router_starts(monkeypatch, tmp_path, capsys):
    """``difflet run --dp 4 --tp-degree 1``: the router compiles first, and that compile's
    budget check exits before a worker or a torchrun rank starts."""
    from difflet.cli.dp import router

    cli_main = importlib.import_module("difflet.cli.main")
    _unset(monkeypatch, *_CACHE_ENV, "NEURON_RT_NUM_CORES")
    _backend(monkeypatch, None)
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    model = _header_only_dit(tmp_path / "model", _REAL_DIT_NUMEL)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(model))
    monkeypatch.setattr(QwenImageOrchestrator, "download", lambda self: None)
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    monkeypatch.setattr(router, "run_router", lambda *a, **k: pytest.fail("no router"))
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["run", "--model-id", _QWEN_ID, "--backend", "neuron", "--dp", "4",
                       "--tp-degree", "1", "--total-cores", "4", "--prompt", "a fox",
                       "--output", str(tmp_path / "o.png"), "--work-dir", str(tmp_path / "w")])
    assert exc.value.code == 2
    assert "38.1 GiB of bf16 weights per core at tp=1" in capsys.readouterr().err


@pytest.mark.parametrize("tp", [2, 4])
def test_dp2tp2_and_tp4_dit_shards_pass_the_budget(monkeypatch, tmp_path, capsys, tp):
    _backend(monkeypatch, "neuron")
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    model = _header_only_dit(tmp_path / "model", _REAL_DIT_NUMEL)
    stages, decoded = _record_stages(monkeypatch)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(model))
    QwenImageOrchestrator(_qwen_args(tp_degree=tp, work_dir=str(tmp_path / "w"))).generate()
    assert stages == [("generate", tp)] and capsys.readouterr().err == ""


def test_weights_not_local_leave_the_budget_to_the_application(monkeypatch, tmp_path):
    _backend(monkeypatch, "neuron")
    stages, _ = _record_stages(monkeypatch)

    def not_local(*a, **k):
        raise FileNotFoundError("not in the cache")

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", not_local)
    QwenImageOrchestrator(_qwen_args(tp_degree=1)).compile()
    assert [s for s, _ in stages] == ["generate"]


@pytest.mark.parametrize("backend", [None, "trainium", "tpu"])
def test_aot_backends_skip_the_dit_budget(monkeypatch, tmp_path, backend):
    _backend(monkeypatch, backend)
    stages, _ = _record_stages(monkeypatch)
    monkeypatch.setattr(QwenImageOrchestrator, "_model_dir",
                        staticmethod(lambda *a: pytest.fail("not consulted off neuron")))
    QwenImageOrchestrator(_qwen_args(tp_degree=1)).compile()
    assert [s for s, _ in stages] == ["text", "generate", "vae"]


# ----------------------------------------------------------------- host VAE decode


def _tiny_vae_snapshot(root: Path):
    """A seeded tiny AutoencoderKLQwenImage (z_dim 4, 8x spatial scale) under ``root/vae``."""
    import torch
    from diffusers import AutoencoderKLQwenImage

    from tests.unit.backends._neuron_qwen_app_toy import TINY_VAE_CONFIG

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        vae = AutoencoderKLQwenImage(**TINY_VAE_CONFIG).eval()
    vae.save_pretrained(str(root / "vae"), safe_serialization=True)
    return vae


def _diffusers_decode(vae, packed, height: int, width: int):
    """diffusers QwenImagePipeline's decode of packed latents (output_type='pt', before
    postprocessing), in fp32."""
    import torch
    from diffusers import QwenImagePipeline

    latents = QwenImagePipeline._unpack_latents(packed, height, width, vae_scale_factor=8)
    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)
    with torch.no_grad():
        return vae.decode(latents / std + mean, return_dict=False)[0][:, :, 0]


def test_host_decode_matches_diffusers_qwen_image_pipeline(tmp_path, monkeypatch):
    import torch

    vae = _tiny_vae_snapshot(tmp_path / "model")
    height, width = 64, 96  # a non-square grid: 4 x 6 packed tokens
    packed = torch.randn(1, (height // 16) * (width // 16), 16,
                         generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    torch.save(packed, tmp_path / "latents.pt")
    # Keep the tensor: a writer that fails falls back to <output>.pt.
    monkeypatch.setattr(qwen_mod, "_save_image_pil",
                        lambda image, path: (_ for _ in ()).throw(RuntimeError("no image")))
    out = tmp_path / "out" / "img.png"

    qwen_mod._decode_latents_host(str(tmp_path / "latents.pt"), str(tmp_path / "model"),
                                  str(out), height=height, width=width)

    image = torch.load(out.with_suffix(".pt"))
    assert image.shape == (1, 3, height, width) and image.dtype == torch.float32
    torch.testing.assert_close(image, _diffusers_decode(vae, packed.float(), height, width),
                               rtol=1e-5, atol=1e-5)


def test_host_decode_writes_the_png_torchvision_would(tmp_path):
    import numpy as np
    import torch
    from PIL import Image

    _tiny_vae_snapshot(tmp_path / "model")
    packed = torch.randn(1, 4, 16, generator=torch.Generator().manual_seed(5))
    torch.save(packed, tmp_path / "latents.pt")
    out = tmp_path / "img.png"
    qwen_mod._decode_latents_host(str(tmp_path / "latents.pt"), str(tmp_path / "model"),
                                  str(out), height=32, width=32)

    with Image.open(out) as png:
        assert png.size == (32, 32) and png.mode == "RGB"
        pixels = np.asarray(png)
    # torchvision.utils.save_image's quantisation: x * 255 + 0.5, clamped, truncated.
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(str(tmp_path / "model" / "vae")).eval()
    ref = (_diffusers_decode(vae, packed, 32, 32)[0] * 0.5 + 0.5).clamp(0, 1)
    expected = ref.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
    assert np.abs(pixels.astype(int) - expected.astype(int)).max() <= 1


def test_host_decode_rejects_a_latent_of_another_shape(tmp_path):
    import torch

    _tiny_vae_snapshot(tmp_path / "model")
    torch.save(torch.zeros(1, 24, 16), tmp_path / "latents.pt")
    with pytest.raises(ValueError, match="length 24 does not match request grid 6x6"):
        qwen_mod._decode_latents_host(str(tmp_path / "latents.pt"), str(tmp_path / "model"),
                                      str(tmp_path / "x.png"), height=96, width=96)


# ----------------------------------------------------------------- CPU torchrun (2 ranks, gloo)
# The real run_stage -> torchrun -> difflet.cli.stage -> QwenImageOrchestrator, with
# from_pretrained replaced in every rank (FAKE_PIPELINE_ENV).

_REAL_POPEN = subprocess.Popen
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _bounded_popen(limit: float):
    """subprocess.Popen whose first untimed wait() gives up after ``limit`` s (TimeoutExpired):
    run_stage then sends torchrun SIGTERM and waits for it, so a hung launch fails the test."""

    class BoundedPopen(_REAL_POPEN):
        _bounded = False

        def wait(self, timeout=None):
            if timeout is None and not self._bounded:
                self._bounded = True
                timeout = limit
            return super().wait(timeout=timeout)

    return BoundedPopen


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"  # a zombie has exited


@pytest.fixture
def neuron_torchrun(monkeypatch, tmp_path):
    site, records = tmp_path / "site", tmp_path / "records"
    fake_pipeline.write_sitecustomize(site)
    records.mkdir()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        filter(None, [str(site), str(_REPO_ROOT), os.environ.get("PYTHONPATH")])))
    monkeypatch.setenv(fake_pipeline.FAKE_PIPELINE_ENV, fake_pipeline.MODULE)
    monkeypatch.setenv(fake_pipeline.FAKE_RECORDS_ENV, str(records))
    _backend(monkeypatch, "neuron")
    monkeypatch.setenv("DIFFLET_DISABLE_PREWARM", "1")
    # run_stage exports the neuron cache locations under it to the ranks, and their
    # `import torch` (torch_neuronx autoload) creates the HLO one: keep it in tmp_path.
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path / "difflet-cache"))
    monkeypatch.setattr(subprocess, "Popen", _bounded_popen(300))
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(tmp_path / "model"))
    return records


def test_generate_neuron_cpu_torchrun_one_writer_and_decode_after_every_rank_exited(
    neuron_torchrun, monkeypatch, tmp_path, capfd,
):
    import torch

    seen = {}

    def decode(latents_path, model_dir, output_path, *, height, width):
        seen["records"] = fake_pipeline.read_rank_records(neuron_torchrun)
        seen["alive"] = [_alive(r["pid"]) for r in seen["records"]]
        seen["latents"] = torch.load(latents_path)
        seen["decode"] = (model_dir, output_path, height, width)

    monkeypatch.setattr(qwen_mod, "_decode_latents_host", decode)
    work, cache = tmp_path / "work", tmp_path / "cache"
    args = _qwen_args(revision=_REV, tp_degree=2, work_dir=str(work), keep_work_dir=True,
                      cache_dir=str(cache), steps=3, guidance_scale=5.0, prompt="a fox",
                      output=str(tmp_path / "fox.png"))
    QwenImageOrchestrator(args).generate()

    records = seen["records"]
    assert [r["rank"] for r in records] == [0, 1] and seen["alive"] == [False, False]
    writes = [w for r in records for w in r["writes"]]
    assert writes == [{"kind": "torch.save", "path": str(work / "latents.pt"), "rank": 0}]
    for record in records:
        [call] = record["from_pretrained"]
        kw = call["kwargs"]
        assert (call["model_id"], kw["backend"], kw["revision"]) == (_QWEN_ID, "neuron", _REV)
        assert kw["parallel"]["tp_degree"] == 2 and record["world_size"] == 2
        assert kw["application_kwargs"] == _NEURON_APP_KWARGS
        assert [c["skip_warmup"] for c in record["pipeline_load_calls"]] == [False]
        assert [(c["prompt"], c["seed"], c["num_inference_steps"], c["guidance_scale"])
                for c in record["app_calls"]] == [("a fox", 42, 3, 5.0)]
    assert torch.equal(seen["latents"], _fake_latents(42))
    assert seen["decode"] == (str(tmp_path / "model"), str(tmp_path / "fox.png"), 1024, 1024)
    log = "".join(capfd.readouterr())
    assert log.count("[generate] packed latents") == 1
    assert log.count("compiled_blocks=40 unwarmed_shapes=[]") == 2  # one line per rank
    assert not cache.exists()  # generate: no stage manifest, no AoT artifact


def test_compile_neuron_cpu_torchrun_rank0_writes_the_manifest(neuron_torchrun, tmp_path):
    from difflet.cli.orchestrators.base import has_valid_stage_manifest

    args = _qwen_args(tp_degree=2, cache_dir=str(tmp_path / "cache"))
    orch = QwenImageOrchestrator(args)
    orch.compile()

    records = fake_pipeline.read_rank_records(neuron_torchrun)
    assert [r["rank"] for r in records] == [0, 1]
    compiled_dir = orch._stage_compiled_dir("generate", args)
    writes = [w for r in records for w in r["writes"]]
    assert writes == [{"kind": "manifest", "path": str(compiled_dir), "rank": 0}]
    assert has_valid_stage_manifest(compiled_dir, orch._stage_cache_inputs("generate", args))
    for record in records:
        assert [c["skip_warmup"] for c in record["pipeline_load_calls"]] == [False]
        assert record["app_calls"] == []


def test_dp_worker_cpu_torchrun_drains_the_queue_and_decodes_each_request(neuron_torchrun,
                                                                          monkeypatch, tmp_path):
    """A DP worker end to end: the real run_stage -> torchrun (2 gloo ranks) -> the generate
    stage drains the queue (rank 0 claims, both ranks run each request, rank 0 alone saves),
    then the driver decodes each request and marks it done."""
    from difflet.cli.dp import claims

    decoded = []
    monkeypatch.setattr(qwen_mod, "_decode_latents_host",
                        lambda latents_path, model_dir, output_path, *, height, width:
                        decoded.append((Path(latents_path).name, Path(output_path).name)))
    requests = _queue(tmp_path, ["p0", "p1", "p2"])
    work = tmp_path / "worker_0"
    args = _qwen_args(tp_degree=2, work_dir=str(work), keep_work_dir=True,
                      cache_dir=str(tmp_path / "cache"), requests_dir=str(requests),
                      worker_index=0, dp_schedule="round_robin")
    QwenImageOrchestrator(args).generate()

    records = fake_pipeline.read_rank_records(neuron_torchrun)
    assert [r["rank"] for r in records] == [0, 1]
    for record in records:
        assert [(c["prompt"], c["seed"]) for c in record["app_calls"]] == [
            ("p0", 10), ("p1", 11), ("p2", 12)]
        assert [c["skip_warmup"] for c in record["pipeline_load_calls"]] == [False]
    writes = [w for r in records for w in r["writes"]]
    assert writes == [{"kind": "torch.save", "path": str(work / f"latents_req000{i}.pt"),
                       "rank": 0} for i in range(3)]
    assert decoded == [(f"latents_req000{i}.pt", f"o{i}.png") for i in range(3)]
    assert claims.summarize(requests).done == [0, 1, 2]


# ----------------------------------------------------------------- `difflet generate` end to end


def test_main_generate_backend_neuron_reaches_the_torchrun_launch(monkeypatch, tmp_path):
    """``difflet generate --model-id Qwen/Qwen-Image --backend neuron --exec-mode eager ...``:
    main's checks pass (the registry lists neuron for Qwen-Image), --backend/--exec-mode are
    exported for the ranks, and the orchestrator launches one torchrun generate stage."""
    cli_main = importlib.import_module("difflet.cli.main")  # the package exports main()

    _unset(monkeypatch, *_CACHE_ENV, "NEURON_RT_NUM_CORES")
    _backend(monkeypatch, None)
    launches = _record_launches(monkeypatch)
    out = tmp_path / "out.png"
    cli_main.main(["generate", "--model-id", _QWEN_ID, "--backend", "neuron",
                   "--exec-mode", "eager", "--tp-degree", "2", "--cp-degree", "2",
                   "--cp-mode", "ring", "--total-cores", "4", "--prompt", "a fox",
                   "--output", str(out), "--cache-dir", str(tmp_path / "cache"),
                   "--work-dir", str(tmp_path / "work")])

    [(cmd, env)] = launches
    assert cmd[cmd.index("--nproc-per-node") + 1] == "4"
    assert cmd[cmd.index("--stage") + 1] == "generate"
    assert (env["DIFFLET_BACKEND"], env["DIFFLET_EXEC_MODE"]) == ("neuron", "eager")
    assert env["DIFFLET_COMPILE_CACHE"] == str(tmp_path / "cache")


@pytest.mark.parametrize("flag", [["--teacache-cadence", "2"],
                                  ["--shapes", "1024x1024,768x1360"]])
def test_main_generate_backend_neuron_rejects_before_any_launch(monkeypatch, tmp_path, capsys,
                                                                flag):
    cli_main = importlib.import_module("difflet.cli.main")  # the package exports main()

    _unset(monkeypatch, *_CACHE_ENV, "NEURON_RT_NUM_CORES")
    _backend(monkeypatch, None)
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["generate", "--model-id", _QWEN_ID, "--backend", "neuron",
                       "--total-cores", "4", "--prompt", "a fox", "--output",
                       str(tmp_path / "o.png"), "--work-dir", str(tmp_path / "work"), *flag])
    # main's own checks let these through (Qwen-Image supports sp and TeaCache on Trainium);
    # the orchestrator's driver hook refuses them.
    assert exc.value.code == 2
    assert capsys.readouterr().err.startswith("Error: the neuron backend does not support --")


# ----------------------------------------------------------------- device text encoder (opt-in)


def _header_only_snapshot(root: Path) -> Path:
    """The real Qwen-Image sizes as safetensors headers only: the DiT, and the text encoder's
    language model, vision tower and lm_head."""
    import json
    import struct

    def write(component, tensors):
        header = json.dumps({name: {"dtype": "BF16", "shape": [n], "data_offsets": [0, 0]}
                             for name, n in tensors.items()})
        (root / component).mkdir(parents=True)
        with open(root / component / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(header)) + header.encode())

    write("transformer", {"w": _REAL_DIT_NUMEL})
    write("text_encoder", {"model.layers.0.w": 7_070_619_136, "visual.w": 676_550_144,
                           "lm_head.weight": 544_997_376})
    return root


_TE_NOTE = "[qwen_image] Qwen2.5-VL on the device needs at least"


@pytest.fixture
def qwen_snapshot(monkeypatch, tmp_path):
    _backend(monkeypatch, "neuron")
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    stages, decoded = _record_stages(monkeypatch)
    root = _header_only_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda orch, stage, **kw: stages.append((stage, kw["cli_args"])))
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(root))
    monkeypatch.setattr(QwenImageOrchestrator, "download", lambda self: None)
    return stages


@pytest.mark.parametrize("overrides", [
    dict(tp_degree=2, cp_degree=2),  # --cp-degree 2 on four cores
    dict(tp_degree=2),               # each replica of --dp 2 --tp-degree 2
], ids=["tp2-cp2", "tp2"])
@pytest.mark.parametrize("mode", ["compile", "generate", "run"])
def test_a_narrow_tp_keeps_qwen_vl_on_the_host_before_any_launch(qwen_snapshot, tmp_path,
                                                                 capsys, mode, overrides):
    args = _qwen_args(device_text_encoder=True, work_dir=str(tmp_path / "w"), **overrides)
    getattr(QwenImageOrchestrator(args), mode)()
    out, err = capsys.readouterr()
    [note] = [line for line in out.splitlines() if line.startswith(_TE_NOTE)]  # once, for run
    assert "25.6 GiB per core beside the DiT at tp=2, over the 22.8 GiB budget" in note
    assert note.endswith("encoding on rank 0's host instead (without --device-text-encoder)")
    assert err == "" and args.device_text_encoder is False
    assert qwen_snapshot and all("--device-text-encoder" not in cli for _, cli in qwen_snapshot)


def test_qwen_vl_goes_to_the_device_where_it_fits(qwen_snapshot, neuron_stage, tmp_path, capsys):
    # tp4: 3.3 GiB of encoder beside 9.5 GiB of DiT per core.
    from difflet.cli import stage as stage_mod

    args = _qwen_args(device_text_encoder=True, work_dir=str(tmp_path / "w"))
    QwenImageOrchestrator(args).generate()
    assert _TE_NOTE not in capsys.readouterr().out and args.device_text_encoder is True
    [(stage, cli)] = qwen_snapshot
    stage_args, _ = stage_mod._build_stage_parser().parse_known_args(
        ["--orchestrator", _QWEN_ID, "--stage", stage, *cli])
    assert stage_args.device_text_encoder is True
    # The ranks build the application with the device encoder.
    stage_args.stage_mode = "compile"
    stage_args.cache_dir = str(tmp_path / "cache")
    QwenImageOrchestrator(stage_args)._stage_generate(stage_args)
    assert neuron_stage.CALLS[0]["kwargs"]["application_kwargs"] == {
        **_NEURON_APP_KWARGS, "text_encoder_device": "neuron"}


def test_the_host_encoder_stays_the_default(qwen_snapshot, neuron_stage, tmp_path):
    args = _qwen_args(work_dir=str(tmp_path / "w"))
    QwenImageOrchestrator(args).generate()
    assert all("--device-text-encoder" not in cli for _, cli in qwen_snapshot)
    args = _qwen_args(stage_mode="compile", cache_dir=str(tmp_path / "cache"))
    QwenImageOrchestrator(args)._stage_generate(args)
    assert "text_encoder_device" not in neuron_stage.CALLS[0]["kwargs"]["application_kwargs"]


def test_weights_not_local_leave_qwen_vl_to_the_application(monkeypatch, tmp_path, capsys):
    _backend(monkeypatch, "neuron")
    _record_stages(monkeypatch)

    def not_local(*a, **k):
        raise FileNotFoundError("not in the cache")

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", not_local)
    args = _qwen_args(device_text_encoder=True, tp_degree=2)
    QwenImageOrchestrator(args).compile()
    assert args.device_text_encoder is True and _TE_NOTE not in capsys.readouterr().out


@pytest.mark.parametrize("backend", [None, "trainium", "tpu"])
def test_aot_backends_never_move_qwen_vl(monkeypatch, tmp_path, capsys, backend):
    _backend(monkeypatch, backend)
    _record_stages(monkeypatch)
    args = _qwen_args(device_text_encoder=True, tp_degree=2, work_dir=str(tmp_path / "w"))
    QwenImageOrchestrator(args).generate()
    assert args.device_text_encoder is True and _TE_NOTE not in capsys.readouterr().out


def test_device_text_encoder_is_parsed_and_dp_workers_inherit_the_fallback(qwen_snapshot,
                                                                          tmp_path):
    from difflet.cli.dp.router import worker_cli_args

    cli_main = importlib.import_module("difflet.cli.main")
    for command in ("compile", "generate", "run"):
        parsed = cli_main._build_parser().parse_args(
            [command, "--model-id", _QWEN_ID, "--device-text-encoder"])
        assert parsed.device_text_encoder is True
    parsed.dp = 2
    assert "--device-text-encoder" in worker_cli_args(parsed)
    # `run` with --dp compiles in the driver, then the router forwards the args: a tp2 replica
    # inherits the host encoder the compile fell back to.
    parsed.tp_degree, parsed.work_dir = 2, str(tmp_path / "w")
    QwenImageOrchestrator(parsed).compile()
    assert "--device-text-encoder" not in worker_cli_args(parsed)


# ----------------------------------------------------------------- true CFG and CFG parallelism


def test_cfg_parallel_doubles_the_cores_and_keys_its_own_artifact(monkeypatch, tmp_path):
    _backend(monkeypatch, "neuron")
    stages, _ = _record_stages(monkeypatch)
    args = _qwen_args(tp_degree=2, cfg_parallel=True, true_cfg_scale=4.0,
                      work_dir=str(tmp_path / "w"), cache_dir="/c")
    orch = QwenImageOrchestrator(args)
    orch.generate()
    assert stages == [("generate", 4)]  # each CFG branch on its own TP group of 2
    plain = _qwen_args(tp_degree=2, cache_dir="/c")
    assert orch._stage_cache_inputs("generate", args)["cfg_parallel"] is True
    assert "cfg_parallel" not in orch._stage_cache_inputs("generate", plain)  # additive
    assert orch._stage_compiled_dir("generate", args) != orch._stage_compiled_dir("generate",
                                                                                   plain)


def test_compile_with_cfg_parallel_needs_no_scale(monkeypatch, tmp_path):
    # compile only warms the batch-2 pair up; the scale is a generate-time setting.
    _backend(monkeypatch, "neuron")
    stages, _ = _record_stages(monkeypatch)
    QwenImageOrchestrator(_qwen_args(cfg_parallel=True)).compile()
    assert stages == [("generate", 8)]


@pytest.mark.parametrize("scale", [None, 1.0])
@pytest.mark.parametrize("mode", ["generate", "run"])
def test_cfg_parallel_without_true_cfg_is_refused_before_any_launch(monkeypatch, tmp_path,
                                                                    capsys, mode, scale):
    _backend(monkeypatch, "neuron")
    monkeypatch.setattr(qwen_mod.runner, "run_stage",
                        lambda *a, **k: pytest.fail("no stage may be launched"))
    monkeypatch.setattr(QwenImageOrchestrator, "download",
                        lambda self: pytest.fail("rejected before the download"))
    args = _qwen_args(cfg_parallel=True, true_cfg_scale=scale, work_dir=str(tmp_path / "w"))
    with pytest.raises(SystemExit) as exc:
        getattr(QwenImageOrchestrator(args), mode)()
    assert exc.value.code == 2
    assert capsys.readouterr().err.strip() == (
        "Error: --cfg-parallel splits the two branches of true classifier-free guidance; "
        "Qwen-Image runs them only with --true-cfg-scale > 1; nothing was launched.")


def test_cfg_parallel_with_cp_is_refused(monkeypatch, tmp_path, capsys):
    _backend(monkeypatch, "neuron")
    monkeypatch.setattr(qwen_mod.runner, "run_stage", lambda *a, **k: pytest.fail("launched"))
    args = _qwen_args(cfg_parallel=True, true_cfg_scale=4.0, tp_degree=1, cp_degree=2)
    with pytest.raises(SystemExit):
        QwenImageOrchestrator(args).generate()
    assert "--cfg-parallel and --cp-degree > 1 are mutually exclusive" in capsys.readouterr().err


@pytest.mark.parametrize("backend", [None, "trainium", "tpu"])
@pytest.mark.parametrize("overrides", [dict(true_cfg_scale=4.0), dict(negative_prompt="x")])
def test_true_cfg_flags_are_refused_off_neuron(monkeypatch, tmp_path, capsys, backend,
                                               overrides):
    _backend(monkeypatch, backend)
    monkeypatch.setattr(qwen_mod.runner, "run_stage", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(SystemExit) as exc:
        QwenImageOrchestrator(_qwen_args(work_dir=str(tmp_path / "w"), **overrides)).generate()
    assert exc.value.code == 2
    assert "run on the neuron backend only" in capsys.readouterr().err


def test_true_cfg_reaches_the_ranks_only_when_it_is_on(neuron_stage, tmp_path):
    from difflet.cli import stage as stage_mod

    def calls(**overrides):
        neuron_stage.CALLS.clear()
        args = _qwen_args(work_dir=str(tmp_path), cache_dir=str(tmp_path / "c"), **overrides)
        shared = QwenImageOrchestrator(args)._shared_cli_args("generate", str(tmp_path))
        stage_args, _ = stage_mod._build_stage_parser().parse_known_args(
            ["--orchestrator", _QWEN_ID, "--stage", "generate", *shared])
        QwenImageOrchestrator(stage_args)._stage_generate(stage_args)
        return neuron_stage.CALLS[0]["pipeline"].app.calls

    [plain] = calls()
    assert "true_cfg_scale" not in plain and "negative_prompt" not in plain  # unchanged call
    assert "true_cfg_scale" not in calls(true_cfg_scale=1.0)[0]  # not above 1: off
    [default_negative] = calls(true_cfg_scale=4.0)
    assert (default_negative["true_cfg_scale"], default_negative["negative_prompt"]) == (4.0, " ")
    [given] = calls(true_cfg_scale=4.0, negative_prompt="blurry", cfg_parallel=True)
    assert (given["true_cfg_scale"], given["negative_prompt"]) == (4.0, "blurry")
    assert neuron_stage.CALLS[0]["kwargs"]["parallel"].cfg_parallel_enabled is True


def test_a_request_line_negative_prompt_wins(neuron_stage, tmp_path):
    from difflet.cli.dp.requests_io import RequestSpec, write_manifest

    requests = tmp_path / "requests"
    write_manifest([RequestSpec(index=0, prompt="a", output=str(tmp_path / "o0.png"),
                                negative_prompt="ugly", assigned_worker=0),
                    RequestSpec(index=1, prompt="b", output=str(tmp_path / "o1.png"),
                                assigned_worker=0)], requests)
    args = _qwen_args(work_dir=str(tmp_path), requests_dir=str(requests), worker_index=0,
                      dp_schedule="round_robin", true_cfg_scale=3.0, negative_prompt="blurry")
    QwenImageOrchestrator(args)._stage_generate(args)
    app = neuron_stage.CALLS[0]["pipeline"].app
    assert [c["negative_prompt"] for c in app.calls] == ["ugly", "blurry"]


def test_main_cfg_parallel_needs_true_cfg_on_neuron(monkeypatch, tmp_path, capsys):
    cli_main = importlib.import_module("difflet.cli.main")

    _unset(monkeypatch, *_CACHE_ENV, "NEURON_RT_NUM_CORES")
    _backend(monkeypatch, None)
    launches = _record_launches(monkeypatch)
    base = ["generate", "--model-id", _QWEN_ID, "--backend", "neuron", "--cfg-parallel",
            "--tp-degree", "2", "--total-cores", "4", "--prompt", "a fox", "--output",
            str(tmp_path / "o.png"), "--work-dir", str(tmp_path / "w")]
    with pytest.raises(SystemExit) as exc:
        cli_main.main(base)
    assert exc.value.code == 1 and launches == []
    assert "runs them only with --true-cfg-scale > 1" in capsys.readouterr().err
    cli_main.main([*base, "--true-cfg-scale", "4", "--negative-prompt", "blurry"])
    [(cmd, _)] = launches
    assert cmd[cmd.index("--nproc-per-node") + 1] == "4"
    assert cmd[cmd.index("--true-cfg-scale") + 1] == "4.0" and "--cfg-parallel" in cmd
    assert cmd[cmd.index("--negative-prompt") + 1] == "blurry"


def test_main_cfg_parallel_stays_refused_on_trainium(monkeypatch, tmp_path, capsys):
    cli_main = importlib.import_module("difflet.cli.main")

    _backend(monkeypatch, None)
    monkeypatch.setattr(cli_main, "_get_orchestrator", lambda args: pytest.fail("dispatched"))
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["generate", "--model-id", _QWEN_ID, "--backend", "trainium",
                       "--cfg-parallel", "--true-cfg-scale", "4", "--prompt", "a",
                       "--output", str(tmp_path / "o.png")])
    assert exc.value.code == 1
    assert "is guidance-distilled" in capsys.readouterr().err


def test_dp_router_forwards_true_cfg_and_keeps_the_negative_prompt(monkeypatch, tmp_path):
    from difflet.cli.dp import router

    cli_main = importlib.import_module("difflet.cli.main")
    _unset(monkeypatch, "NEURON_RT_NUM_CORES")
    _backend(monkeypatch, None)
    routed = {}

    def fake_router(args, requests, *, replica_cores):
        routed.update(args=args, requests=requests, replica_cores=replica_cores)
        return 0

    monkeypatch.setattr(router, "run_router", fake_router)
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["generate", "--model-id", _QWEN_ID, "--backend", "neuron", "--dp", "2",
                       "--tp-degree", "1", "--cfg-parallel", "--true-cfg-scale", "4",
                       "--negative-prompt", "blurry", "--total-cores", "4", "--prompt", "a fox",
                       "--output", str(tmp_path / "o.png")])
    assert exc.value.code == 0
    [request] = routed["requests"]
    assert request.negative_prompt == "blurry" and routed["replica_cores"] == 2
    forwarded = router.worker_cli_args(routed["args"])
    assert forwarded[forwarded.index("--true-cfg-scale") + 1] == "4.0"
    assert forwarded[forwarded.index("--negative-prompt") + 1] == "blurry"
    assert "--cfg-parallel" in forwarded
