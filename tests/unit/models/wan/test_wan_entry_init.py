"""Coverage for difflet.models.wan.entry and the package __init__ lazy loader."""

from __future__ import annotations

import pytest
import torch

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.models.wan import entry
from difflet.registry import resolve_model


def _create_on_cuda(tmp_path, parallel, shape=None):
    """The registry's route for a GPU backend: the generic GPU flow, not ``entry.py``."""
    return resolve_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers").create_application(
        model_path=str(tmp_path),
        parallel=parallel,
        dtype="bf16",
        shape=shape if shape is not None else {},
        backend="cuda",
    )


def test_create_wan_application_rejects_unsupported_backend(tmp_path):
    with pytest.raises(NotImplementedError, match="trainium, tpu and neuron backends, got 'rocm'"):
        entry.create_wan_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(),
            dtype="bf16",
            shape={},
            backend="rocm",
        )


def test_create_wan_application_builds_neuron_app(tmp_path):
    """The neuron branch builds TorchNeuronWanApplication with the keyword arguments the CLI
    stage passes through the registry (difflet/cli/orchestrators/wan.py), never the Trainium
    application. ``device="cpu"``: nothing touches a NeuronCore; nothing loads either."""
    import torch

    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.registry import resolve_model
    from tests.unit.backends._neuron_wan_toy import TINY_SHAPE, write_tiny_wan_model

    pytest.importorskip("safetensors.torch")
    write_tiny_wan_model(tmp_path)
    cli_kwargs = {
        "text_seq_len": 8, "batch_size": 1, "enable_text_encoder": True,
        "enable_transformer": True, "enable_transformer_2": False,
        "enable_vae_decoder": False, "exec_mode": None, "encoder_threads": 12,
    }
    direct = entry.create_wan_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype=torch.bfloat16,
        shape=dict(TINY_SHAPE),
        backend="neuron",
        device="cpu",
        **cli_kwargs,
    )
    via_registry = resolve_model("Wan-AI/Wan2.1-T2V-14B-Diffusers").create_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype=torch.bfloat16,
        shape=dict(TINY_SHAPE),
        backend="neuron",
        application_kwargs={**cli_kwargs, "device": "cpu"},
    )
    for app in (direct, via_registry):
        assert isinstance(app, TorchNeuronWanApplication)
        assert app.device == torch.device("cpu") and app.dtype is torch.bfloat16
        assert app.shape == TINY_SHAPE and app.text_seq_len == 8
        assert app.transformer is not None and app.text_encoder is not None
        assert app.text_encoder.threads == 12
        assert app.pipeline is None  # built, not loaded
    with_vae = entry.create_wan_application(  # the device VAE stage (bf16 up to 9 frames)
        model_path=str(tmp_path), parallel=DiffletParallelConfig(tp_degree=1),
        dtype=torch.bfloat16, shape=dict(TINY_SHAPE), backend="neuron", device="cpu",
        enable_vae_decoder=True,
    )
    assert type(with_vae.vae_decoder).__name__ == "_DeviceVaeDecoder"
def test_registry_builds_the_cuda_wan_application(tmp_path, monkeypatch):
    """The cuda backend routes to the one-process GPU application.

    No checkpoint under ``tmp_path``, so no experts — enough to prove the dispatch.
    """
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    app = _create_on_cuda(
        tmp_path,
        DiffletParallelConfig(tp_degree=1),
        shape={"height": None, "width": None, "num_frames": None},
    )
    assert type(app).__name__ == "CudaWanApplication"
    assert app.transformer is None and app.transformer_2 is None
    assert app.shape == {"height": 480, "width": 832, "num_frames": 9}
    assert app.dtype is torch.bfloat16
    assert app.device == torch.device("cuda")
    # Both experts fit a 96 GB card: resident by default on the GPU flow only.
    assert app.enable_transformer_2 is True


@pytest.mark.parametrize(
    "parallel,expected",
    [
        (DiffletParallelConfig(tp_degree=4), "not supported"),
        (DiffletParallelConfig(tp_degree=1, cp_degree=2), "not supported"),
        (DiffletParallelConfig(tp_degree=1, sp_enabled=True), "not supported"),
    ],
)
def test_cuda_refuses_multi_rank_and_sp_up_front(tmp_path, parallel, expected):
    # Refused before either 14B expert is touched, not at the first collective.
    with pytest.raises(NotImplementedError, match=expected):
        _create_on_cuda(tmp_path, parallel)


def test_cuda_accepts_cfg_parallel(tmp_path, monkeypatch):
    """CFG-parallel runs the cond and uncond branches on two GPUs under torchrun."""
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")
    app = _create_on_cuda(tmp_path, DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True))
    assert type(app).__name__ == "CudaWanApplication"
    assert app.parallel.cfg_parallel_enabled is True


def test_create_wan_application_builds_tpu_app(tmp_path):
    """The tpu branch must not fall through to the Trainium application.

    ``tmp_path`` has no ``transformer/config.json``, so the app is built
    without a DiT — enough to prove the dispatch without a device.
    """
    from difflet.models.wan.tpu_application import TpuWanApplication

    app = entry.create_wan_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": None, "width": None, "num_frames": None},
        backend="tpu",
    )
    assert isinstance(app, TpuWanApplication)
    assert app.transformer is None
    # Two resident experts do not fit a v5e chip at tp=4; opting in is the
    # caller's decision, never the default.
    assert app.enable_transformer_2 is False


def test_create_wan_application_builds_skeleton_app(tmp_path):
    from difflet.models.wan.application import NeuronWanApplication

    app = entry.create_wan_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": None, "width": None, "num_frames": None},
    )
    assert isinstance(app, NeuronWanApplication)
    # No config.json in tmp_path → no active components.
    assert app.components() == []


def test_package_lazy_getattr_exports():
    import difflet.models.wan as wan_pkg

    assert wan_pkg.NeuronWanApplication.__name__ == "NeuronWanApplication"
    assert wan_pkg.WanOrchestrator.__name__ == "WanOrchestrator"
    assert wan_pkg.WanPipelineOutput.__name__ == "WanPipelineOutput"


def test_package_lazy_getattr_unknown_attribute_raises():
    import difflet.models.wan as wan_pkg

    with pytest.raises(AttributeError, match="has no attribute"):
        _ = wan_pkg.NotARealThing
