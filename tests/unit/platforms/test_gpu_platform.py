"""The GPU platform family reads its device type and device hooks from the platform: a
platform with another device type overrides the hooks and the generic flow follows it without
naming ``cuda`` anywhere."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from difflet.platforms.gpu import GpuPlatform


class _FakeGpuPlatform(GpuPlatform):
    """A GPU-family platform on a made-up device type, recording every hook call."""

    name = "fake"
    display_name = "Fake"
    device_type = "fake"
    distributed_backend = "gloo"

    def __init__(self, available: bool = True):
        self.available = available
        self.calls: list[tuple] = []

    def device_available(self) -> bool:
        self.calls.append(("device_available",))
        return self.available

    def set_current_device(self, index: int) -> None:
        self.calls.append(("set_current_device", index))

    def device_count(self):
        return 3 if self.available else None

    def device_memory_bytes(self):
        return 7 if self.available else None

    def toolchain_versions(self):
        return {"fake-runtime": "1.0"}


def _no_cuda(monkeypatch):
    """Make any ``torch.cuda`` device call fail loudly: the fake platform must not reach it."""

    def boom(*args, **kwargs):
        raise AssertionError("torch.cuda was called for a non-cuda GPU platform")

    monkeypatch.setattr(torch.cuda, "set_device", boom)
    monkeypatch.setattr(torch.cuda, "device_count", boom)
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)


def test_prepare_runtime_uses_the_platform_hooks_not_torch_cuda(monkeypatch):
    _no_cuda(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "2")
    platform = _FakeGpuPlatform()

    platform.prepare_runtime(parallel=None)

    assert platform.calls == [("device_available",), ("set_current_device", 2)]
    assert platform.device_count() == 3 and platform.device_memory_bytes() == 7


def test_require_device_names_the_platform_device_type():
    platform = _FakeGpuPlatform(available=False)
    with pytest.raises(RuntimeError, match=r"torch\.fake\.is_available\(\) is False"):
        platform.prepare_runtime(parallel=None)


def test_cuda_defaults_are_the_torch_cuda_hooks(monkeypatch):
    from difflet.platforms.cuda import CudaPlatform

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.append(i))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    platform = CudaPlatform()

    assert platform.device_type == "cuda"
    assert platform.device_available() is True
    platform.set_current_device(1)
    assert calls == [1] and platform.device_count() == 4


def test_component_default_device_follows_the_platform(monkeypatch):
    from difflet.backends.gpu import application

    # ``cpu`` is a device type torch can construct; the point is that the component asks the
    # platform instead of assuming ``cuda``.
    class _CpuTyped(_FakeGpuPlatform):
        device_type = "cpu"

    monkeypatch.setattr(application, "gpu_platform", lambda: _CpuTyped())

    assert application.GpuComponent().device == torch.device("cpu")


def test_manifest_names_the_platform_and_old_manifests_stay_valid(monkeypatch, tmp_path):
    from difflet.backends.gpu import application

    monkeypatch.setattr(application, "gpu_platform", lambda: _FakeGpuPlatform())
    component = application.GpuComponent(device="cpu", dtype=torch.bfloat16)
    spec = SimpleNamespace(world_size=1, dp=1, cfg=1, cp=1, tp=1)

    body = component._manifest_body(spec)
    assert body["device_type"] == "fake" and body["toolchain"] == {"fake-runtime": "1.0"}
    assert "cuda" not in body

    # A manifest written before the platform split: ``cuda`` instead of ``device_type``.
    old = {
        "format": application.MANIFEST_FORMAT,
        "world_size": 1,
        "mesh": {"dp": 1, "cfg": 1, "cp": 1, "tp": 1},
        "dtype": str(torch.bfloat16),
        "torch": torch.__version__,
        "cuda": "12.8",
    }
    (tmp_path / application.MANIFEST_FILE_NAME).write_text(json.dumps(old))
    component._validate_manifest(tmp_path, spec)  # does not raise
