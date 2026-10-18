import json
import struct

import pytest

from difflet.cli.dp.hbm_check import assert_replica_fits, component_weight_bytes


def _write_safetensors(path, tensors):
    """Minimal valid safetensors file: header only, zero-filled data."""
    header, offset = {}, 0
    for name, shape in tensors.items():
        n = 1
        for d in shape:
            n *= d
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [offset, offset + 2 * n]}
        offset += 2 * n
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


def test_component_weight_bytes(tmp_path):
    _write_safetensors(tmp_path / "transformer" / "model.safetensors",
                       {"w1": (1024, 1024), "w2": (512,)})
    _write_safetensors(tmp_path / "vae" / "diffusion_pytorch_model.safetensors",
                       {"conv": (16, 16, 3, 3)})
    sizes = component_weight_bytes(tmp_path, dtype_bytes=2)
    assert sizes["transformer"] == 2 * (1024 * 1024 + 512)
    assert sizes["vae"] == 2 * (16 * 16 * 3 * 3)


def test_assert_replica_fits_passes_under_limit(tmp_path):
    _write_safetensors(tmp_path / "transformer" / "m.safetensors", {"w": (10, 10)})
    assert_replica_fits(tmp_path, limit_bytes=1_000_000)  # no raise


def test_assert_replica_fits_raises_with_breakdown(tmp_path):
    _write_safetensors(tmp_path / "transformer" / "m.safetensors", {"w": (1000, 1000)})
    with pytest.raises(RuntimeError) as exc:
        assert_replica_fits(tmp_path, limit_bytes=1_000_000)
    assert "transformer" in str(exc.value)


def test_no_safetensors_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no safetensors"):
        component_weight_bytes(tmp_path)


# ------------------------------------------------- per-backend limit


class _Props:
    def __init__(self, total_memory):
        self.total_memory = total_memory


def _fake_cuda(monkeypatch, *, available, total_memory=0):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _Props(total_memory))


def test_default_limit_on_trainium_is_one_trn2_chip(tmp_path, monkeypatch):
    from difflet.cli.dp import hbm_check

    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
    assert hbm_check.replica_limit_bytes() == hbm_check.HBM_LIMIT_BYTES


def test_default_limit_on_cuda_is_the_device_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    _fake_cuda(monkeypatch, available=True, total_memory=1_000_000)
    _write_safetensors(tmp_path / "transformer" / "m.safetensors", {"w": (1000, 1000)})
    with pytest.raises(RuntimeError, match="exceed"):
        assert_replica_fits(tmp_path)  # 2 MB of weights > the 1 MB "GPU"
    _fake_cuda(monkeypatch, available=True, total_memory=10_000_000)
    assert_replica_fits(tmp_path)  # fits a 10 MB "GPU"


def test_cuda_without_a_device_skips_with_a_note(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    _fake_cuda(monkeypatch, available=False)
    _write_safetensors(tmp_path / "transformer" / "m.safetensors", {"w": (1000, 1000)})
    assert_replica_fits(tmp_path)  # no raise: the limit is unknown
    assert "skipping" in capsys.readouterr().out


def test_explicit_limit_wins_on_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    _fake_cuda(monkeypatch, available=True, total_memory=10**12)
    _write_safetensors(tmp_path / "transformer" / "m.safetensors", {"w": (1000, 1000)})
    with pytest.raises(RuntimeError):
        assert_replica_fits(tmp_path, limit_bytes=1_000_000)
