from __future__ import annotations

import torch

from difflet.utils.latent_dump import dump_latents


def test_noop_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("DIFFLET_DUMP_LATENTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    dump_latents("final", torch.ones(2, 3))
    assert list(tmp_path.iterdir()) == []


def test_writes_cpu_tensor(monkeypatch, tmp_path):
    out_dir = tmp_path / "dump"
    monkeypatch.setenv("DIFFLET_DUMP_LATENTS_DIR", str(out_dir))
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    dump_latents("initial_video", tensor)
    loaded = torch.load(out_dir / "initial_video.pt")
    assert loaded.device.type == "cpu"
    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, tensor)
