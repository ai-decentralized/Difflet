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


def test_only_rank_zero_writes(monkeypatch, tmp_path):
    out_dir = tmp_path / "dump"
    monkeypatch.setenv("DIFFLET_DUMP_LATENTS_DIR", str(out_dir))
    monkeypatch.setenv("RANK", "1")
    dump_latents("final", torch.ones(2))
    assert not out_dir.exists()


def test_no_temporary_file_is_left_behind(monkeypatch, tmp_path):
    monkeypatch.setenv("DIFFLET_DUMP_LATENTS_DIR", str(tmp_path))
    monkeypatch.delenv("RANK", raising=False)
    dump_latents("final", torch.ones(2))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final.pt"]


def test_frames_uint8_rules():
    import numpy as np

    from difflet.utils.latent_dump import frames_uint8

    x = np.array([0.0, 0.5, 0.999, 1.2, -0.1], dtype=np.float32)
    assert frames_uint8(x, "truncate").tolist() == [0, 127, 254, 255, 0]
    assert frames_uint8(x, "round").tolist() == [0, 128, 255, 255, 0]  # 127.5 rounds to even 128
    assert frames_uint8(x, "round_half_up").tolist() == [0, 128, 255, 255, 0]


def test_dump_frames_writes_fhwc_uint8_and_its_hash(monkeypatch, tmp_path):
    import hashlib
    import json

    import numpy as np

    from difflet.utils.latent_dump import dump_frames

    monkeypatch.setenv("DIFFLET_DUMP_LATENTS_DIR", str(tmp_path))
    monkeypatch.delenv("RANK", raising=False)
    video = torch.rand(3, 2, 4, 5)  # C, F, H, W in [0, 1]
    dump_frames(video, rule="truncate", layout="CFHW")
    frames = np.load(tmp_path / "frames.npy")
    meta = json.loads((tmp_path / "frames.json").read_text())
    assert frames.shape == (2, 4, 5, 3) and frames.dtype == np.uint8
    assert meta["rule"] == "truncate" and meta["shape"] == [2, 4, 5, 3]
    assert meta["sha256"] == hashlib.sha256(frames.tobytes()).hexdigest()
    expect = (video.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
    assert np.array_equal(frames, expect)


def test_dump_frames_keeps_uint8_input_as_is(monkeypatch, tmp_path):
    import numpy as np

    from difflet.utils.latent_dump import dump_frames

    monkeypatch.setenv("DIFFLET_DUMP_LATENTS_DIR", str(tmp_path))
    monkeypatch.delenv("RANK", raising=False)
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    dump_frames(image, rule="round", layout="HWC")
    assert np.array_equal(np.load(tmp_path / "frames.npy")[0], image)
