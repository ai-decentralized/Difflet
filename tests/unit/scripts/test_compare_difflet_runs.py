"""``scripts/compare_difflet_runs.py`` on decoded outputs: an image pair keeps its output; an
``.mp4`` pair, or a directory of frame PNGs, is compared frame by frame with the per-frame
metrics averaged, the frame count and the worst frame reported, and a count mismatch refused.
The clips are written with imageio, the library the script reads them with."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "compare_difflet_runs.py"
_FRAMES, _H, _W = 4, 32, 32  # 16-aligned: libx264 pads anything else


@pytest.fixture(scope="module")
def cdr():
    spec = importlib.util.spec_from_file_location("compare_difflet_runs", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_lpips(cdr, monkeypatch):
    """The AlexNet LPIPS net loads pretrained weights: the tests feed a fake or none at all."""
    monkeypatch.setattr(cdr, "_lpips_net", lambda: None)


def _clip(frames: int = _FRAMES) -> np.ndarray:
    """Smooth moving gradients (uint8 [F, H, W, 3]) that survive H.264 recognisably."""
    y, x = np.mgrid[0:_H, 0:_W]
    stack = []
    for f in range(frames):
        r = (x * 4 + f * 16) % 256
        g = (y * 4 + f * 8) % 256
        b = ((x + y) * 2 + f * 32) % 256
        stack.append(np.stack([r, g, b], axis=-1))
    return np.asarray(stack, dtype=np.uint8)


def _write_mp4(path: Path, frames: np.ndarray) -> Path:
    iio.imwrite(path, frames, fps=8)
    return path


def _write_frame_dir(path: Path, frames: np.ndarray) -> Path:
    path.mkdir()
    for i, frame in enumerate(frames):
        iio.imwrite(path / f"frame_{i:03d}.png", frame)
    return path


def test_identical_clip_reads_as_identical_frames(cdr, tmp_path):
    a = _write_mp4(tmp_path / "a.mp4", _clip())
    result = cdr.compare(a, a)
    assert math.isinf(result["psnr"])
    assert result["ssim"] == pytest.approx(1.0)
    assert result["lpips"] is None
    assert result["frames"] == _FRAMES
    assert result["min_psnr_frame"] == 0
    assert (result["ref"], result["cand"]) == (str(a), str(a))


def test_clip_pair_is_compared_frame_by_frame(cdr, tmp_path):
    clip = _clip()
    other = clip.copy()
    other[2] = 255 - other[2]  # one inverted frame: the worst by far, whatever libx264 does
    a = _write_mp4(tmp_path / "a.mp4", clip)
    b = _write_mp4(tmp_path / "b.mp4", other)
    result = cdr.compare(a, b)
    assert result["frames"] == _FRAMES
    assert 0.0 < result["psnr"] < math.inf
    assert result["ssim"] < 1.0
    assert result["min_psnr_frame"] == 2


def test_frame_count_mismatch_names_both_counts(cdr, tmp_path):
    a = _write_mp4(tmp_path / "a.mp4", _clip(4))
    b = _write_mp4(tmp_path / "b.mp4", _clip(3))
    with pytest.raises(ValueError, match=r"a\.mp4 has 4 frames.*b\.mp4 has 3 frames"):
        cdr.compare(a, b)


def test_frame_directory_averages_the_per_frame_psnr(cdr, tmp_path):
    ref = np.full((_FRAMES, _H, _W, 3), 100, dtype=np.uint8)
    cand = ref + 1  # 48.1308 dB
    cand[2] += 4  # 5 off: 10 log10(255^2 / 25)
    result = cdr.compare(_write_frame_dir(tmp_path / "ref", ref), _write_frame_dir(tmp_path / "cand", cand))
    per_frame = [10 * math.log10(255**2 / 1)] * 3 + [10 * math.log10(255**2 / 25)]
    assert result["psnr"] == pytest.approx(sum(per_frame) / 4, abs=1e-6)
    assert result["frames"] == _FRAMES
    assert result["min_psnr_frame"] == 2


def test_empty_frame_directory_is_refused(cdr, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="empty"):
        cdr.compare(tmp_path / "empty", tmp_path / "empty")


def test_lpips_is_averaged_over_the_frames(cdr, tmp_path, monkeypatch):
    seen = []

    def fake_net(x, y):
        seen.append(tuple(x.shape))
        return (x - y).abs().mean()

    monkeypatch.setattr(cdr, "_lpips_net", lambda: fake_net)
    ref = np.full((_FRAMES, _H, _W, 3), 100, dtype=np.uint8)
    cand = ref + 1
    cand[2] += 4
    result = cdr.compare(_write_frame_dir(tmp_path / "ref", ref), _write_frame_dir(tmp_path / "cand", cand))
    assert seen == [(1, 3, _H, _W)] * _FRAMES  # one frame pair per call, in the net's layout
    assert result["lpips"] == pytest.approx((1 + 1 + 5 + 1) / 4 / 127.5, abs=1e-6)


def test_image_pair_keeps_its_output(cdr, tmp_path):
    ref = np.full((_H, _W, 3), 100, dtype=np.uint8)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    iio.imwrite(a, ref)
    iio.imwrite(b, ref + 1)
    result = cdr.compare(a, b)
    assert set(result) == {"psnr", "ssim", "lpips", "ref", "cand"}
    assert result["psnr"] == pytest.approx(48.1308, abs=1e-3)


def test_compare_runs_nests_a_clip_under_video_and_an_image_under_image(cdr, tmp_path):
    a = _write_mp4(tmp_path / "a.mp4", _clip())
    png = tmp_path / "a.png"
    iio.imwrite(png, _clip()[0])
    latents = tmp_path / "latents.pt"
    torch.save(torch.ones(2, 3), latents)
    video = cdr.compare_runs(latents, latents, a, a)
    assert set(video) == {"ref_latents", "cand_latents", "latents", "video"}
    assert video["latents"]["bit_identical"] is True
    assert video["video"]["frames"] == _FRAMES
    image = cdr.compare_runs(None, None, png, png)
    assert set(image) == {"image"}
    assert "frames" not in image["image"]


def test_main_takes_a_clip_pair_as_the_outputs(cdr, tmp_path, monkeypatch, capsys):
    a = _write_mp4(tmp_path / "a.mp4", _clip())
    out = tmp_path / "metrics.json"
    monkeypatch.setattr("sys.argv", ["compare_difflet_runs.py", "--ref-image", str(a), "--cand-image", str(a),
                                     "--out", str(out)])
    assert cdr.main() == 0
    written = json.loads(out.read_text())
    assert written["video"]["frames"] == _FRAMES
    assert written["video"]["psnr"] == math.inf  # Infinity: the determinism signal, as for images
    assert json.loads(capsys.readouterr().out) == written
