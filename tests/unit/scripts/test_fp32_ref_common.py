"""Phase G's CPU fp32 reference helper (``scripts/fp32_ref/_common.py``) and scorer
(``scripts/fp32_ref/score.py``): the device-noise law and its md5 against the values the phase G
research recorded, the device-layout saves (an image and a 4-frame clip, no model), the fps the
device's video orchestrators encode at, and the scorer's device-file table against this host's
bench directory."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import torch
from PIL import Image

_DIR = Path(__file__).resolve().parents[3] / "scripts" / "fp32_ref"
_MODELS = {"flux", "qwen_image", "ltx_2", "wan21", "wan22", "hunyuan_video", "hunyuan_video_15"}
#: The device reruns writing these now (Wan 2.1's MXFP8, both Wan 2.2 arms): a missing file
#: under them is pending, not wrong; every other device file must be on the host.
_PENDING = ("wan_h3/final/wan21_mx/", "wan_h3/final/wan22d_bf16/", "wan_h3/final/wan22d_mx/")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"fp32_ref_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load("_common")
score = _load("score")


def _gen(seed: int = 42) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_device_noise_bf16_draw_is_the_bf16_randn_upcast():
    shape = (1, 16, 3, 8, 8)
    noise = common.device_noise(shape, 42, bf16_draw=True)
    assert noise.dtype == torch.float32
    assert torch.equal(noise, torch.randn(shape, generator=_gen(), dtype=torch.bfloat16).float())
    assert not torch.equal(noise, torch.randn(shape, generator=_gen(), dtype=torch.float32))


def test_device_noise_fp32_draw_is_the_fp32_randn():
    shape = (1, 16, 3, 8, 8)
    noise = common.device_noise(shape, 7, bf16_draw=False)
    assert torch.equal(noise, torch.randn(shape, generator=_gen(7), dtype=torch.float32))


def test_noise_md5s_match_the_research():
    """The md5s the research recorded for HunyuanVideo's fp32 draw and FLUX's packed bf16 draw,
    so the per-model scripts can assert theirs with ``tensor_md5``."""
    hv1 = common.device_noise((1, 16, 16, 40, 64), 42, bf16_draw=False)
    assert common.tensor_md5(hv1) == "fa10d6052fb81005792bbdab0b95d5d3"
    flux = common.device_noise((1, 16, 128, 128), 42, bf16_draw=True).to(torch.bfloat16)
    packed = flux.view(1, 16, 64, 2, 64, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, 4096, 64)
    assert common.tensor_md5(packed) == "be1ad0a63781cbc9b93ec9007d44b4ad"


def test_tensor_md5_is_stable_and_dtype_sensitive():
    t = torch.arange(8, dtype=torch.float32).view(2, 4)
    assert common.tensor_md5(t) == common.tensor_md5(t.clone())
    assert common.tensor_md5(t.t()) == common.tensor_md5(t.t().contiguous())
    assert common.tensor_md5(t) != common.tensor_md5(t.double())
    assert common.tensor_md5(t) != common.tensor_md5(t.to(torch.bfloat16))


def test_set_threads():
    before = torch.get_num_threads()
    try:
        common.set_threads(1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)
    with pytest.raises(ValueError, match="threads"):
        common.set_threads(0)


def test_device_fps_is_the_orchestrators():
    from difflet.cli.orchestrators import hunyuan_video, hunyuan_video_15, ltx_2, wan

    wan_fps = inspect.signature(wan._save_video).parameters["fps"].default
    assert common.DEVICE_FPS == {"ltx_2": ltx_2._FPS, "wan21": wan_fps, "wan22": wan_fps,
                                 "hunyuan_video": hunyuan_video._FPS,
                                 "hunyuan_video_15": hunyuan_video_15._FPS}


def _record(out_dir: Path) -> dict:
    return json.loads((out_dir / "run.json").read_text())


def test_save_run_image(tmp_path):
    latents = torch.randn(1, 4096, 64)
    image = Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8))
    common.save_run(tmp_path / "ref", latents=latents, output=image, settings={"steps": 50},
                    noise_md5="abc", timings={"steps_s": 1.5})
    ref = tmp_path / "ref"
    assert sorted(p.name for p in ref.iterdir()) == ["latents.pt", "out.png", "run.json"]
    assert torch.equal(torch.load(ref / "latents.pt", weights_only=True), latents)
    assert np.asarray(Image.open(ref / "out.png")).tolist() == np.asarray(image).tolist()
    record = _record(ref)
    assert record["settings"] == {"steps": 50} and record["noise_md5"] == "abc"
    assert record["timings"] == {"steps_s": 1.5} and record["output"] == "out.png"


def test_save_run_video(tmp_path):
    latents = {"video": torch.randn(1, 6, 8), "audio": torch.randn(1, 3, 8)}
    frames = np.linspace(0.0, 1.0, 4 * 32 * 32 * 3, dtype=np.float32).reshape(4, 32, 32, 3)
    common.save_run(tmp_path, latents=latents, output=frames, settings={}, noise_md5="abc",
                    timings={}, fps=16)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["latents.pt", "out.mp4", "run.json"]
    saved = torch.load(tmp_path / "latents.pt", weights_only=True)
    assert saved.keys() == latents.keys() and torch.equal(saved["audio"], latents["audio"])
    assert len(list(iio.imiter(tmp_path / "out.mp4"))) == 4
    assert iio.immeta(tmp_path / "out.mp4")["fps"] == 16
    assert _record(tmp_path)["fps"] == 16 and _record(tmp_path)["output"] == "out.mp4"


@pytest.mark.parametrize("frames, fps, match", [
    (np.zeros((4, 32, 32, 3), dtype=np.float32), None, "fps"),
    (np.zeros((4, 32, 32, 3), dtype=np.uint8), 16, "float"),
    (np.zeros((1, 3, 4, 32, 32), dtype=np.float32), 16, r"\(F, H, W, 3\)"),
    (np.full((4, 32, 32, 3), -1.0, dtype=np.float32), 16, r"\[0, 1\]"),
])
def test_save_run_refuses_frames_the_device_writer_would_encode_wrong(tmp_path, frames, fps,
                                                                       match):
    with pytest.raises(ValueError, match=match):
        common.save_run(tmp_path, latents=torch.zeros(1), output=frames, settings={},
                        noise_md5="", timings={}, fps=fps)


def test_device_outputs_name_every_model():
    assert set(score.DEVICE_OUTPUTS) == _MODELS
    for arms in score.DEVICE_OUTPUTS.values():
        assert set(arms) == {"bf16", "mx"}
        assert all(set(arm) == {"output", "latents"} for arm in arms.values())
    assert score.DEVICE_OUTPUTS["flux"]["bf16"]["latents"] is None  # tp4 decodes in process


_PATHS = sorted({p for arms in score.DEVICE_OUTPUTS.values() for arm in arms.values()
                 for p in arm.values() if p is not None})


@pytest.mark.parametrize("rel", _PATHS)
def test_device_output_exists_on_this_host(rel):
    if not score.BENCH.is_dir():
        pytest.skip(f"no bench directory {score.BENCH} on this host")
    if not (score.BENCH / rel).exists() and rel.startswith(_PENDING):
        pytest.skip(f"{rel}: its device rerun has not written it yet")
    assert (score.BENCH / rel).exists()


def test_score_writes_one_row_per_model(tmp_path, monkeypatch, capsys):
    """A tiny image model end to end: the reference saved by save_run, two device arms, the
    score.json and the printed row with the mx - bf16 PSNR difference."""
    monkeypatch.setattr(score._cdr(), "_lpips_net", lambda: None)
    bench, ref = tmp_path / "bench", tmp_path / "ref"
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    latents = torch.randn(1, 16, 64)
    common.save_run(ref, latents=latents, output=Image.fromarray(base), settings={},
                    noise_md5="", timings={})
    for arm, noise in (("bf16", 2), ("mx", 8)):
        (bench / arm).mkdir(parents=True)
        Image.fromarray(np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)).save(
            bench / arm / "gen.png")
        torch.save(latents + noise * 1e-3, bench / arm / "latents.pt")
    monkeypatch.setattr(score, "BENCH", bench)
    monkeypatch.setitem(score.DEVICE_OUTPUTS, "flux", {
        arm: {"output": f"{arm}/gen.png", "latents": f"{arm}/latents.pt"}
        for arm in ("bf16", "mx")})
    assert score.main(["--model", "flux", "--ref-dir", str(ref)]) == 0
    result = json.loads((ref / "score.json").read_text())
    bf16, mx = result["bf16"]["image"]["psnr"], result["mx"]["image"]["psnr"]
    assert bf16 > mx and result["mx_minus_bf16_db"] == pytest.approx(mx - bf16)
    assert result["mx"]["latents"]["cosine"] < 1.0
    row = capsys.readouterr().out.strip().splitlines()[-1]
    assert row.startswith("flux | bf16 vs fp32 ") and f"| mx - bf16 {mx - bf16:+.2f} dB" in row
