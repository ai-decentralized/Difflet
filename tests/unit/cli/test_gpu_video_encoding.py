"""CLI and resident serving must decode the same float video consistently."""

from pathlib import Path

import numpy as np
import pytest
import torch

from difflet.models.spec import gpu_model_specs
from tests.unit.cli.test_gpu_cli import _orch


@pytest.mark.parametrize(
    "spec", [s for s in gpu_model_specs() if s.output == "video"], ids=lambda s: s.model_type
)
@pytest.mark.parametrize(
    "destination",
    [None, "clip.mp4", "results/clip.mp4"],
    ids=["absolute", "relative", "nested-relative"],
)
def test_cli_and_serving_encode_identical_decoded_frames(spec, destination, tmp_path, monkeypatch):
    av = pytest.importorskip("av")
    pytest.importorskip("imageio_ffmpeg")
    from difflet.serving.video_media import encode_tensor_to_mp4

    generator = torch.Generator().manual_seed(42)
    frames = torch.rand(1, 3, 9, 16, 16, generator=generator) * 2 - 1
    monkeypatch.chdir(tmp_path)
    cli = Path(destination) if destination is not None else tmp_path / "cli.mp4"
    cli.parent.mkdir(parents=True, exist_ok=True)
    served = tmp_path / "served.mp4"
    _orch(spec).save_result(frames, cli)
    encode_tensor_to_mp4(
        frames, served, fps=spec.default_fps, layout="BCTHW", value_range="minus_one_to_one"
    )

    def decoded(path):
        with av.open(str(path)) as container:
            return np.stack(
                [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            )

    assert np.array_equal(decoded(cli), decoded(served))
