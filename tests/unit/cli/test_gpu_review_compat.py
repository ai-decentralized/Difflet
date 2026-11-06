"""Adding a model preserves the established GPU CLI defaults and video writer."""

from types import SimpleNamespace

import pytest
import torch

from difflet.models.spec import gpu_model_specs
from tests.unit.cli.test_gpu_cli import _orch


@pytest.mark.parametrize("spec", gpu_model_specs(), ids=lambda s: s.model_type)
def test_zero_request_steps_keep_existing_default(spec):
    kwargs = _orch(spec, steps=7).generate_kwargs(
        SimpleNamespace(steps=0, seed=0, guidance_scale=None, negative_prompt=None)
    )
    assert kwargs["num_inference_steps"] == spec.default_steps


@pytest.mark.parametrize(
    "spec", [s for s in gpu_model_specs() if s.output == "video"], ids=lambda s: s.model_type
)
def test_video_writer_receives_model_fps(spec, monkeypatch, tmp_path):
    from difflet.cli.orchestrators import wan

    calls = []
    monkeypatch.setattr(
        wan, "_save_video", lambda result, path, *, fps: calls.append((path, fps)) or True
    )
    output = tmp_path / "clip.mp4"
    _orch(spec).save_result(torch.zeros(1, 3, 9, 16, 16), output)
    assert calls == [(str(output), spec.default_fps)]
