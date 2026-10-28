"""Qwen-Image on the GPU CLI flow: what only Qwen-Image does (no guidance scale, the probe-free
TeaCache cadence). The flow itself is tested once per model in test_gpu_cli.py.

CPU-only: DiffletPipeline.from_pretrained is replaced by a recorder; no weights, no GPU.
"""

from __future__ import annotations

import pytest

from difflet.models.qwen_image.spec import SPEC
from tests.unit.cli import test_gpu_cli as gpu_cli_tests
from tests.unit.cli.test_gpu_cli import _batch, _manifest, _orch

recorder = gpu_cli_tests.recorder


@pytest.fixture
def spec():
    return SPEC


def test_guidance_scale_is_refused_not_dropped(recorder):
    # The application refuses guidance_scale (guidance_embeds=False → silent no-op); the CLI
    # must not hide that by never forwarding the flag.
    with pytest.raises(SystemExit) as exc:
        _orch(SPEC, guidance_scale=3.0).generate()
    assert "guidance" in str(exc.value) and recorder == []


def test_teacache_cadence_is_forwarded_as_an_application_kwarg(recorder, tmp_path):
    out = tmp_path / "qwen.png"
    _orch(SPEC, teacache_cadence=2, output=str(out)).generate()
    assert recorder[0][1]["application_kwargs"] == {"teacache_cadence": 2}


def test_a_request_guidance_scale_fails_that_request(recorder, tmp_path):
    # Qwen-Image takes no guidance scale: a request carrying one is failed, never dropped.
    from difflet.cli.dp.claims import summarize

    requests_dir = _manifest(
        SPEC, tmp_path, [dict(prompt="p0", guidance_scale=4.0), dict(prompt="p1")]
    )
    orch = _orch(SPEC, **_batch(requests_dir))
    orch.generate(argv=[])
    s = summarize(requests_dir)
    assert s.done == [1] and list(s.failed) == [0] and "guidance" in s.failed[0]
    assert [p for p, _ in orch._last_pipe.app.calls] == ["p1"]


def test_a_request_negative_prompt_fails_that_request(recorder, tmp_path):
    from difflet.cli.dp.claims import summarize

    requests_dir = _manifest(SPEC, tmp_path, [dict(prompt="p0", negative_prompt="blurry")])
    _orch(SPEC, **_batch(requests_dir)).generate(argv=[])
    s = summarize(requests_dir)
    assert list(s.failed) == [0] and "negative" in s.failed[0]
