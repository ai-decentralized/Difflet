"""Hunyuan CUDA registration and lightweight spec imports."""

import subprocess
import sys

from difflet.models.hunyuan_video.spec import SPEC


def test_original_hunyuan_is_a_gpu_spec_but_15_remains_separate():
    from difflet.models.spec import gpu_model_spec, gpu_spec_for_cli_model_id
    from difflet.registry import resolve_model

    assert gpu_model_spec("hunyuan_video") is SPEC
    assert gpu_spec_for_cli_model_id(SPEC.cli_model_ids[0]) is SPEC
    assert "cuda" in resolve_model(SPEC.cli_model_ids[0]).backends
    assert gpu_model_spec("hunyuan_video_15") is None
    assert (
        "cuda"
        not in resolve_model("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v").backends
    )


def test_hunyuan_spec_does_not_load_torch_or_trainium():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import difflet.models.hunyuan_video.spec; "
            "assert 'torch' not in sys.modules; "
            "assert 'difflet.models.hunyuan_video.application' not in sys.modules",
        ],
        check=True,
    )
