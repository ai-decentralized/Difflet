"""The CUDA serving adapters' shared half: the one-GPU resident plan and the refusals every
CUDA profile shares. CPU-only."""

from __future__ import annotations

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.serving.errors import DiffletServingError
from difflet.serving.orchestrators import cuda_resident as mod
from difflet.serving.types import ResolvedModelSource, ServingProfile

_SOURCE = ResolvedModelSource("hf_snapshot", "m/x", None, "/snap/x", "b" * 40)


def _profile(**over) -> ServingProfile:
    d = dict(
        model_id="m/x",
        model_type="wan",
        height=480,
        width=832,
        num_frames=9,
        parallel=DiffletParallelConfig(tp_degree=1),
        cache_dir="/c",
    )
    d.update(over)
    return ServingProfile(**d)


def test_resident_plan_claims_slot_zero_for_one_cuda_stage():
    plan = mod.resident_runtime_plan(_profile(), _SOURCE, allocation_id="wan-cuda")
    assert plan.mode == "resident" and plan.environment.available_core_ids == (0,)
    (allocation,) = plan.allocations
    assert allocation.allocation_id == "wan-cuda" and allocation.effective_num_cores == 1
    (stage,) = plan.stages
    assert stage.stage_id == "pipeline" and stage.placement == "cuda"
    assert stage.allocation_id == "wan-cuda" and stage.artifact_id is None


def test_identity_tracks_the_video_shape_and_leaves_image_identities_as_they_were():
    a = mod.resident_runtime_plan(_profile(), _SOURCE, allocation_id="a").profile_identity
    b = mod.resident_runtime_plan(
        _profile(num_frames=17), _SOURCE, allocation_id="a"
    ).profile_identity
    assert a != b
    # An image profile (no frames) hashes exactly the F3b inputs, so Qwen's identity is unchanged.
    from difflet.serving.types import CompileArtifactIdentity

    image = _profile(model_type="qwen_image", height=1024, width=1024, num_frames=None)
    expected = CompileArtifactIdentity.from_cache_inputs(
        {
            "backend": "cuda",
            "model_type": "qwen_image",
            "source_id": _SOURCE.resolved_source_id,
            "height": 1024,
            "width": 1024,
            "tp_degree": 1,
            "dtype": image.dtype,
        }
    ).digest
    assert mod.resident_runtime_plan(image, _SOURCE, allocation_id="q").profile_identity == (
        expected
    )


@pytest.mark.parametrize(
    "over,expected",
    [
        ({"shapes": ((480, 832, 9), (480, 832, 17))}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=2)}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, sp_enabled=True)}, "not supported"),
        ({"parallel": DiffletParallelConfig(tp_degree=1, cp_degree=2)}, "not supported"),
    ],
)
def test_shared_refusals_name_the_flag_and_the_model(over, expected):
    with pytest.raises(DiffletServingError) as exc:
        mod.refuse_shared_profile(_profile(**over), tag="[wan/cuda serve]")
    assert exc.value.code == "feature_not_supported"
    assert expected in exc.value.message and "[wan/cuda serve]" in exc.value.message


def test_a_release1_profile_passes_the_shared_refusals():
    mod.refuse_shared_profile(_profile(), tag="[x]")
