"""DiffletPipeline keying of the tensor-parallel all-reduce dtype."""

from __future__ import annotations

import inspect

import pytest

import difflet.registry as registry
from difflet.pipeline import difflet_pipeline as dp

BUILTIN_MODELS = {
    "flux", "wan", "hunyuan_video", "hunyuan_video_15", "qwen_image", "minimax_h3", "ltx_2",
}


def test_every_builtin_model_has_a_tp_reduce_decision():
    # A new built-in model must be added to BUILTIN_MODELS and to one of the
    # tables, so it cannot silently share one key between precisions.
    builtins = [name for name, _ in inspect.getmembers(registry) if name.startswith("_register_builtin_")]
    assert len(builtins) == len(BUILTIN_MODELS)
    decided = set(dp._TP_REDUCE_LEGACY) | dp._TP_REDUCE_UNAFFECTED
    assert BUILTIN_MODELS <= decided, f"undecided: {sorted(BUILTIN_MODELS - decided)}"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (None, {"text_mask": "valid-len", "tp_reduce_dtype": "bfloat16"}),
        ({"tp_reduce_dtype": "bf16"}, {"text_mask": "valid-len", "tp_reduce_dtype": "bfloat16"}),
        # The fp32 reduce adds no field; the masked DiT's graph revision is always named, so no
        # pre-mask artifact is reused.
        ({"tp_reduce_dtype": "fp32"}, {"text_mask": "valid-len"}),
    ],
)
def test_qwen_image_is_keyed(kwargs, expected):
    assert dp._cache_application_kwargs(kwargs, model_name="qwen_image") == expected


@pytest.mark.parametrize("model_name", sorted(dp._TP_REDUCE_UNAFFECTED))
def test_unaffected_models_add_nothing(model_name):
    assert dp._cache_application_kwargs({"tp_reduce_dtype": "fp32"}, model_name=model_name) in (
        None,
        {"attention_mask_accumulation": "fp32-v1"},
    )


def test_unlisted_model_gets_no_field_and_never_hashes_the_raw_value():
    assert dp._cache_application_kwargs({"tp_reduce_dtype": "fp32"}, model_name="toy") is None
