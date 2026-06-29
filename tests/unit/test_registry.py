"""Unit tests for difflet.registry targeting branch/error paths."""

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.registry import (
    ModelEntry,
    _resolve_factory,
    register_model,
    registered_models,
    resolve_model,
)


def _factory(**kwargs):
    return ("app", kwargs)


# ---------------------------------------------------------------------------
# ModelEntry behavior
# ---------------------------------------------------------------------------
def test_matches_by_hf_path():
    entry = ModelEntry(name="e", application_factory=_factory, hf_paths=("org/Model",))
    assert entry.matches("org/Model") is True
    assert entry.matches("org/Model/") is True  # trailing slash stripped
    assert entry.matches("org/Other") is False


def test_matches_by_detector():
    entry = ModelEntry(
        name="e", application_factory=_factory, detector=lambda mid: "yes" in mid
    )
    assert entry.matches("a-yes-b") is True
    assert entry.matches("no") is False


def test_matches_returns_false_without_paths_or_detector():
    entry = ModelEntry(name="e", application_factory=_factory)
    assert entry.matches("anything") is False


def test_resolve_shape_overrides_defaults():
    entry = ModelEntry(
        name="e",
        application_factory=_factory,
        default_shape={"height": 10, "width": 20, "num_frames": None},
    )
    shape = entry.resolve_shape(height=100, num_frames=8)
    assert shape == {"height": 100, "width": 20, "num_frames": 8}


def test_create_application_invokes_factory():
    entry = ModelEntry(name="e", application_factory=_factory)
    result = entry.create_application(
        model_path="/p",
        parallel=DiffletParallelConfig(),
        dtype="bf16",
        shape={"height": 1},
        backend="trainium",
        application_kwargs={"extra": 1},
    )
    app, kwargs = result
    assert app == "app"
    assert kwargs["extra"] == 1
    assert kwargs["backend"] == "trainium"


def test_require_backend_passes_and_fails():
    entry = ModelEntry(name="e", application_factory=_factory, backends=("trainium",))
    entry.require_backend("trainium")  # no raise
    with pytest.raises(ValueError):
        entry.require_backend("cuda")


# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------
def test_register_model_missing_factory_raises():
    with pytest.raises(ValueError):

        @register_model(name="no_factory_model_xyz")
        class _Bad:
            pass


def test_register_and_resolve_custom_model():
    @register_model(
        name="zzz_unique_registry_test_model",
        application_factory=_factory,
        detector=lambda mid: "zzzuniqueid" in mid,
    )
    class _Reg:
        pass

    entry = resolve_model("some-zzzuniqueid-model")
    assert entry.name == "zzz_unique_registry_test_model"
    assert entry in registered_models()


# ---------------------------------------------------------------------------
# resolve_model error paths
# ---------------------------------------------------------------------------
def test_resolve_model_unknown_model_type():
    with pytest.raises(ValueError):
        resolve_model("anything", model_type="not_a_registered_type_qqq")


def test_resolve_model_no_match_raises():
    with pytest.raises(ValueError):
        resolve_model("totally-unmatched-model-id-9z9z9z")


def test_resolve_model_multiple_matches_raises():
    @register_model(
        name="dup_match_a_test",
        application_factory=_factory,
        detector=lambda mid: "dupmatchtoken" in mid,
    )
    class _A:
        pass

    @register_model(
        name="dup_match_b_test",
        application_factory=_factory,
        detector=lambda mid: "dupmatchtoken" in mid,
    )
    class _B:
        pass

    with pytest.raises(ValueError):
        resolve_model("x-dupmatchtoken-y")


def test_resolve_model_by_model_type():
    @register_model(
        name="explicit_type_test_model",
        application_factory=_factory,
    )
    class _T:
        pass

    entry = resolve_model("ignored", model_type="explicit_type_test_model")
    assert entry.name == "explicit_type_test_model"


# ---------------------------------------------------------------------------
# _resolve_factory
# ---------------------------------------------------------------------------
def test_resolve_factory_callable_passthrough():
    assert _resolve_factory(_factory) is _factory


def test_resolve_factory_from_string():
    fn = _resolve_factory("difflet.registry:resolve_model")
    assert fn is resolve_model


def test_resolve_factory_invalid_reference():
    with pytest.raises(ValueError):
        _resolve_factory("no_colon_here")


def test_resolve_factory_not_callable():
    with pytest.raises(TypeError):
        _resolve_factory("difflet.registry:_REGISTRY")


def test_register_uses_class_attribute_factory():
    @register_model(name="class_attr_factory_model_test")
    class _WithAttr:
        application_factory = staticmethod(_factory)

    entry = resolve_model("ignored", model_type="class_attr_factory_model_test")
    assert callable(_resolve_factory(entry.application_factory))


def test_builtin_models_registered():
    names = {entry.name for entry in registered_models()}
    assert {"flux", "wan", "hunyuan_video", "qwen_image", "ltx_2"} <= names
