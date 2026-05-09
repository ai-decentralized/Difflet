from pathlib import Path

import pytest

from nova import NovaParallelConfig, NovaPipeline, register_model
from nova.pipeline.nova_pipeline import _resolve_load_rank_range


class DummyApplication:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.compile_calls = []
        self.load_calls = []
        self.call_args = None

    def compile(self, compiled_model_path, debug=False):
        self.compile_calls.append((compiled_model_path, debug))
        Path(compiled_model_path).mkdir(parents=True, exist_ok=True)
        Path(compiled_model_path, "compiled.txt").write_text("ok\n", encoding="utf-8")

    def load(self, compiled_model_path, start_rank_id=None, local_ranks_size=None, skip_warmup=False):
        self.load_calls.append(
            {
                "path": compiled_model_path,
                "start_rank_id": start_rank_id,
                "local_ranks_size": local_ranks_size,
                "skip_warmup": skip_warmup,
            }
        )

    def __call__(self, *args, **kwargs):
        self.call_args = (args, kwargs)
        return {"args": args, "kwargs": kwargs}


def create_dummy_application(**kwargs):
    return DummyApplication(**kwargs)


@register_model(
    name="unit_dummy",
    application_factory=create_dummy_application,
    detector=lambda model_id: model_id.endswith("unit-dummy-model"),
    default_parallel=NovaParallelConfig(tp_degree=2),
    default_shape={"height": 64, "width": 64, "num_frames": None},
)
class _DummyRegistration:
    pass


def test_pipeline_compiles_and_loads_on_cache_miss(tmp_path):
    model_dir = tmp_path / "unit-dummy-model"
    model_dir.mkdir()

    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="unit_dummy",
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        debug_compile=True,
    )

    assert pipe.parallel == NovaParallelConfig(tp_degree=2)
    assert pipe.backend.name == "trainium"
    assert pipe.app.kwargs["backend"] == "trainium"
    assert pipe.shape == {"height": 64, "width": 64, "num_frames": None}
    assert len(pipe.app.compile_calls) == 1
    assert pipe.app.compile_calls[0][1] is True
    assert len(pipe.app.load_calls) == 1
    assert (pipe.compiled_path / "manifest.json").exists()


def test_pipeline_skips_compile_on_cache_hit(tmp_path):
    model_dir = tmp_path / "unit-dummy-model"
    model_dir.mkdir()
    cache_dir = tmp_path / "cache"

    NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="unit_dummy",
        dtype="bf16",
        compile_cache_dir=str(cache_dir),
    )
    second = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="unit_dummy",
        dtype="bf16",
        compile_cache_dir=str(cache_dir),
    )

    assert second.app.compile_calls == []
    assert len(second.app.load_calls) == 1


def test_parallel_config_rejects_conflicting_parallel_modes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        NovaParallelConfig(tp_degree=8, cp_enabled=True, cfg_parallel_enabled=True)


def test_pipeline_call_delegates_to_application(tmp_path):
    model_dir = tmp_path / "unit-dummy-model"
    model_dir.mkdir()

    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="unit_dummy",
        dtype="bf16",
        compile_cache_dir=str(tmp_path / "cache"),
        skip_compile=True,
        load=False,
    )

    result = pipe("prompt", steps=1)

    assert result == {"args": ("prompt",), "kwargs": {"steps": 1}}


# ---------------------------------------------------------------------------
# P1-P4 regression tests (added in M1.0.1).
# ---------------------------------------------------------------------------


def _spec(**overrides):
    """Helper: build a CacheSpec with sensible defaults for cache-key tests."""
    from nova.pipeline.compile_cache import CacheSpec

    defaults = dict(
        model_id="org/test-model",
        model_path="/tmp/some/local/path",
        model_name="unit_dummy",
        parallel=NovaParallelConfig(tp_degree=2),
        dtype="bf16",
        height=64,
        width=64,
        num_frames=None,
        revision=None,
    )
    defaults.update(overrides)
    return CacheSpec(**defaults)


def test_p2_dtype_string_and_torch_alias_collapse_to_same_key():
    """P2: dtype='bf16' and torch.bfloat16 must hash identically."""
    import torch

    from nova.pipeline.compile_cache import cache_key

    key_string = cache_key(_spec(dtype="bf16"))
    key_long = cache_key(_spec(dtype="bfloat16"))
    key_torch = cache_key(_spec(dtype=torch.bfloat16))

    assert key_string == key_long == key_torch, (
        f"dtype aliases diverged: bf16={key_string} bfloat16={key_long} torch={key_torch}"
    )


def test_p3_python_patch_version_excluded_from_cache_inputs():
    """P3: only major.minor python should appear in cache_inputs."""
    spec = _spec()
    py_version = spec.cache_inputs()["toolchain"]["python"]
    # Should be of the form "3.12", not "3.12.3".
    assert py_version.count(".") == 1, f"expected major.minor, got {py_version!r}"
    # Patch version should be available in metadata, however.
    py_full = spec.manifest_metadata()["python_full"]
    assert py_full.startswith(py_version + "."), (
        f"manifest_metadata.python_full={py_full!r} should extend cache_inputs.python={py_version!r}"
    )


def test_p4_model_path_does_not_affect_cache_key():
    """P4: same model_id resolved to different local paths share a cache key."""
    from nova.pipeline.compile_cache import cache_key

    key_a = cache_key(_spec(model_path="/cache_a/snapshots/abc/FLUX.1-dev"))
    key_b = cache_key(_spec(model_path="/different/host/cache/FLUX.1-dev"))

    assert key_a == key_b, "model_path must not influence cache key"

    # But it must still appear in the manifest for human inspection.
    md_a = _spec(model_path="/cache_a/.../FLUX.1-dev").manifest_metadata()
    assert md_a["model_path"] == "/cache_a/.../FLUX.1-dev"


def test_p4_changing_real_input_changes_cache_key():
    """P4 negative: legitimate input changes still invalidate the cache."""
    from nova.pipeline.compile_cache import cache_key

    base = cache_key(_spec())
    assert base != cache_key(_spec(model_id="org/different-model"))
    assert base != cache_key(_spec(parallel=NovaParallelConfig(tp_degree=4)))
    assert base != cache_key(_spec(height=128))
    assert base != cache_key(_spec(revision="v1.2.3"))


def test_p1_compile_recompiles_when_manifest_missing(tmp_path):
    """P1: pipe.compile(force=False) must trigger when manifest absent.

    Regression: previously compile() only checked directory existence, so a
    cache directory without (or with a corrupt) manifest would silently skip
    recompilation.
    """
    model_dir = tmp_path / "unit-dummy-model"
    model_dir.mkdir()
    cache_dir = tmp_path / "cache"

    # Build a pipeline without compiling.
    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="unit_dummy",
        dtype="bf16",
        compile_cache_dir=str(cache_dir),
        skip_compile=True,
        load=False,
    )
    assert pipe.app.compile_calls == []

    # Pre-create the cache directory but leave the manifest missing.
    pipe.compiled_path.mkdir(parents=True, exist_ok=True)
    assert not (pipe.compiled_path / "manifest.json").exists()

    # compile() should recognize the manifest is missing and recompile.
    pipe.compile()
    assert len(pipe.app.compile_calls) == 1
    assert (pipe.compiled_path / "manifest.json").exists()

    # Second compile() should hit the cache and skip.
    pipe.compile()
    assert len(pipe.app.compile_calls) == 1


def test_p1_compile_recompiles_when_manifest_payload_mismatched(tmp_path):
    """P1: a manifest with mismatched cache_inputs counts as miss."""
    import json as _json

    model_dir = tmp_path / "unit-dummy-model"
    model_dir.mkdir()
    cache_dir = tmp_path / "cache"

    pipe = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="unit_dummy",
        dtype="bf16",
        compile_cache_dir=str(cache_dir),
        skip_compile=True,
        load=False,
    )

    # Write a manifest that looks valid but with a tampered cache_inputs.
    pipe.compiled_path.mkdir(parents=True, exist_ok=True)
    manifest_file = pipe.compiled_path / "manifest.json"
    manifest_file.write_text(
        _json.dumps(
            {
                "schema_version": 1,
                "cache_key": "deadbeef",
                "cache_inputs": {"tampered": True},
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    pipe.compile()
    assert len(pipe.app.compile_calls) == 1


def test_force_compile_overrides_cache_hit(tmp_path):
    """from_pretrained(force_compile=True) recompiles even when cache valid."""
    model_dir = tmp_path / "unit-dummy-model"
    model_dir.mkdir()
    cache_dir = tmp_path / "cache"

    NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="unit_dummy",
        dtype="bf16",
        compile_cache_dir=str(cache_dir),
    )
    second = NovaPipeline.from_pretrained(
        str(model_dir),
        model_type="unit_dummy",
        dtype="bf16",
        compile_cache_dir=str(cache_dir),
        force_compile=True,
    )
    assert len(second.app.compile_calls) == 1, (
        "force_compile=True should bypass cache hit"
    )


def test_torchrun_load_defaults_to_one_rank_per_process(monkeypatch):
    """In torchrun MPMD each Python process owns one NeuronCore."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")

    assert _resolve_load_rank_range(start_rank_id=None, local_ranks_size=None) == (2, 1)


def test_explicit_load_rank_range_is_preserved(monkeypatch):
    """Manual load ranges still allow single-process multi-core loading."""
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")

    assert _resolve_load_rank_range(start_rank_id=0, local_ranks_size=4) == (0, 4)


def test_backend_override_must_be_supported_by_model(tmp_path):
    model_dir = tmp_path / "unit-dummy-model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="does not support backend 'cuda'"):
        NovaPipeline.from_pretrained(
            str(model_dir),
            model_type="unit_dummy",
            dtype="bf16",
            backend="cuda",
            compile_cache_dir=str(tmp_path / "cache"),
        )


def test_nova_backend_env_selects_backend_before_model_check(tmp_path, monkeypatch):
    model_dir = tmp_path / "unit-dummy-model"
    model_dir.mkdir()
    monkeypatch.setenv("NOVA_BACKEND", "cuda")

    with pytest.raises(ValueError, match="does not support backend 'cuda'"):
        NovaPipeline.from_pretrained(
            str(model_dir),
            model_type="unit_dummy",
            dtype="bf16",
            compile_cache_dir=str(tmp_path / "cache"),
        )


def test_backend_helpers_reflect_env(monkeypatch):
    from nova.backends import current_backend
    from nova.ops.platform import is_cuda, is_trainium

    monkeypatch.setenv("NOVA_BACKEND", "trainium")
    assert current_backend() == "trainium"
    assert is_trainium() is True
    assert is_cuda() is False

    monkeypatch.setenv("NOVA_BACKEND", "cuda")
    assert current_backend() == "cuda"
    assert is_trainium() is False
    assert is_cuda() is True
