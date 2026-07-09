from __future__ import annotations

from types import SimpleNamespace

import pytest

from difflet.pipeline import compile_cache
from difflet.serving.options import CompilePolicy
from difflet.serving.orchestrators import flux


def test_flux_preflight_never_fails_when_cache_not_ready(monkeypatch):
    preparer = flux.FluxServingArtifactPreparer()
    pipe = SimpleNamespace(
        compiled_path="/tmp/missing-flux",
        cache_spec=object(),
        app=SimpleNamespace(has_compiled_artifacts=lambda path: False),
    )
    monkeypatch.setattr(preparer, "_build_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(compile_cache, "has_valid_manifest", lambda path, spec: False)

    with pytest.raises(RuntimeError, match="missing Flux compiled artifacts"):
        preparer.ensure_artifacts(SimpleNamespace(), CompilePolicy.NEVER)
