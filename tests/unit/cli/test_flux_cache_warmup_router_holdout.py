from __future__ import annotations

import json
from pathlib import Path

from scripts.flux_cache_brake_intervention import canonical_sha256
from scripts.evaluate_flux_cache_warmup_router_holdout import _load_protocol


def test_registered_warmup_router_holdout_protocol_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    path = root / "benchmark/flux_cache/warmup-vqa-router-holdout.json"
    protocol = json.loads(path.read_text())
    digest = protocol.pop("sha256")

    assert digest == canonical_sha256(protocol)


def test_warmup_router_holdout_is_fixed_before_generation(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    path = root / "benchmark/flux_cache/warmup-vqa-router-holdout.json"
    registered = json.loads(path.read_text())
    monkeypatch.setattr(
        "scripts.evaluate_flux_cache_warmup_router_holdout.python_source_sha256",
        lambda _root: registered["collection"]["python_source_sha256"],
    )

    protocol = _load_protocol(path)

    assert protocol["sample_matrix"]["expected_comparisons"] == 48
    assert protocol["sample_matrix"]["no_sample_expansion_after_scoring"] is True
