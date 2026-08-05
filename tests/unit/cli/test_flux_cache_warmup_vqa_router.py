from __future__ import annotations

import json
from pathlib import Path

from scripts.flux_cache_brake_intervention import canonical_sha256
from scripts.flux_cache_warmup_vqa_router import _load_protocol


def test_registered_warmup_vqa_router_protocol_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    path = root / "benchmark/flux_cache/warmup-vqa-router-pilot.json"
    protocol = json.loads(path.read_text())
    digest = protocol.pop("sha256")

    assert digest == canonical_sha256(protocol)


def test_warmup_vqa_router_protocol_reproduces_signal_dataset():
    root = Path(__file__).resolve().parents[3]

    protocol = _load_protocol(root / "benchmark/flux_cache/warmup-vqa-router-pilot.json")

    assert protocol["online_signal"]["window"] == [2, 5]
