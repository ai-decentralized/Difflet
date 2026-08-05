from __future__ import annotations

import json
from pathlib import Path

from scripts.flux_cache_brake_intervention import canonical_sha256
from scripts.flux_cache_terminal_brake import _load_protocol


def test_registered_terminal_brake_protocol_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    protocol_path = root / "benchmark/flux_cache/terminal-brake-followup.json"
    protocol = json.loads(protocol_path.read_text())
    digest = protocol.pop("sha256")

    assert digest == canonical_sha256(protocol)


def test_terminal_brake_protocol_loads_bound_source_artifacts():
    root = Path(__file__).resolve().parents[3]

    protocol = _load_protocol(root / "benchmark/flux_cache/terminal-brake-followup.json")

    assert protocol["action"] == "terminal"


def test_registered_terminal_brake_result_digest_is_current():
    root = Path(__file__).resolve().parents[3]
    result = json.loads(
        (root / "benchmark/flux_cache/terminal-brake-followup-result.json").read_text()
    )
    digest = result.pop("sha256")

    assert digest == canonical_sha256(result)
