import json
from pathlib import Path

from scripts.flux_cache_brake_intervention import canonical_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = ROOT / "benchmark/flux_cache/component-residual-signal-result.json"


def test_registered_component_residual_signal_result_is_bound() -> None:
    document = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    digest = document.pop("sha256")
    assert digest == canonical_sha256(document)

    for binding in document["artifacts"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        assert sha256_file(path) == binding["file_sha256"]


def test_component_residual_signal_cannot_claim_a_serving_threshold() -> None:
    document = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert document["serving_claim"] is False
    assert document["real_trajectory_mapping"]["preselected_raw_signal_gate_passed"] is False
    assert document["end_to_end_quality_mapping"]["quality_gate_passed"] is False
    assert document["decision"]["component_error_online_brake"] == "rejected"
