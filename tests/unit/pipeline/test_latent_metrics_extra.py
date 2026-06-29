"""Extra tests for difflet.pipeline.latent_metrics targeting branches not
covered by tests/unit/pipeline/test_latent_metrics.py."""

import json

import pytest
import torch

from difflet.pipeline.latent_metrics import (
    DEFAULT_FORK_COSINE_THRESHOLD,
    LatentMetricCollector,
)


def test_record_rejects_scalar_latents():
    coll = LatentMetricCollector()
    with pytest.raises(ValueError):
        coll.record(0, torch.tensor(1.0))  # ndim == 0


def test_empty_collector_metrics_defaults():
    coll = LatentMetricCollector()
    m = coll.to_metrics_dict()
    assert m["num_steps"] == 0
    assert m["num_candidates"] == 0
    assert m["min_cross_candidate_cosine_mean"] == 1.0
    assert m["min_cross_candidate_cosine_pstdev"] == 0.0
    assert m["shared_prefix_fraction"] == 0.0
    assert "final_latents_shape" not in m


def test_shared_prefix_fraction_empty_is_zero():
    coll = LatentMetricCollector()
    assert coll.shared_prefix_fraction == 0.0
    assert coll.shared_prefix_steps == 0


def test_single_step_pstdev_is_zero_and_final_recorded():
    torch.manual_seed(0)
    coll = LatentMetricCollector()
    coll.record(0, torch.randn(2, 3))
    m = coll.to_metrics_dict()
    assert m["num_steps"] == 1
    assert m["min_cross_candidate_cosine_pstdev"] == 0.0
    assert m["final_latents_shape"] == [2, 3]
    assert "final_latents_mean" in m
    assert "final_latents_std" in m


def test_pstdev_branch_with_multiple_steps():
    torch.manual_seed(0)
    coll = LatentMetricCollector(fork_cosine_threshold=0.5)
    coll.record(0, torch.randn(2, 4))
    coll.record(1, torch.randn(2, 4))
    m = coll.to_metrics_dict()
    assert len(m["per_step_min_cross_candidate_cosine"]) == 2


def test_reference_per_candidate_when_shapes_match():
    torch.manual_seed(0)
    coll = LatentMetricCollector()
    latents = torch.randn(3, 4)
    coll.record(0, latents, reference=latents.clone())  # reference.shape[0] == n
    entry = coll.steps[0]
    assert len(entry["cosine_vs_reference"]) == 3
    assert all(abs(c - 1.0) < 1e-5 for c in entry["cosine_vs_reference"])


def test_reference_broadcast_when_shapes_differ():
    torch.manual_seed(0)
    coll = LatentMetricCollector()
    latents = torch.randn(3, 4)
    single_ref = torch.randn(1, 4)  # shape[0] != n -> broadcast same ref
    coll.record(0, latents, reference=single_ref)
    entry = coll.steps[0]
    assert len(entry["cosine_vs_reference"]) == 3


def test_write_json_creates_parent_and_roundtrips(tmp_path):
    torch.manual_seed(0)
    coll = LatentMetricCollector()
    coll.record(0, torch.randn(2, 4))
    out = tmp_path / "nested" / "metrics.json"
    coll.write_json(out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "difflet-latent-metrics-v1"


def test_default_threshold_constant():
    coll = LatentMetricCollector()
    assert coll.fork_cosine_threshold == DEFAULT_FORK_COSINE_THRESHOLD
