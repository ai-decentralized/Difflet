"""M5.0.3.2 — decode-free latent metric harness."""

import json

import torch

from difflet.pipeline.latent_metrics import LatentMetricCollector


def _traj(n, steps, *, converge_after=None, seed=0):
    """N candidates that start identical then diverge after a step."""
    g = torch.Generator().manual_seed(seed)
    base = [torch.randn(8, 16, generator=g) for _ in range(steps)]
    out = []
    for s in range(steps):
        cands = []
        for i in range(n):
            x = base[s].clone()
            if converge_after is not None and s >= converge_after and i > 0:
                x = x + 0.5 * torch.randn(8, 16, generator=g)
            cands.append(x)
        out.append(torch.stack(cands, dim=0))
    return out


def test_single_candidate_is_trivially_converged():
    coll = LatentMetricCollector()
    for s, lat in enumerate(_traj(1, 4)):
        coll.record(s, lat)
    m = coll.to_metrics_dict()
    assert m["num_candidates"] == 1
    assert m["shared_prefix_fraction"] == 1.0
    for step in m["per_step"]:
        assert len(step["cross_candidate_cosine_vs_0"]) == 1
        assert abs(step["cross_candidate_cosine_vs_0"][0] - 1.0) < 1e-5


def test_shared_prefix_detects_divergence_point():
    # 6 steps, candidates diverge starting at step 3 -> prefix = 3 steps.
    coll = LatentMetricCollector(fork_cosine_threshold=0.999)
    for s, lat in enumerate(_traj(4, 6, converge_after=3)):
        coll.record(s, lat)
    assert coll.shared_prefix_steps == 3
    assert coll.shared_prefix_fraction == 0.5
    m = coll.to_metrics_dict()
    assert m["per_step"][0]["candidates_converged"] is True
    assert m["per_step"][3]["candidates_converged"] is False


def test_cosine_vs_reference_recorded():
    coll = LatentMetricCollector()
    ref = torch.randn(4, 8, 16)
    for s in range(3):
        coll.record(s, ref.clone(), reference=ref)
    m = coll.to_metrics_dict()
    for step in m["per_step"]:
        assert all(abs(c - 1.0) < 1e-5 for c in step["cosine_vs_reference"])


def test_metrics_json_convention_and_roundtrip(tmp_path):
    coll = LatentMetricCollector()
    for s, lat in enumerate(_traj(2, 5, converge_after=2)):
        coll.record(s, lat)
    path = tmp_path / "m.json"
    coll.write_json(path)
    data = json.loads(path.read_text())

    # qwen-convention surface
    for key in (
        "schema",
        "num_steps",
        "per_step",
        "min_cross_candidate_cosine_mean",
        "min_cross_candidate_cosine_pstdev",
        "final_latents_shape",
        "final_latents_mean",
        "final_latents_std",
        "shared_prefix_fraction",
    ):
        assert key in data
    assert data["schema"] == "difflet-latent-metrics-v1"
    assert data["num_steps"] == 5
    assert len(data["per_step"]) == 5
    assert data["final_latents_shape"] == [2, 8, 16]


def test_deterministic():
    a = LatentMetricCollector()
    b = LatentMetricCollector()
    for s, lat in enumerate(_traj(3, 4, converge_after=2, seed=7)):
        a.record(s, lat)
    for s, lat in enumerate(_traj(3, 4, converge_after=2, seed=7)):
        b.record(s, lat)
    assert a.to_metrics_dict() == b.to_metrics_dict()
