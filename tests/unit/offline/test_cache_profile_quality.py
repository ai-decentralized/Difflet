from __future__ import annotations

from difflet.offline.cache_profile.quality import metric_identity


def test_metric_identity_ignores_runtime_and_transient_cache_files():
    stable = {"path": "/cache/model.bin", "bytes": 10, "sha256": "a" * 64}
    left = {
        "image_reward": {
            "model": "ImageReward-v1.0",
            "load_seconds": 1.0,
            "checkpoint_files": [
                stable,
                {"path": "/cache/model.bin.metadata", "sha256": "b" * 64},
            ],
        },
        "vqa_score": {"model": "clip-flant5-xl", "load_seconds": 2.0},
    }
    right = {
        "image_reward": {
            "model": "ImageReward-v1.0",
            "load_seconds": 9.0,
            "checkpoint_files": [
                stable,
                {"path": "/cache/model.bin.metadata", "sha256": "c" * 64},
            ],
        },
        "vqa_score": {"model": "clip-flant5-xl", "load_seconds": 8.0},
    }

    assert metric_identity(left) == metric_identity(right)
