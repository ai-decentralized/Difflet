"""Helpers for `/v1/models` style metadata."""

from __future__ import annotations

from difflet.serving.model_registry import serving_models


def list_served_model_ids() -> list[str]:
    ids: list[str] = []
    for metadata in serving_models():
        ids.extend(metadata.checkpoint_ids)
    return ids
