"""Resolve local and Hugging Face model paths."""

from __future__ import annotations

from pathlib import Path


def resolve_model_path(
    model_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str:
    path = Path(model_id).expanduser()
    if path.exists():
        return str(path.resolve())

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve non-local model ids"
        ) from exc

    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
