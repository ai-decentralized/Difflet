"""bf16 copies of fp32 DiT checkpoints, for components that run in bf16.

Some checkpoints store their DiT in fp32 (Wan 2.2's experts, HunyuanVideo 1.5's transformer) while it
runs in bf16: loading casts every tensor, so a cold start reads twice the bytes the model uses. The
Trainium path pays the cast once, at compile time, in its pre-sharded weights; this module does the
same with a bf16 copy of the component directory, made once by ``difflet compile`` and preferred
at load. The copy keeps the original's layout (shard files, index, ``config.json``), so loaders are
unchanged; rounding fp32 to bf16 offline gives bit-identical weights to rounding at load.

The copy lives under ``<DIFFLET_COMPILE_CACHE>/weights-bf16/<snapshot>/<component>/`` with a
manifest of the source files (name, size, modification time); a source that no longer matches is
ignored at load and rebuilt at the next compile. A copy is written to a temporary directory and
renamed when complete, so an interrupted conversion is never used.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

MANIFEST = "difflet-bf16-manifest.json"


def _cache_root() -> Path:
    return Path(os.path.expanduser(os.environ.get("DIFFLET_COMPILE_CACHE", "~/.cache/difflet")))


def copy_dir(component_dir: str | os.PathLike, cache_root: str | os.PathLike | None = None) -> Path:
    """Where the bf16 copy of ``component_dir`` lives: keyed by the snapshot directory (a
    Hugging Face snapshot is named by its revision) and the component name."""
    src = Path(component_dir).resolve()
    root = Path(cache_root).expanduser() if cache_root is not None else _cache_root()
    return root / "weights-bf16" / src.parent.name / src.name


def _source_manifest(src: Path) -> dict:
    files = {}
    for path in sorted(src.iterdir()):
        if path.is_file():
            stat = path.stat()
            files[path.name] = [stat.st_size, stat.st_mtime_ns]
    return {"source": str(src), "files": files, "dtype": "bfloat16"}


def _has_fp32(src: Path) -> bool:
    from safetensors import safe_open

    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if f.get_slice(key).get_dtype() == "F32":
                    return True
    return False


def valid_copy(component_dir: str | os.PathLike, cache_root=None) -> Path | None:
    """The bf16 copy if it exists and matches its source, else None."""
    src = Path(component_dir).resolve()
    dst = copy_dir(src, cache_root)
    try:
        manifest = json.loads((dst / MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    return dst if manifest == _source_manifest(src) else None


def ensure_bf16_copy(component_dir: str | os.PathLike, cache_root=None) -> Path | None:
    """Make the bf16 copy of ``component_dir`` unless a valid one exists or nothing is fp32.
    Returns the copy's directory, or None when the checkpoint has no fp32 tensors."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(component_dir).resolve()
    existing = valid_copy(src, cache_root)
    if existing is not None:
        return existing
    if not _has_fp32(src):
        return None
    dst = copy_dir(src, cache_root)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for path in sorted(src.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            tensors = {}
            with safe_open(str(path), framework="pt") as f:
                metadata = f.metadata()
                for key in f.keys():
                    t = f.get_tensor(key)
                    tensors[key] = t.to(torch.bfloat16) if t.dtype == torch.float32 else t
            save_file(tensors, str(tmp / path.name), metadata=metadata)
            del tensors
        else:  # config.json, the shard index (same file names, so its weight map still holds)
            shutil.copy2(path, tmp / path.name)
    (tmp / MANIFEST).write_text(json.dumps(_source_manifest(src)))
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return dst


def resolve_component_dir(component_dir: str | os.PathLike, dtype, cache_root=None) -> str:
    """The directory to load a component from: its bf16 copy when the component runs in bf16 and
    a valid copy exists, else the original."""
    if dtype == torch.bfloat16:
        copy = valid_copy(component_dir, cache_root)
        if copy is not None:
            return str(copy)
    return str(component_dir)
