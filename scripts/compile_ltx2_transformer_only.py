"""Compile the single-mode LTX-2 transformer to a device artifact (transformer ONLY).

The LTX-2 snapshot on this box has only ``transformer`` + ``scheduler`` (no text
encoder / VAE / connectors), so the host pipeline cannot be loaded. The e2e
TeaCache driver (scripts/run_ltx2_teacache_e2e.py) builds the app with
``enable_host_pipeline=True`` and reads the artifact under
.nova-cache/ltx_2_transformer_full with skip_compile=True.

To land the compiled ``model.pt`` at the cache hash the driver expects, we
compute the CacheSpec with the DRIVER's exact application_kwargs (which include
enable_host_pipeline=True) but build the actual application with
enable_host_pipeline=False / enable_decode_components=False so the missing text
encoder does not block the transformer compile. The host pipeline is irrelevant
to the transformer AOT graph; only ``app.components()`` (the transformer) is
compiled.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from nova import NovaParallelConfig
from nova.pipeline.compile_cache import (
    CacheSpec,
    cache_path,
    has_valid_manifest,
    write_manifest,
)
from nova.pipeline.path_resolver import resolve_model_path
from nova.registry import resolve_model

P = "[ltx2-compile]"

MODEL_DIR = (
    "/home/ubuntu/.cache/huggingface/hub/models--Lightricks--LTX-2/"
    "snapshots/47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
)
COMPILE_CACHE = "/home/ubuntu/nova/.nova-cache/ltx_2_transformer_full"
HEIGHT, WIDTH, NUM_FRAMES = 512, 768, 121
TEXT_SEQ_LEN = 1024
TP_DEGREE = 4

# EXACTLY the application_kwargs the e2e driver passes to from_pretrained.
# This is what gets hashed into the cache key — must match byte-for-byte so the
# driver finds our artifact. (Note: enable_host_pipeline=True is part of the
# hash but does NOT change the transformer graph.)
DRIVER_APP_KWARGS = {
    "transformer_mode": "single",
    "enable_host_pipeline": True,
    "enable_decode_components": False,
    "host_device": "cpu",
    "text_seq_len": TEXT_SEQ_LEN,
    "frame_rate": 24.0,
}


def main() -> int:
    dtype = torch.bfloat16
    parallel = NovaParallelConfig(tp_degree=TP_DEGREE)

    entry = resolve_model(MODEL_DIR, model_type="ltx_2")
    shape = entry.resolve_shape(height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES)
    model_path = resolve_model_path(MODEL_DIR, local_files_only=True)

    spec = CacheSpec(
        model_id=MODEL_DIR,
        model_path=model_path,
        model_name=entry.name,
        parallel=parallel,
        dtype=dtype,
        height=shape.get("height"),
        width=shape.get("width"),
        num_frames=shape.get("num_frames"),
        revision=None,
        application_kwargs=DRIVER_APP_KWARGS,
    )
    compiled_path = cache_path(COMPILE_CACHE, spec)
    print(f"{P} target compiled_path = {compiled_path}", flush=True)
    print(f"{P} cache hash = {compiled_path.name}", flush=True)

    # Build the app WITHOUT the host pipeline so the missing text encoder does
    # not block. Transformer-only kwargs; everything else mirrors the driver.
    build_kwargs = dict(DRIVER_APP_KWARGS)
    build_kwargs["enable_host_pipeline"] = False
    build_kwargs["enable_transformer"] = True

    print(f"{P} building NeuronLTX2Application (transformer-only)...", flush=True)
    app = entry.create_application(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        backend="trainium",
        application_kwargs=build_kwargs,
    )
    comps = app.components()
    print(f"{P} components = {[c.name for c in comps]}", flush=True)
    if not comps:
        print(f"{P} FAIL: no compile components (transformer not active)", file=sys.stderr)
        return 2

    compiled_path.mkdir(parents=True, exist_ok=True)
    print(f"{P} starting compile (this is long)...", flush=True)
    t0 = time.monotonic()
    app.compile(str(compiled_path))
    dt = time.monotonic() - t0
    print(f"{P} compile finished in {dt:.1f}s", flush=True)

    write_manifest(compiled_path, spec)
    print(f"{P} wrote manifest", flush=True)

    # Verify model.pt exists and is real.
    pt = compiled_path / "transformer" / "model.pt"
    if not pt.is_file():
        # some layouts nest one more level; search.
        found = list(compiled_path.rglob("model.pt"))
        print(f"{P} model.pt not at expected path; rglob found: {found}", flush=True)
        if not found:
            print(f"{P} FAIL: no model.pt produced", file=sys.stderr)
            return 3
        pt = found[0]
    size = pt.stat().st_size
    print(f"{P} OK model.pt = {pt} size={size} ({size/1e6:.1f} MB)", flush=True)
    if size < 24 * 1024:
        print(f"{P} WARN: model.pt is suspiciously small (<24KB)", file=sys.stderr)
    print(f"{P} manifest valid = {has_valid_manifest(compiled_path, spec)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
