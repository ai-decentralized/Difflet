"""Pure-Python facts about the HunyuanVideo 1.5 AOT DiT graph: the image-token count, the
compiler args, the graph version and the diagnostic switches.

They key the AOT stage cache (the orchestrator's ``_aot_stage_cache_inputs``) and configure
the trainium backbone (``backends/trainium/hunyuan_video/backbone15.py``, which re-exports
them). Kept free of the Neuron SDK so the cache key is computable, and testable, on any host.
"""

from __future__ import annotations

import os

_HV15_SPATIAL_COMPRESSION_RATIO = 16
_HV15_TEMPORAL_COMPRESSION_RATIO = 4


def hv15_num_image_tokens(
    height: int,
    width: int,
    num_frames: int,
    *,
    spatial_compression_ratio: int = _HV15_SPATIAL_COMPRESSION_RATIO,
    temporal_compression_ratio: int = _HV15_TEMPORAL_COMPRESSION_RATIO,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> int:
    """Full image-stream length N after patchify for a pixel shape (the one source of N)."""
    frames = (int(num_frames) - 1) // int(temporal_compression_ratio) + 1
    lat_h = int(height) // int(spatial_compression_ratio)
    lat_w = int(width) // int(spatial_compression_ratio)
    return (frames // int(patch_size_t)) * (lat_h // int(patch_size)) * (lat_w // int(patch_size))


# --enable-ccop-compute-overlap speeds the DiT up on short sequences but trips NCC_IRPX901
# (RelaxPredicates, text-stream LayerNorm) on long ones. Measured (tp2cfg): with it, 320x512x61
# (N = 16*20*32 = 10240) runs at ~1602 ms/step vs 1816.5 ms without (13.4% slower); with it,
# 480x848x121 (N = 31*30*53 = 49290) fails to compile, without it it works. So keep it up to
# this image-token count N (latent_frames * latent_height * latent_width, patch size 1).
_HV15_CCOP_OVERLAP_MAX_IMAGE_TOKENS = 16384
_HV15_DIT_COMPILER_ARGS_HEAD = "--model-type=transformer -O1 "
_HV15_DIT_COMPILER_ARGS_TAIL = (
    "--auto-cast=none "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
)
_HV15_CCOP_OVERLAP_ARGS = "--tensorizer-options='--enable-ccop-compute-overlap' "


def hv15_dit_compiler_args(num_image_tokens: int) -> str:
    """The effective DiT compiler args for an image-token count N: the env override if set
    (verbatim), else the defaults, with the ccop overlap option only for N <= the threshold."""
    override = os.environ.get("DIFFLET_HUNYUAN15_COMPILER_ARGS")
    if override:
        return override
    if int(num_image_tokens) <= _HV15_CCOP_OVERLAP_MAX_IMAGE_TOKENS:
        return _HV15_DIT_COMPILER_ARGS_HEAD + _HV15_CCOP_OVERLAP_ARGS + _HV15_DIT_COMPILER_ARGS_TAIL
    return _HV15_DIT_COMPILER_ARGS_HEAD + _HV15_DIT_COMPILER_ARGS_TAIL


# The traced DiT graph's version, part of the AOT stage cache key (the orchestrator's
# ``_aot_stage_cache_inputs``), so a changed graph compiles into a fresh dir instead of
# loading a stale artifact. Bump it whenever the traced DiT graph's inputs or structure
# change. 2: ``text_valid`` became graph input #10 (no in-graph mask count, NCC_IRPX901).
_AOT_DIT_GRAPH_VERSION = 2

# Diagnostic switches read by ``_HunyuanVideo15TraceModule`` (capacity / attention probes):
# each changes the traced graph, so the set ones are part of the AOT stage cache key too.
_HV15_DIAGNOSTIC_GRAPH_SWITCHES = (
    "DIFFLET_HUNYUAN15_NKI_ATTENTION",
    "DIFFLET_HUNYUAN15_NO_ATTENTION_MASK",
    "DIFFLET_HUNYUAN15_ATTENTION_QUERY_CHUNK_SIZE",
    "DIFFLET_HUNYUAN15_ZERO_BLOCK_ATTN",
    "DIFFLET_HUNYUAN15_ZERO_BLOCK_FF",
)


def hv15_diagnostic_graph_switches() -> dict[str, str]:
    """The diagnostic graph switches set in the environment (empty in production)."""
    return {name: os.environ[name] for name in _HV15_DIAGNOSTIC_GRAPH_SWITCHES if os.environ.get(name)}
