"""Geometry config for the Qwen-Image transformer on TPU.

The Trainium side derives this from ``InferenceConfig``/``NeuronConfig``,
which are NxD types. The geometry itself is not Trainium-specific — it is
patch/VAE arithmetic — so this recomputes the same properties from the
diffusers ``config.json`` without dragging in NxD.

Kept deliberately as a plain dataclass rather than a shared base with the
Trainium config: the two backends need the same *numbers*, not a coupled
class hierarchy, and duplicating ~40 lines of arithmetic is cheaper than a
refactor of the NxD config on hardware we cannot run here.

Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

# The dataclass lives in difflet.models.qwen_image.dit_config so the GPU backends can share
# it; this name stays for the TPU path.
from difflet.models.qwen_image.dit_config import QwenImageDiTConfig  # noqa: F401

TpuQwenImageConfig = QwenImageDiTConfig

__all__ = ["TpuQwenImageConfig"]
