"""Top-level checkpoint conversion entry point.

Reads a HuggingFace Wan2.2 snapshot (e.g. ``Wan-AI/Wan2.2-T2V-A14B-Diffusers``)
and writes Difflet-compatible safetensors that NXD's load path can ingest
without further renaming.

Layout produced (mirrors the upstream snapshot, with renamed keys):

    <out_dir>/
    ├── transformer/
    │   └── model.safetensors                  # or sharded *.safetensors[.index.json]
    ├── transformer_2/
    │   └── ...
    └── text_encoder/
        └── ...

This is the *file-level* output. NXD per-rank pre-sharding (writing
``weights/tp{rank}_sharded_checkpoint.safetensors``) happens inside
``NeuronApplicationBase.compile()`` when ``skip_sharding=False`` and
``save_sharded_checkpoint=True``. The conversion produced here is the
**HF state dict transformation** step that runs *before* NXD sharding.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Iterable

from difflet.models.wan.checkpoint.backbone import convert_backbone_state_dict
from difflet.models.wan.checkpoint.text_encoder import convert_text_encoder_state_dict
from difflet.models.wan.checkpoint.vae import convert_vae_decoder_state_dict

logger = logging.getLogger(__name__)


# Map of component subdirectory → conversion function. New components
# (e.g. ``vae``) plug in here.
COMPONENT_CONVERTERS: dict[str, Callable] = {
    "transformer": convert_backbone_state_dict,
    "transformer_2": convert_backbone_state_dict,
    "text_encoder": convert_text_encoder_state_dict,
    "vae": convert_vae_decoder_state_dict,
}


def convert_diffusers_checkpoint(
    model_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    components: Iterable[str] | None = None,
    *,
    overwrite: bool = False,
    max_shard_size: str = "5GB",
) -> dict[str, str]:
    """Read HF safetensors from ``model_dir/<component>/`` and write
    Difflet-renamed safetensors to ``out_dir/<component>/``.

    Args:
        model_dir: snapshot root (containing ``transformer/``, ``text_encoder/``,
            etc.).
        out_dir: output root. Components are written to subdirectories.
        components: which components to convert. ``None`` → every entry of
            ``COMPONENT_CONVERTERS`` whose subdirectory exists in ``model_dir``.
        overwrite: if False (default), skip a component when its output dir
            already has ``model.safetensors`` or shard files.
        max_shard_size: passed through to
            ``save_state_dict_safetensors`` for HF-style sharded output.

    Returns:
        Mapping ``component_name -> output_subdir`` for components that were
        successfully written.

    Raises:
        FileNotFoundError: ``model_dir`` doesn't exist or no component
            subdirectory exists.
    """
    # Imports kept inside the function so this module can be imported and
    # type-checked without dragging in heavy state-dict loaders.
    from difflet.backends.trainium.core.modules.checkpoint import (
        load_state_dict,
        save_state_dict_safetensors,
    )

    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    if components is None:
        components = [
            name for name in COMPONENT_CONVERTERS if (model_dir / name).is_dir()
        ]
    else:
        components = list(components)

    if not components:
        raise FileNotFoundError(
            f"no convertable components found under {model_dir}; "
            f"expected one or more of {sorted(COMPONENT_CONVERTERS)}"
        )

    out_paths: dict[str, str] = {}
    for component in components:
        if component not in COMPONENT_CONVERTERS:
            raise ValueError(f"unknown component {component!r}")

        comp_in = model_dir / component
        comp_out = out_dir / component
        if not comp_in.is_dir():
            logger.warning("skipping %s (not present in %s)", component, model_dir)
            continue

        if not overwrite and _output_exists(comp_out):
            logger.info("skipping %s (output already exists at %s)", component, comp_out)
            out_paths[component] = str(comp_out)
            continue

        logger.info("converting %s: %s -> %s", component, comp_in, comp_out)
        state_dict = load_state_dict(str(comp_in))
        converter = COMPONENT_CONVERTERS[component]
        state_dict = converter(state_dict, config=None)

        comp_out.mkdir(parents=True, exist_ok=True)
        save_state_dict_safetensors(
            state_dict, str(comp_out), max_shard_size=max_shard_size
        )
        out_paths[component] = str(comp_out)
        logger.info("wrote %s (%d tensors)", comp_out, len(state_dict))

    return out_paths


def _output_exists(comp_out: Path) -> bool:
    if not comp_out.is_dir():
        return False
    if (comp_out / "model.safetensors").exists():
        return True
    if (comp_out / "model.safetensors.index.json").exists():
        return True
    return False


__all__ = ["COMPONENT_CONVERTERS", "convert_diffusers_checkpoint"]
