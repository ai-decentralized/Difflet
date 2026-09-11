# >>> NxDI fork banner — managed by scripts/add_fork_banner.py >>>
# Forked from neuronx-distributed-inference v0.9.17334+ced6ae4e
# Original path: neuronx_distributed_inference/utils/diffusers_adapter.py
# Fork date: 2026-05-08
# Modifications: InferenceConfig import is annotation-only (TYPE_CHECKING) so the
#   module imports on hosts without the Neuron toolchain (TPU backend).
# <<< NxDI fork banner <<<
from __future__ import annotations

from diffusers.configuration_utils import ConfigMixin
from typing import TYPE_CHECKING, Optional, Union
import os

if TYPE_CHECKING:  # only used in a type annotation below
    from difflet.backends.trainium.core.config import InferenceConfig


def load_diffusers_config(
    model_path_or_name: Optional[Union[str, os.PathLike]] = None,
    hf_config: Optional[ConfigMixin] = None,
):
    """Return a load_config hook for InferenceConfig that loads the config from a config.json for diffuser models."""
    class DiffusersConfig(ConfigMixin):
        config_name = "config.json"

        def __init__(self):
            super().__init__()

    def load_config(self: InferenceConfig):
        if (model_path_or_name is None and hf_config is None) or (
            model_path_or_name is not None and hf_config is not None
        ):
            raise ValueError('Please provide only one of "model_path_or_name" or "hf_config"')

        if model_path_or_name is not None:
            config = DiffusersConfig()
            config = config.load_config(model_path_or_name)
        else:
            config = hf_config
        config["_name_or_path"] = model_path_or_name  # we need this attribute to load weight.
        self.__dict__.update(config)
    return load_config
