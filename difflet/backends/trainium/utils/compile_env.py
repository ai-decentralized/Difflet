# >>> NxDI fork banner — managed by scripts/add_fork_banner.py >>>
# Forked from neuronx-distributed-inference v0.9.17334+ced6ae4e
# Original path: neuronx_distributed_inference/utils/compile_env.py
# Fork date: 2026-05-08
# Modifications: (none — verbatim copy; see git log for divergence)
# <<< NxDI fork banner <<<
import collections
import os
from difflet.backends.trainium.core.config import NeuronConfig


def get_compile_env_vars(neuron_config: NeuronConfig) -> dict[str, str]:
    """
    Get environment variables needed for Neuron compilation based on the provided configuration.

    Args:
        neuron_config (NeuronConfig): config contains env var info

    Returns:
        dict[str, str]: Dictionary of environment variables and their values needed for compilation.
                        Keys and values are both strings.

    Example:
        >>> config = NeuronConfig(disable_numeric_cc_token=True)
        >>> get_compile_env_vars(config)
        {'DISABLE_NUMERIC_CC_TOKEN': '1'}
    """
    env_vars = collections.defaultdict()
    if neuron_config.disable_numeric_cc_token:
        env_vars.update({"DISABLE_NUMERIC_CC_TOKEN": "1"})

    return env_vars


def set_compile_env_vars(neuron_config: NeuronConfig) -> None:
    """
    Set "compile-time" environment variables based on the provided configuration.
    These environment variables are required to be set before model tracing/compilation to take effect.

    Args:
        neuron_config (NeuronConfig): config contains env var info
    """

    env_vars = get_compile_env_vars(neuron_config)
    for var, value in env_vars.items():
        if var not in os.environ:
            os.environ[var] = str(value)
