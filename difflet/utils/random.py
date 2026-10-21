# >>> NxDI fork banner — managed by scripts/add_fork_banner.py >>>
# Forked from neuronx-distributed-inference v0.9.17334+ced6ae4e
# Original path: neuronx_distributed_inference/utils/random.py
# Fork date: 2026-05-08
# Modifications: neuronx_distributed is imported lazily so this module (and
#   difflet.utils.testing, which uses it) imports on hosts without the Neuron
#   SDK — CUDA backend plan, Phase 1 task 1.6.
# <<< NxDI fork banner <<<
import importlib.util
import random

import numpy as np
import torch


def _nxd():
    """Return ``neuronx_distributed`` when it is installed, else ``None``.

    Kept as a function so tests can substitute a stub and so the import cost is
    paid only on Neuron hosts.
    """
    if importlib.util.find_spec("neuronx_distributed") is None:
        return None
    import neuronx_distributed as nxd

    return nxd


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    nxd = _nxd()
    if nxd is not None and nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
        nxd.parallel_layers.random.model_parallel_xla_manual_seed(seed)
