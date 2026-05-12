"""Centralized environment variable registry for Nova.

Inspired by `vllm/envs.py` (via FastVideo). Every environment variable that
Nova reads from production code is declared here with a typed parser. Code
should access values lazily as ``nova.envs.NOVA_BACKEND`` etc. — never via
``os.environ.get`` directly — so that:

- the set of consumed env vars is discoverable in one file;
- types and defaults are documented next to the name;
- tests can monkey-patch the variable and immediately see the new value
  (lambdas are re-evaluated on every attribute access).

Writes that broadcast a value into the process environment for a downstream
consumer (e.g. ``os.environ["LOCAL_WORLD_SIZE"] = ...`` before a Neuron
subprocess is launched) are intentionally NOT modeled here. This module is
the *consumer* side of the boundary.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # ----- Nova-defined runtime config -----
    NOVA_BACKEND: str | None = None
    NOVA_COMPILE_CACHE: str = "~/.cache/nova"

    # ----- Distributed framework (set by torchrun / launcher / tests) -----
    RANK: int = 0
    WORLD_SIZE: int = 1
    LOCAL_RANK: int = 0
    LOCAL_WORLD_SIZE: int = 1
    MASTER_ADDR: str = "127.0.0.1"
    MASTER_PORT: str = "29500"

    # ----- Neuron runtime / compiler -----
    BASE_COMPILE_WORK_DIR: str = "/tmp/nxd_model/"
    NEURON_RT_VIRTUAL_CORE_SIZE: int = 1
    NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS: str | None = None
    NEURON_PLATFORM_TARGET_OVERRIDE: str | None = None
    NEURON_LOGICAL_NC_CONFIG: int = -1

    # ----- NxD framework (consumed by neuronx-distributed) -----
    NXD_CPU_MODE: str | None = None
    NXD_INFERENCE_CAPTURE_SNAPSHOT: bool = False
    NXD_INFERENCE_CAPTURE_OUTPUT_PATH: str | None = None
    NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT: str | None = None
    NXD_INFERENCE_CAPTURE_AT_REQUESTS: str | None = None
    NXD_INFERENCE_CAPTURE_FOR_TOKENS: str | None = None


# Doc generators and grep look for these markers.
# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Nova runtime config ==================

    # Active hardware backend. None means "infer from registry default or
    # platform detection". Set explicitly to "trainium" / "cuda" / "rocm" /
    # "cpu" to override.
    "NOVA_BACKEND":
    lambda: os.environ.get("NOVA_BACKEND"),

    # Root directory for the content-addressed AOT compile cache. Each
    # cache entry lives at ``<NOVA_COMPILE_CACHE>/<model>/<sha256-prefix>/``
    # with per-component ``model.pt`` + ``neuron_config.json``.
    "NOVA_COMPILE_CACHE":
    lambda: os.path.expanduser(
        os.environ.get("NOVA_COMPILE_CACHE", "~/.cache/nova")),

    # ================== Distributed framework ==================
    # These are typically set by torchrun / launcher / test fixtures. Nova
    # reads them; it does not set them in production paths.

    # Global rank of this process.
    "RANK":
    lambda: int(os.environ.get("RANK", "0")),

    # Total number of processes in the distributed group.
    "WORLD_SIZE":
    lambda: int(os.environ.get("WORLD_SIZE", "1")),

    # Local rank within the node.
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # Local world size on the node. Set by Nova modeling code before NxD
    # spawns its workers; read by NxD internals.
    "LOCAL_WORLD_SIZE":
    lambda: int(os.environ.get("LOCAL_WORLD_SIZE", "1")),

    # Distributed rendezvous endpoint.
    "MASTER_ADDR":
    lambda: os.environ.get("MASTER_ADDR", "127.0.0.1"),
    "MASTER_PORT":
    lambda: os.environ.get("MASTER_PORT", "29500"),

    # ================== Neuron runtime / compiler ==================

    # Working directory the Neuron compiler writes intermediate artifacts
    # to. ``MultiComponentApplication`` mutates this per component then
    # restores it; the registry default matches the historical Flux value.
    "BASE_COMPILE_WORK_DIR":
    lambda: os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/"),

    # NeuronCore virtualization size (1 = no virtual cores; 2 = pair two
    # physical cores into one logical NC for higher-memory components like
    # T5 text encoders).
    "NEURON_RT_VIRTUAL_CORE_SIZE":
    lambda: int(os.environ.get("NEURON_RT_VIRTUAL_CORE_SIZE", "1")),

    # Max inflight Neuron runtime requests; raised to "2" by some encoders.
    "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS":
    lambda: os.environ.get("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"),

    # Override target platform string surfaced to the compiler. Nova sets
    # this at import time when unset; readers see the resolved value.
    "NEURON_PLATFORM_TARGET_OVERRIDE":
    lambda: os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"),

    # Logical NeuronCore config tag. ``-1`` means "use config default".
    "NEURON_LOGICAL_NC_CONFIG":
    lambda: int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "-1")),

    # ================== NxD framework ==================

    # When "1", NxD forces a CPU-only path (used by unit tests).
    "NXD_CPU_MODE":
    lambda: os.environ.get("NXD_CPU_MODE"),

    # NxD inference capture hooks. Enabled iff snapshot flag is truthy;
    # output path/format and request/token filters refine where and when
    # captures fire.
    "NXD_INFERENCE_CAPTURE_SNAPSHOT":
    lambda: bool(os.environ.get("NXD_INFERENCE_CAPTURE_SNAPSHOT", "")),
    "NXD_INFERENCE_CAPTURE_OUTPUT_PATH":
    lambda: os.environ.get("NXD_INFERENCE_CAPTURE_OUTPUT_PATH"),
    "NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT":
    lambda: os.environ.get("NXD_INFERENCE_CAPTURE_OUTPUT_FORMAT"),
    "NXD_INFERENCE_CAPTURE_AT_REQUESTS":
    lambda: os.environ.get("NXD_INFERENCE_CAPTURE_AT_REQUESTS"),
    "NXD_INFERENCE_CAPTURE_FOR_TOKENS":
    lambda: os.environ.get("NXD_INFERENCE_CAPTURE_FOR_TOKENS"),
}

# end-env-vars-definition


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(environment_variables.keys())
