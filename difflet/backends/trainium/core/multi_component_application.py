"""Lifecycle orchestration for Trainium applications with multiple components."""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn

from difflet import envs

logger = logging.getLogger(__name__)

COMPILED_MODEL_FILE_NAME = "model.pt"
NEURON_CONFIG_FILE_NAME = "neuron_config.json"
DEFAULT_COMPILE_WAIT_TIMEOUT_S = 7200


@dataclass(frozen=True)
class ComponentSpec:
    """A Trainium sub-application and its artifact/load ordering metadata."""

    name: str
    component: Any
    world_size: int | None = None
    load_priority: int | None = None
    artifact_name: str | None = None


class MultiComponentApplication(nn.Module, ABC):
    """Base for Trainium applications composed of multiple Neuron sub-apps.

    This class owns the Trainium lifecycle FSM: race-safe per-component AOT
    compile, artifact validation, and rank-lockstep load ordering. It does not
    model the prompt/latent forward dataflow.
    """

    @abstractmethod
    def components(self) -> list[ComponentSpec]:
        """Return active components in compile order."""

    def no_components_message(self, action: str) -> str:
        return f"{self.__class__.__name__} {action} requires at least one active component"

    def compile(
        self,
        compiled_model_path: str,
        debug: bool = False,
        select: Collection[str] | None = None,
    ) -> None:
        specs = self._selected_components(select)
        if not specs:
            raise NotImplementedError(self.no_components_message("compile"))

        compiler_workdir = envs.BASE_COMPILE_WORK_DIR
        try:
            for spec in specs:
                self._compile_one(spec, compiled_model_path, compiler_workdir, debug)
        finally:
            os.environ["BASE_COMPILE_WORK_DIR"] = compiler_workdir

    def load(
        self,
        compiled_model_path: str,
        start_rank_id: int | None = None,
        local_ranks_size: int | None = None,
        skip_warmup: bool = False,
        select: Collection[str] | None = None,
    ) -> None:
        specs = self._selected_components(select)
        if not specs:
            raise NotImplementedError(self.no_components_message("load"))

        for spec in self._load_ordered(specs):
            self._load_one(
                spec,
                compiled_model_path,
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                skip_warmup=skip_warmup,
            )

    def has_compiled_artifacts(
        self,
        compiled_model_path: str,
        select: Collection[str] | None = None,
    ) -> bool:
        specs = self._selected_components(select)
        if not specs:
            return True

        for spec in specs:
            component_path = self._component_path(compiled_model_path, spec)
            if not (component_path / COMPILED_MODEL_FILE_NAME).exists():
                return False
            if not (component_path / NEURON_CONFIG_FILE_NAME).exists():
                return False
            if not self._compiled_config_matches(spec, component_path):
                return False
        return True

    def _selected_components(self, select: Collection[str] | None) -> list[ComponentSpec]:
        specs = list(self.components())
        if select is None:
            return specs

        selected_names = {select} if isinstance(select, str) else set(select)
        by_name = {spec.name: spec for spec in specs}
        missing = sorted(selected_names.difference(by_name))
        if missing:
            known = ", ".join(sorted(by_name)) or "<none>"
            raise ValueError(
                f"Unknown component(s) for {self.__class__.__name__}: {missing}; "
                f"known components: {known}"
            )
        return [spec for spec in specs if spec.name in selected_names]

    def _load_ordered(self, specs: list[ComponentSpec]) -> list[ComponentSpec]:
        indexed_specs = list(enumerate(specs))
        indexed_specs.sort(key=lambda item: (self._load_priority(item[1]), item[0]))
        return [spec for _index, spec in indexed_specs]

    def _load_priority(self, spec: ComponentSpec) -> int:
        if spec.load_priority is not None:
            return spec.load_priority
        return -self._component_world_size(spec)

    def _compile_one(
        self,
        spec: ComponentSpec,
        compiled_model_path: str,
        compiler_workdir: str,
        debug: bool,
    ) -> None:
        component_path = self._component_path(compiled_model_path, spec)
        compiled_marker = component_path / COMPILED_MODEL_FILE_NAME

        _spmd_barrier()
        if compiled_marker.exists() and self._compiled_config_matches(spec, component_path):
            logger.info(
                "%s already compiled at %s, skipping compilation.",
                spec.name,
                component_path,
            )
        elif not self._should_compile_component(spec):
            logger.info(
                "Waiting for compile owner to build replicated component %s at %s.",
                spec.name,
                component_path,
            )
            self._wait_for_compiled_component(spec, component_path, compiled_marker)
        else:
            os.environ["BASE_COMPILE_WORK_DIR"] = str(Path(compiler_workdir) / spec.name)
            spec.component.compile(str(component_path), debug=debug)

        _spmd_barrier()
        if not compiled_marker.exists() or not self._compiled_config_matches(spec, component_path):
            raise RuntimeError(
                f"{self.__class__.__name__} component {spec.name} was not compiled "
                f"correctly at {component_path}"
            )

    def _load_one(
        self,
        spec: ComponentSpec,
        compiled_model_path: str,
        *,
        start_rank_id: int | None,
        local_ranks_size: int | None,
        skip_warmup: bool,
    ) -> None:
        component_path = self._component_path(compiled_model_path, spec)
        component_start_rank_id, component_local_ranks_size = self._component_load_rank_range(
            spec.component,
            start_rank_id=start_rank_id,
            local_ranks_size=local_ranks_size,
        )
        logger.info("Loading %s component from %s", spec.name, component_path)
        _spmd_barrier()
        spec.component.load(
            str(component_path),
            start_rank_id=component_start_rank_id,
            local_ranks_size=component_local_ranks_size,
            skip_warmup=skip_warmup,
        )
        _spmd_barrier()
        logger.info("Loaded %s component from %s", spec.name, component_path)

    def _should_compile_component(self, spec: ComponentSpec) -> bool:
        rank, dist_world_size = _dist_rank_world()
        if dist_world_size > 1 and self._component_world_size(spec) == 1:
            return rank == 0
        return True

    def _wait_for_compiled_component(
        self,
        spec: ComponentSpec,
        component_path: Path,
        compiled_marker: Path,
    ) -> None:
        deadline = time.monotonic() + DEFAULT_COMPILE_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            if compiled_marker.exists() and self._compiled_config_matches(spec, component_path):
                return
            time.sleep(2)
        raise RuntimeError(
            f"Timed out waiting for component {spec.name} to compile at {component_path}"
        )

    def _compiled_config_matches(self, spec: ComponentSpec, component_path: Path) -> bool:
        config_path = component_path / NEURON_CONFIG_FILE_NAME
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)["neuron_config"]
        except (OSError, KeyError, json.JSONDecodeError):
            return False

        current = self._component_neuron_config(spec.component)
        keys = ("tp_degree", "world_size", "start_rank_id", "local_ranks_size")
        for key in keys:
            current_value = getattr(current, key, None)
            if saved.get(key) != current_value:
                logger.info(
                    "Compiled component config mismatch at %s: %s saved=%r current=%r; "
                    "recompiling.",
                    component_path,
                    key,
                    saved.get(key),
                    current_value,
                )
                return False
        return True

    def _component_world_size(self, spec: ComponentSpec) -> int:
        if spec.world_size is not None:
            return int(spec.world_size)
        return int(getattr(self._component_neuron_config(spec.component), "world_size", 1))

    @staticmethod
    def _component_neuron_config(component: Any) -> Any:
        config = getattr(component, "config", None)
        return getattr(config, "neuron_config", None)

    @staticmethod
    def _component_load_rank_range(
        component: Any,
        *,
        start_rank_id: int | None,
        local_ranks_size: int | None,
    ) -> tuple[int | None, int | None]:
        config = getattr(component, "config", None)
        neuron_config = getattr(config, "neuron_config", None)
        world_size = getattr(neuron_config, "world_size", None)
        if world_size == 1:
            return 0 if start_rank_id is not None else None, 1
        return start_rank_id, local_ranks_size

    @staticmethod
    def _component_path(compiled_model_path: str, spec: ComponentSpec) -> Path:
        return Path(compiled_model_path) / (spec.artifact_name or spec.name)


def _spmd_barrier() -> None:
    """Synchronize initialized torch.distributed ranks; no-op otherwise."""

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
    except Exception:  # pragma: no cover - defensive: lifecycle should not crash on barrier
        pass


def _dist_rank_world() -> tuple[int, int]:
    """Return rank/world for torchrun-style launches."""

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass

    try:
        return envs.RANK, envs.WORLD_SIZE
    except ValueError:
        return 0, 1
