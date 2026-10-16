"""Test double for ``DiffletPipeline`` in the Wan CLI transformer stage (and the Qwen-Image CLI
generate stage) on the neuron backend.

The stage (``WanOrchestrator._stage_transformer_neuron``, ``QwenImageOrchestrator.
_stage_generate_neuron``) builds its application through
``DiffletPipeline.from_pretrained``; this module replaces that classmethod with
``fake_from_pretrained``, which returns a ``FakePipeline`` around a ``FakeWanApp`` on the CPU: no
weights, no device, no Neuron runtime. Not collected by pytest.

* In process, a test patches it in directly (``monkeypatch.setattr(DiffletPipeline,
  "from_pretrained", classmethod(fake_from_pretrained))``) and reads ``CALLS``.
* In the ranks of a real torchrun launch (``runner.run_stage`` -> torchrun ->
  ``difflet.cli.stage``), ``FAKE_PIPELINE_ENV`` selects it, the way
  ``tests/unit/backends/_neuron_toy.py`` selects its CPU device with ``TOY_DEVICE_ENV``: the test
  puts a ``sitecustomize.py`` (``write_sitecustomize``) first on ``PYTHONPATH``, and every Python
  process of the launch imports this module at start-up and calls ``install()``. Product code
  has no test hook. With ``FAKE_RECORDS_ENV`` set, each rank that built a pipeline writes
  ``rank<r>.json`` there at exit (``RankRecord``): the calls it received, every file the stage
  wrote (``torch.save`` and the stage manifest), its pid, and when it exited. Ranks other than
  0 linger ``LINGER_S`` at exit first, so a launcher that returned before every rank had exited
  would be caught by a test.

The fake app's latents are drawn from the request's generator, so every rank holds the same
tensor; under a process group they go through one gloo all-reduce, like the real DiT's.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FAKE_PIPELINE_ENV = "DIFFLET_TEST_FAKE_PIPELINE"
FAKE_RECORDS_ENV = "DIFFLET_TEST_FAKE_PIPELINE_RECORDS"
MODULE = "tests.unit.cli._fake_neuron_pipeline"
LATENT_SHAPE = (1, 16, 3, 4, 4)
COMPILED_BLOCKS = [f"blocks.{i}" for i in range(40)]
LINGER_S = 1.5

#: Every from_pretrained call in this process: {"model_id": ..., "kwargs": {...}, "pipeline": ...}.
CALLS: list[dict[str, Any]] = []
#: The ``backend.name`` the fake pipeline reports; "neuron" makes the stage track fallbacks.
BACKEND_NAME = "cpu"


def _rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _jsonable(value: Any) -> Any:
    def default(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return str(obj)

    return json.loads(json.dumps(value, default=default))


class FakeWanApp:
    """What the stage reads from ``TorchNeuronWanApplication``: ``load``, ``__call__`` and the
    evidence attributes (``compiled_blocks``, ``unwarmed_shapes``, ``phase_seconds``)."""

    def __init__(self) -> None:
        self.compiled_blocks = list(COMPILED_BLOCKS)
        self.unwarmed_shapes: list = []
        self.phase_seconds = {"warmup forward": 0.0}
        self.load_calls: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def load(self, compiled_model_path=None, start_rank_id=None, local_ranks_size=None,
             skip_warmup: bool = False) -> None:
        self.load_calls.append({"compiled_model_path": compiled_model_path,
                                "skip_warmup": skip_warmup})

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        import torch
        import torch.distributed as dist

        generator = kwargs["generator"]
        self.calls.append({**{k: v for k, v in kwargs.items() if k != "generator"},
                           "seed": generator.initial_seed()})
        latents = torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            dist.all_reduce(latents)  # every rank contributes the same tensor
            latents = latents / world
        return SimpleNamespace(latents=latents)


class FakePipeline:
    def __init__(self, *, app: FakeWanApp, compiled_path: Path) -> None:
        self.app = app
        self.compiled_path = compiled_path
        self.backend = SimpleNamespace(name=BACKEND_NAME)
        self.load_calls: list[dict[str, Any]] = []

    def load(self, *, start_rank_id=None, local_ranks_size=None, skip_warmup: bool = False):
        self.load_calls.append({"skip_warmup": skip_warmup})
        self.app.load(str(self.compiled_path), start_rank_id=start_rank_id,
                      local_ranks_size=local_ranks_size, skip_warmup=skip_warmup)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.app(*args, **kwargs)


def fake_from_pretrained(cls, model_id: str, **kwargs: Any) -> FakePipeline:
    """``DiffletPipeline.from_pretrained`` without weights or device. Under torchrun it starts
    the gloo process group itself, standing in for ``prepare_runtime``'s neuron group."""
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
    cache = kwargs.get("compile_cache_dir") or "/nonexistent-fake-cache"
    pipe = FakePipeline(app=FakeWanApp(), compiled_path=Path(cache) / "wan" / "fake")
    if kwargs.get("load", True):
        pipe.load(skip_warmup=kwargs.get("skip_warmup", False))
    CALLS.append({"model_id": model_id, "kwargs": kwargs, "pipeline": pipe})
    return pipe


# ------------------------------------------------------------- torchrun ranks (install)


class RankRecord:
    """What one rank of a torchrun launch did; written to FAKE_RECORDS_ENV at exit."""

    writes: list[dict[str, Any]] = []

    @classmethod
    def write(cls) -> None:
        records = os.environ.get(FAKE_RECORDS_ENV)
        if not records or not CALLS:
            return  # not a rank (the torchrun launcher), or the stage never built a pipeline
        rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
        if rank != 0:
            time.sleep(LINGER_S)
        pipe = CALLS[-1]["pipeline"]
        payload = {
            "rank": rank,
            "world_size": world,
            "pid": os.getpid(),
            "from_pretrained": [
                {"model_id": c["model_id"], "kwargs": _jsonable(c["kwargs"])} for c in CALLS
            ],
            "pipeline_load_calls": pipe.load_calls,
            "app_calls": _jsonable(pipe.app.calls),
            "writes": cls.writes,
            "exited_at": time.time(),
        }
        path = Path(records) / f"rank{rank}.json"
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        partial.rename(path)  # appears whole


def install() -> None:
    """Replace ``DiffletPipeline.from_pretrained`` in this process and record the stage's file
    writes (``torch.save``, ``write_stage_manifest``) with the writing rank."""
    import torch

    import difflet.cli.orchestrators.qwen_image as qwen_mod
    import difflet.cli.orchestrators.wan as wan_mod
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    DiffletPipeline.from_pretrained = classmethod(fake_from_pretrained)
    real_save = torch.save
    real_manifest = wan_mod.write_stage_manifest  # the base module's, imported by both

    def recorded_save(obj, f, *args, **kwargs):
        RankRecord.writes.append({"kind": "torch.save", "path": str(f),
                                  "rank": _rank_world()[0]})
        return real_save(obj, f, *args, **kwargs)

    def recorded_manifest(path, cache_inputs):
        RankRecord.writes.append({"kind": "manifest", "path": str(path),
                                  "rank": _rank_world()[0]})
        return real_manifest(path, cache_inputs)

    torch.save = recorded_save
    wan_mod.write_stage_manifest = recorded_manifest
    qwen_mod.write_stage_manifest = recorded_manifest
    atexit.register(RankRecord.write)


SITECUSTOMIZE = f"""\
import os

if os.environ.get({FAKE_PIPELINE_ENV!r}):
    import importlib

    importlib.import_module(os.environ[{FAKE_PIPELINE_ENV!r}]).install()
"""


def write_sitecustomize(directory) -> Path:
    """Write the start-up hook that installs the module FAKE_PIPELINE_ENV names; put
    ``directory`` first on PYTHONPATH."""
    path = Path(directory) / "sitecustomize.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SITECUSTOMIZE)
    return path


def read_rank_records(directory) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob("rank*.json"))]
