"""Backend-neutral Wan stage/step clocks for `difflet generate`, injected via PYTHONPATH.

Neither Wan CLI path prints per-step timing, and the two backends launch the denoise loop
in different processes (the NxD ``transformer`` stage subprocess; one torchrun rank per
NeuronCore on the neuron backend). Both run the same ``WanOrchestrator._denoise`` loop
(difflet/models/wan/pipeline.py), and every iteration of it ends in ``_scheduler_step``.
With this directory first on PYTHONPATH and ``DIFFLET_WAN_PROBE_DIR`` set, every Python
process imports this module at startup; when ``difflet.models.wan.pipeline`` is imported it
patches, at class level (before any instance exists, so the neuron application's
per-instance ``_scheduler_step`` wrapper captures the patched method):

* ``_scheduler_step``: device-synchronize the returned latents (``torch.neuron.synchronize``
  when they live on the neuron device; NxD returns host tensors, already complete), then
  ``perf_counter`` — one stamp per denoise iteration, the same point on both backends;
* ``_denoise``: its wall and a stamp just before the loop (step 0 = first stamp minus it);
* ``encode_prompt``: wall per call.

On ``_denoise`` exit the process writes ``<DIFFLET_WAN_PROBE_DIR>/probe-<pid>-<n>.json``.
Nothing is imported or patched when the variable is unset. Python's own sitecustomize is
chained so the interpreter's behaviour is otherwise unchanged.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import time

_DIR = os.environ.get("DIFFLET_WAN_PROBE_DIR")
_TARGET = "difflet.models.wan.pipeline"
#: The denoising process's view of the runtime topology, kept with its timings.
_RECORDED_ENV = ("NEURON_RT_VIRTUAL_CORE_SIZE", "NEURON_LOGICAL_NC_CONFIG",
                 "NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES", "WORLD_SIZE")


def _chain_system_sitecustomize() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == here:
            continue
        path = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_chained_sitecustomize", path)
            if spec is not None and spec.loader is not None:
                try:
                    spec.loader.exec_module(importlib.util.module_from_spec(spec))
                except Exception:  # noqa: BLE001 - never break interpreter startup
                    pass
            return


def _sync(t) -> None:
    device = getattr(t, "device", None)
    if device is not None and device.type == "neuron":
        import torch

        torch.neuron.synchronize()


def _patch(module) -> None:
    cls = module.WanOrchestrator
    state = {"n": 0}
    record: dict = {}

    def reset() -> None:
        record.clear()
        record.update(pid=os.getpid(), rank=os.environ.get("RANK"), argv=sys.argv[:3],
                      env={k: os.environ.get(k) for k in _RECORDED_ENV},
                      encode_s=[], step_stamps=[])

    reset()
    scheduler_step, denoise, encode_prompt = cls._scheduler_step, cls._denoise, cls.encode_prompt

    def probed_scheduler_step(self, *args, **kwargs):
        latents = scheduler_step(self, *args, **kwargs)
        _sync(latents)
        record["step_stamps"].append(time.perf_counter())
        return latents

    def probed_encode_prompt(self, *args, **kwargs):
        start = time.perf_counter()
        out = encode_prompt(self, *args, **kwargs)
        _sync(out)
        record["encode_s"].append(time.perf_counter() - start)
        return out

    def probed_denoise(self, *args, **kwargs):
        record["denoise_start"] = time.perf_counter()
        out = denoise(self, *args, **kwargs)
        _sync(out)
        record["denoise_s"] = time.perf_counter() - record["denoise_start"]
        state["n"] += 1
        stamps = [record["denoise_start"], *record["step_stamps"]]
        record["step_s"] = [b - a for a, b in zip(stamps, stamps[1:])]
        os.makedirs(_DIR, exist_ok=True)
        if os.environ.get("RANK", "0") == "0":
            # Off the clocks: the initial noise and final latents, for the quality comparison.
            import torch

            torch.save({"initial": state.pop("initial", None), "final": out.detach().cpu()},
                       os.path.join(_DIR, f"latents-{os.getpid()}-{state['n']}.pt"))
        path = os.path.join(_DIR, f"probe-{os.getpid()}-{state['n']}.json")
        with open(path, "w") as fh:
            json.dump({k: v for k, v in record.items() if k != "step_stamps"}, fh)
        reset()
        return out

    prepare_latents = cls.prepare_latents

    def probed_prepare_latents(self, *args, **kwargs):
        latents = prepare_latents(self, *args, **kwargs)
        state["initial"] = latents.detach().cpu().clone()
        return latents

    cls.prepare_latents = probed_prepare_latents
    cls._scheduler_step = probed_scheduler_step
    cls._denoise = probed_denoise
    cls.encode_prompt = probed_encode_prompt


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


_chain_system_sitecustomize()
if _DIR:
    sys.meta_path.insert(0, _Finder())
