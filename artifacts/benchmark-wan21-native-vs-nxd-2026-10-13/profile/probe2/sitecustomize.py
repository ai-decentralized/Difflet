"""probe2: host-side per-step decomposition of the Wan denoise loop, both backends.

Activated by DIFFLET_WAN_PROBE2_DIR. Patches (at import of the target modules):
  * WanOrchestrator._denoise: wraps self.transformer (and transformer_2) in a timing proxy for
    the duration of the loop -> per DiT call wall (host tensors in -> host tensor out);
    step stamps after each _scheduler_step (device-synced like the benchmark probe);
    the scheduler step's own duration.
  * neuron only: _OnDeviceTransformer.__call__ re-implemented with syncs to split
    h2d / module forward / d2h (sync after each part; the sync after the forward is also in the
    original). With DIFFLET_WAN_PROBE2_SEGMENTS=1 hooks on the eager top-level submodules split
    the forward into rope / patch_embed / cond_embed / blocks / head (each stamped after a
    torch.neuron.synchronize(), so it serialises host dispatch with device execution).
Writes <dir>/probe2-<pid>-<n>.json on _denoise exit (every rank).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import time

_DIR = os.environ.get("DIFFLET_WAN_PROBE2_DIR")
_SEGMENTS = os.environ.get("DIFFLET_WAN_PROBE2_SEGMENTS") == "1"
_TARGETS = {"difflet.models.wan.pipeline", "difflet.models.wan.neuron_application"}
_STATE: dict = {"n": 0, "rec": None}


def _now():
    return time.perf_counter()


def _sync_dev():
    import torch

    if hasattr(torch, "neuron"):
        try:
            torch.neuron.synchronize()
        except Exception:  # noqa: BLE001
            pass


def _sync(t) -> None:
    device = getattr(t, "device", None)
    if device is not None and device.type == "neuron":
        _sync_dev()


def _rec():
    return _STATE["rec"]


class _TimedModel:
    """Proxy for the orchestrator's transformer: times each call, delegates attributes."""

    def __init__(self, inner, name):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_inner"), item)

    def __call__(self, *args, **kwargs):
        start = _now()
        out = object.__getattribute__(self, "_inner")(*args, **kwargs)
        first = out[0] if isinstance(out, (tuple, list)) else out
        _sync(first)
        rec = _rec()
        if rec is not None:
            rec["dit_call_s"].append(_now() - start)
        return out


def _patch_pipeline(module) -> None:
    cls = module.WanOrchestrator
    scheduler_step, denoise = cls._scheduler_step, cls._denoise

    def probed_scheduler_step(self, *args, **kwargs):
        start = _now()
        latents = scheduler_step(self, *args, **kwargs)
        _sync(latents)
        end = _now()
        rec = _rec()
        if rec is not None:
            rec["sched_s"].append(end - start)
            rec["stamps"].append(end)
        return latents

    def probed_denoise(self, *args, **kwargs):
        rec = {"pid": os.getpid(), "rank": os.environ.get("RANK"), "dit_call_s": [],
               "sched_s": [], "stamps": [], "h2d_s": [], "fwd_s": [], "d2h_s": [],
               "seg": [], "segments_enabled": _SEGMENTS}
        _STATE["rec"] = rec
        saved = {}
        for attr in ("transformer", "transformer_2"):
            inner = getattr(self, attr, None)
            if inner is not None:
                saved[attr] = inner
                setattr(self, attr, _TimedModel(inner, attr))
        rec["denoise_start"] = _now()
        _event("denoise:begin")
        try:
            out = denoise(self, *args, **kwargs)
            _sync(out)
        finally:
            for attr, inner in saved.items():
                setattr(self, attr, inner)
        rec["denoise_s"] = _now() - rec["denoise_start"]
        _event("denoise:end")
        stamps = [rec["denoise_start"], *rec.pop("stamps")]
        rec["step_s"] = [b - a for a, b in zip(stamps, stamps[1:])]
        _STATE["n"] += 1
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, f"probe2-{os.getpid()}-{_STATE['n']}.json"), "w") as fh:
            json.dump(rec, fh)
        _STATE["rec"] = None
        return out

    cls._scheduler_step = probed_scheduler_step
    cls._denoise = probed_denoise


def _install_segment_hooks(module) -> None:
    """Hooks on the eager top-level submodules of WanTransformer3DModel (never on blocks:
    those are torch.compile'd with fullgraph=True and a sync inside would break the graph)."""
    marks: list = []

    def mark(name):
        def hook(*_a, **_k):
            _sync_dev()
            marks.append((name, _now()))
        return hook

    def start_hook(*_a):
        marks.clear()
        _sync_dev()
        marks.append(("start", _now()))
        return None

    module.register_forward_pre_hook(start_hook)
    module.rope.register_forward_hook(mark("rope"))
    module.patch_embedding.register_forward_hook(mark("patch_embed"))
    module.condition_embedder.register_forward_pre_hook(mark("flatten"))
    module.condition_embedder.register_forward_hook(mark("cond_embed"))
    module.norm_out.register_forward_pre_hook(mark("blocks"))
    module.proj_out.register_forward_hook(mark("norm_proj"))

    def end_hook(*_a):
        _sync_dev()
        marks.append(("unpatchify", _now()))
        rec = _rec()
        if rec is not None and marks and marks[0][0] == "start":
            seg = {}
            for (_, a), (name, b) in zip(marks, marks[1:]):
                seg[name] = b - a
            rec["seg"].append(seg)
        return None

    module.register_forward_hook(end_hook)


def _patch_neuron_app(module) -> None:
    cls = module._OnDeviceTransformer

    def timed_call(self, hidden_states, timestep, encoder_hidden_states):
        import torch

        if _SEGMENTS and not getattr(self, "_probe2_hooked", False):
            inner = getattr(self.module, "module", None)
            if inner is not None:
                _install_segment_hooks(inner)
            self._probe2_hooked = True
        t0 = _now()
        args = tuple(
            tensor.to(self.dtype).contiguous().to(self.device)
            for tensor in (hidden_states, timestep, encoder_hidden_states)
        )
        if self.device.type == "neuron":
            torch.neuron.synchronize()
        t1 = _now()
        out = self.module(*args)
        if self.device.type == "neuron":
            torch.neuron.synchronize()
        t2 = _now()
        out = out.cpu()
        t3 = _now()
        self.calls += 1
        rec = _rec()
        if rec is not None:
            rec["h2d_s"].append(t1 - t0)
            rec["fwd_s"].append(t2 - t1)
            rec["d2h_s"].append(t3 - t2)
        hook = self.forward_hook if self.hook_from is None else self.hook_from.forward_hook
        if hook is not None:
            hook(t3 - t0)
        return out

    cls.__call__ = timed_call




# ------------------------------------------------------------------ event timeline
def _proc_start_epoch():
    try:
        with open("/proc/self/stat") as fh:
            start_ticks = int(fh.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK")
        return time.time() - uptime + start_ticks / hz
    except Exception:  # noqa: BLE001
        return None


def _event(name, **extra):
    if not _DIR:
        return
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, f"events-{os.getpid()}.jsonl"), "a") as fh:
            fh.write(json.dumps({"t": time.time(), "ev": name, "pid": os.getpid(),
                                 "rank": os.environ.get("RANK"), **extra}) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _wrap(owner, attr, label, after=None):
    orig = getattr(owner, attr)

    def wrapped(*args, **kwargs):
        _event(label + ":begin")
        try:
            return orig(*args, **kwargs)
        finally:
            extra = {}
            if after is not None:
                try:
                    extra = after(*args) or {}
                except Exception:  # noqa: BLE001
                    extra = {}
            _event(label + ":end", **extra)

    wrapped.__wrapped__ = orig
    setattr(owner, attr, wrapped)


def _phase_info(self, *_a):
    return {"cls": type(self).__name__,
            "phase_seconds": {k: round(v, 3) for k, v in getattr(self, "phase_seconds", {}).items()}}


def _patch_cli_wan(module):
    cls = module.WanOrchestrator
    for attr in ("_stage_transformer", "_stage_transformer_neuron", "_stage_vae", "_stage_vae_neuron"):
        if hasattr(cls, attr):
            _wrap(cls, attr, attr)
    _wrap(module, "_write_frames", "write_frames")


def _patch_runner(module):
    _wrap(module, "run_stage", "run_stage")


def _patch_neuron_base(module):
    _wrap(module.TorchNeuronApplicationBase, "load", "neuron_app_load", after=_phase_info)
    group = int(os.environ.get("DIFFLET_PROBE2_BLOCK_GROUP", "1"))
    if group > 1:
        import torch

        orig_compile_blocks = module.compile_blocks

        class _BlockGroup(torch.nn.Module):
            """``group`` consecutive Wan blocks run as one module (one compiled graph)."""

            def __init__(self, members):
                super().__init__()
                self.members = torch.nn.ModuleList(members)

            def forward(self, hidden_states, *args):
                for member in self.members:
                    hidden_states = member(hidden_states, *args)
                return hidden_states

        def grouped_compile_blocks(model, *, block_attrs=("blocks",), **kwargs):
            if type(model).__name__ == "WanTransformer3DModel" and tuple(block_attrs) == ("blocks",):
                blocks = list(model.blocks)
                if len(blocks) % group == 0:
                    model.blocks = torch.nn.ModuleList(
                        [_BlockGroup(blocks[i:i + group]) for i in range(0, len(blocks), group)])
                    _event("probe2_block_group", group=group, groups=len(model.blocks))
            return orig_compile_blocks(model, block_attrs=block_attrs, **kwargs)

        module.compile_blocks = grouped_compile_blocks


def _patch_trn_base(module):
    _wrap(module.NeuronApplicationBase, "load", "trn_app_load", after=lambda self, *a: {"cls": type(self).__name__})


def _patch_trn_multi(module):
    _wrap(module.MultiComponentApplication, "load", "trn_multi_load")


def _patch_pipeline_events(module):
    cls = module.WanOrchestrator
    for attr in ("__call__", "encode_prompt", "_decode_latents"):
        _wrap(cls, attr, "pipe" + attr)


def _patch_neuron_runtime(module):
    _wrap(module.NeuronBackend, "prepare_runtime", "prepare_runtime")
    _wrap(module, "init_process_group", "init_process_group")


def _patch_difflet_pipeline(module):
    cls = module.DiffletPipeline
    orig = cls.from_pretrained  # bound classmethod

    def from_pretrained(*args, **kwargs):
        _event("from_pretrained:begin")
        try:
            return orig(*args, **kwargs)
        finally:
            _event("from_pretrained:end")

    cls.from_pretrained = staticmethod(from_pretrained)
    _wrap(cls, "load", "difflet_pipeline_load")


def _mark_import(module):
    _event("imported:" + module.__name__)


def _patch_neuron_compile(module):
    if os.environ.get("DIFFLET_PROBE2_WLO") == "1":
        from types import MappingProxyType

        module.NEURON_COMPILE_OPTIONS = MappingProxyType(
            {**module.NEURON_COMPILE_OPTIONS, "weight_layout_optimization": True})
        _event("probe2_wlo_enabled", options=dict(module.NEURON_COMPILE_OPTIONS))


_PATCHERS = {
    "difflet.models.wan.pipeline": _patch_pipeline,
    "difflet.models.wan.neuron_application": _patch_neuron_app,
    "difflet.cli.orchestrators.wan": _patch_cli_wan,
    "difflet.cli.runner": _patch_runner,
    "difflet.backends.neuron.compile": _patch_neuron_compile,
    "difflet.backends.neuron.runtime": _patch_neuron_runtime,
    "difflet.pipeline.difflet_pipeline": _patch_difflet_pipeline,
    "torch": _mark_import,
    "torch_neuronx": _mark_import,
    "difflet.backends.neuron.core.application_base": _patch_neuron_base,
    "difflet.backends.trainium.core.application_base": _patch_trn_base,
    "difflet.backends.trainium.core.multi_component_application": _patch_trn_multi,
}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PATCHERS or name in _STATE.setdefault("seen", set()):
            return None
        _STATE["seen"].add(name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(mod):
            exec_module(mod)
            _PATCHERS[name](mod)
            if name == "difflet.models.wan.pipeline":
                _patch_pipeline_events(mod)

        spec.loader.exec_module = exec_and_patch
        return spec


if _DIR:
    import atexit

    _event("proc_start", argv=sys.argv[:4], ppid=os.getppid(), exec_start=_proc_start_epoch())
    atexit.register(lambda: _event("proc_exit"))
    sys.meta_path.insert(0, _Finder())
