"""V3 observer for `difflet generate --backend neuron` (Qwen-Image): stage clocks, host RAM
peaks, per-rank latents sha256, compile counters. Active only with DIFFLET_V3_PROBE=1 and this
directory on PYTHONPATH. It wraps methods (calls the originals unchanged) and prints one
`[v3probe] {json}` line per event; it changes no argument, tensor or control flow.
"""
import os
import sys

if os.environ.get("DIFFLET_V3_PROBE"):
    import importlib.abc
    import json
    import time

    _T0 = time.time()

    def _rank():
        return os.environ.get("RANK", "driver")

    def _status_kb():
        out = {}
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith(("VmHWM:", "VmRSS:")):
                        key, value = line.split(":", 1)
                        out[key + "_gib"] = round(int(value.split()[0]) / 2**20, 3)
        except OSError:
            pass
        return out

    def _emit(event, **fields):
        rec = {"event": event, "rank": _rank(), "t_epoch": round(time.time(), 3),
               "t_since_start_s": round(time.time() - _T0, 3), **_status_kb(), **fields}
        print("[v3probe] " + json.dumps(rec, default=str), flush=True)

    def _device_extras():
        out = {}
        try:
            import torch_neuronx

            out["max_memory_allocated_gib"] = round(torch_neuronx.max_memory_allocated() / 2**30, 3)
            out["memory_allocated_gib"] = round(torch_neuronx.memory_allocated() / 2**30, 3)
        except Exception as exc:  # noqa: BLE001
            out["device_memory_error"] = repr(exc)
        try:
            import torch_neuronx.metrics as metrics

            out["counters"] = {n: metrics.get_counter_value("CompilationCache." + n) or 0
                               for n in ("TotalCompilations", "PersistentHits", "InMemoryHits")}
        except Exception as exc:  # noqa: BLE001
            out["counters_error"] = repr(exc)
        return out

    def _sha(tensor):
        import hashlib

        import torch

        t = tensor.detach().to("cpu").contiguous()
        raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
        return hashlib.sha256(raw.numpy().tobytes()).hexdigest()

    def _patch_app(module):
        cls = module.TorchNeuronQwenImageApplication
        orig_load, orig_encode, orig_generate = cls.load, cls.encode_prompt, cls.generate

        def load(self, *args, **kwargs):
            t = time.perf_counter()
            result = orig_load(self, *args, **kwargs)
            self._v3_stamps, self._v3_fwd = [], []
            self.step_hook = lambda: self._v3_stamps.append(time.perf_counter())
            if self.transformer_adapter is not None:
                self.transformer_adapter.forward_hook = self._v3_fwd.append
            pipe = self.pipeline
            orig_denoise = pipe._denoise

            def denoise(*a, **k):
                s = time.perf_counter()
                out = orig_denoise(*a, **k)
                self._v3_denoise_s = time.perf_counter() - s
                self._v3_denoise_start = s
                return out

            pipe._denoise = denoise
            save_traj = os.environ.get("DIFFLET_V3_PROBE_SAVE_TRAJ")
            feed_path = os.environ.get("DIFFLET_V3_PROBE_FEED")
            if save_traj or feed_path:
                # Diagnostics only: record every DiT call's input/output (rank 0 saves them) and,
                # with FEED, replace each call's hidden_states by a recorded trajectory's
                # (teacher forcing: per-step parity on identical inputs; the image is then
                # meaningless).
                import dataclasses

                import torch

                feed = torch.load(feed_path, map_location="cpu") if feed_path else None

                class _Recorder:
                    def __init__(rec, inner):
                        rec.inner, rec.dtype, rec.config = inner, inner.dtype, inner.config
                        rec.inputs, rec.outputs, rec.timesteps = [], [], []

                    def __call__(rec, bundle):
                        i = len(rec.inputs)
                        if feed is not None:
                            bundle = dataclasses.replace(
                                bundle, hidden_states=feed["inputs"][i].to(
                                    bundle.hidden_states.dtype))
                        rec.inputs.append(bundle.hidden_states.detach().cpu().clone())
                        rec.timesteps.append(bundle.timestep.detach().cpu().clone())
                        out = rec.inner(bundle)
                        rec.outputs.append(out.detach().cpu().clone())
                        if save_traj and _rank() == "0" and len(rec.inputs) == self._v3_steps:
                            torch.save({"inputs": rec.inputs, "outputs": rec.outputs,
                                        "timesteps": rec.timesteps, "fed": feed_path}, save_traj)
                            _emit("traj_saved", path=save_traj, n=len(rec.inputs))
                        return out

                self._v3_steps = int(os.environ.get("DIFFLET_V3_PROBE_STEPS", "50"))
                pipe.transformer = _Recorder(pipe.transformer)
            _emit("app_load", app_load_s=round(time.perf_counter() - t, 3),
                  phase_seconds={k: round(v, 3) for k, v in self.phase_seconds.items()},
                  compiled_blocks=len(self.compiled_blocks), **_device_extras())
            return result

        def encode_prompt(self, prompt):
            s = time.perf_counter()
            embeds, mask = orig_encode(self, prompt)
            dt = time.perf_counter() - s
            dump = os.environ.get("DIFFLET_V3_PROBE_EMBEDS")
            if dump and _rank() == "0":
                import torch

                torch.save({"prompt": prompt, "embeds": embeds, "mask": mask}, dump)
            _emit("encode_prompt", encode_s=round(dt, 3), valid_rows=int(mask.sum()),
                  embeds_sha256=_sha(embeds), embeds_absmean=float(embeds.float().abs().mean()))
            return embeds, mask

        def generate(self, **kwargs):
            s = time.perf_counter()
            self._v3_stamps.clear()
            self._v3_fwd.clear()
            out = orig_generate(self, **kwargs)
            total = time.perf_counter() - s
            stamps = self._v3_stamps
            deltas = [b - a for a, b in zip(stamps, stamps[1:])]  # step 0 excluded
            ds = sorted(deltas)
            fw = sorted(self._v3_fwd)

            def pct(v, q):
                return round(v[min(len(v) - 1, int(q * len(v)))] * 1000, 2) if v else None

            lat = out.latents
            _emit("generate", generate_s=round(total, 3),
                  denoise_s=round(getattr(self, "_v3_denoise_s", float("nan")), 3),
                  pre_denoise_s=round(getattr(self, "_v3_denoise_start", s) - s, 3),
                  steps=len(stamps), step0_ms=round((stamps[0] - self._v3_denoise_start) * 1000, 2)
                  if stamps else None,
                  step_ms={"n": len(ds), "median": pct(ds, 0.5), "p10": pct(ds, 0.1),
                           "p90": pct(ds, 0.9), "min": pct(ds, 0.0),
                           "max": round(ds[-1] * 1000, 2) if ds else None,
                           "mean": round(sum(ds) / len(ds) * 1000, 2) if ds else None},
                  forward_ms={"n": len(fw), "median": pct(fw, 0.5), "p90": pct(fw, 0.9)},
                  latents_shape=list(lat.shape), latents_dtype=str(lat.dtype),
                  latents_sha256=_sha(lat), latents_finite=bool(lat.float().isfinite().all()),
                  latents_std=float(lat.float().std()), **_device_extras())
            return out

        cls.load, cls.encode_prompt, cls.generate = load, encode_prompt, generate
        _emit("patched", module=module.__name__)

    def _patch_orch(module):
        if _rank() != "driver":
            try:  # counters read 0 until enabled; before prepare_runtime in the stage
                import torch_neuronx.metrics as metrics

                metrics.set_enabled(True)
                _emit("metrics_enabled", enabled=bool(metrics.is_enabled()))
            except Exception as exc:  # noqa: BLE001
                _emit("metrics_enabled", error=repr(exc))
            return
        orig = module._decode_latents_host

        def decode(latents_path, model_dir, output_path, **kw):
            s = time.perf_counter()
            orig(latents_path, model_dir, output_path, **kw)
            import torch

            _emit("host_decode", decode_s=round(time.perf_counter() - s, 3),
                  torch_threads=torch.get_num_threads(), latents_pt=latents_path,
                  latents_pt_sha256=_sha(torch.load(latents_path, map_location="cpu")))

        module._decode_latents_host = decode
        _emit("patched", module=module.__name__)

    _TARGETS = {
        "difflet.models.qwen_image.neuron_application": _patch_app,
        "difflet.cli.orchestrators.qwen_image": _patch_orch,
    }

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _TARGETS:
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(mod):
                orig_exec(mod)
                _TARGETS[name](mod)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
