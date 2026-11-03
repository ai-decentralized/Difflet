"""Shared helpers for cpu_ref_{qwen,ltx2,wan,hv}.py (CPU fp32 references, and since
2026-10-28 the CPU bf16 baseline via ``--dtype bfloat16``; the fp32 path is unchanged).

Not product code: measurement tooling for plan phase 4, kept in the bench dir.
"""

from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import threading
import time
from pathlib import Path

import torch

PROMPT = "a cinematic shot of a red fox running through a snowy forest"


def utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_sha(t: torch.Tensor) -> str:
    """sha256 of the tensor's raw bytes (dtype-exact, independent of torch.save framing)."""
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_rec(t: torch.Tensor) -> dict:
    return {"dtype": str(t.dtype), "shape": list(t.shape), "raw_sha256": raw_sha(t)}


class RssMonitor:
    """Samples memory every second in a daemon thread; keeps peaks and phase marks.

    VmRSS counts file-backed pages too (safetensors are mmapped, and those pages are
    reclaimable), so RssAnon (the part that cannot be reclaimed; there is no swap) and the
    host's lowest MemAvailable are recorded as well.
    """

    def __init__(self, period_s: float = 1.0) -> None:
        self.period_s = period_s
        self.peak_kib = 0
        self.peak_anon_kib = 0
        self.min_avail_kib: int | None = None
        self.marks: list[dict] = []
        self._t0 = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def status_kib() -> dict:
        out = {}
        with open("/proc/self/status") as f:
            for line in f:
                key = line.split(":", 1)[0]
                if key in ("VmRSS", "RssAnon", "RssFile"):
                    out[key] = int(line.split()[1])
        return out

    @staticmethod
    def mem_available_kib() -> int:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
        return 0

    @staticmethod
    def rss_kib() -> int:
        return RssMonitor.status_kib().get("VmRSS", 0)

    def _sample(self) -> dict:
        s = self.status_kib()
        avail = self.mem_available_kib()
        self.peak_kib = max(self.peak_kib, s.get("VmRSS", 0))
        self.peak_anon_kib = max(self.peak_anon_kib, s.get("RssAnon", 0))
        self.min_avail_kib = avail if self.min_avail_kib is None else min(self.min_avail_kib, avail)
        return {**s, "MemAvailable": avail}

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.period_s)

    def mark(self, name: str) -> None:
        s = self._sample()
        g = lambda kib: round(kib / 2**20, 2)  # noqa: E731
        rec = {
            "phase": name,
            "t_s": round(time.perf_counter() - self._t0, 2),
            "utc": utc(),
            "rss_gib": g(s.get("VmRSS", 0)),
            "rss_anon_gib": g(s.get("RssAnon", 0)),
            "rss_file_gib": g(s.get("RssFile", 0)),
            "mem_available_gib": g(s["MemAvailable"]),
            "peak_rss_gib_so_far": g(self.peak_kib),
            "peak_rss_anon_gib_so_far": g(self.peak_anon_kib),
        }
        self.marks.append(rec)
        print(f"[cpu-ref] {json.dumps(rec)}", flush=True)

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=5)
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
        return {
            "peak_rss_gib_sampled": round(self.peak_kib / 2**20, 2),
            "peak_rss_anon_gib_sampled": round(self.peak_anon_kib / 2**20, 2),
            "min_host_mem_available_gib": round((self.min_avail_kib or 0) / 2**20, 2),
            "ru_maxrss_gib": round(ru / 2**20, 2),
            "phases": self.marks,
        }


def environment(script: Path, args) -> dict:
    import diffusers
    import transformers

    cpu = ""
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    return {
        "argv": sys.argv,
        "args": {k: str(v) for k, v in vars(args).items()},
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_file": torch.__file__,
        "diffusers": diffusers.__version__,
        "diffusers_file": diffusers.__file__,
        "transformers": transformers.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS") if k in os.environ},
        "mkldnn_enabled": torch.backends.mkldnn.is_available(),
        "cpu": cpu,
        "nproc": os.cpu_count(),
        "host": platform.node(),
        "script": str(script),
        "script_sha256": file_sha(script),
        "common_sha256": file_sha(Path(__file__)),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "started_utc": utc(),
    }


def load_initial(path: Path, *, shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, dict]:
    """Loads a device dump of the latents entering the denoise loop; checks dtype and shape."""
    t = torch.load(path, map_location="cpu")
    if not torch.is_tensor(t):
        raise TypeError(f"{path}: expected a tensor, got {type(t).__name__}")
    if t.dtype != dtype or tuple(t.shape) != tuple(shape):
        raise ValueError(f"{path}: {t.dtype} {tuple(t.shape)}; expected {dtype} {tuple(shape)}")
    if not torch.isfinite(t.float()).all():
        raise ValueError(f"{path}: non-finite values")
    rec = {"path": str(path), "file_sha256": file_sha(path), **tensor_rec(t)}
    return t, rec


class FirstCallCapture:
    """Forward pre-hook: clones the named tensor kwargs of the transformer's FIRST call.

    This is what actually enters the denoise loop's first DiT forward, so comparing it with
    the injected device dump proves (or disproves) bit-exact injection.
    """

    def __init__(self, module: torch.nn.Module, names: tuple[str, ...]) -> None:
        self.names = names
        self.captured: dict[str, torch.Tensor] = {}
        self.calls = 0
        self.forward_s: list[float] = []
        self._t = None
        self._h1 = module.register_forward_pre_hook(self._pre, with_kwargs=True)
        self._h2 = module.register_forward_hook(self._post, with_kwargs=True)

    def _pre(self, module, args, kwargs):
        if self.calls == 0:
            for n in self.names:
                v = kwargs.get(n)
                if v is None:
                    raise RuntimeError(f"first transformer call has no kwarg {n!r}")
                self.captured[n] = v.detach().clone()
        self.calls += 1
        self._t = time.perf_counter()

    def _post(self, module, args, kwargs, output):
        self.forward_s.append(time.perf_counter() - self._t)

    def remove(self) -> None:
        self._h1.remove()
        self._h2.remove()


def injection_check(captured: torch.Tensor, injected_bf16: torch.Tensor) -> dict:
    """Bit-exact check of the first DiT input against the device dump (bf16 -> fp32 is exact)."""
    expect = injected_bf16.to(torch.float32)
    rec = {
        "captured": tensor_rec(captured),
        "expected_fp32_upcast": tensor_rec(expect),
        "bit_identical_to_fp32_upcast": bool(
            captured.dtype == torch.float32
            and tuple(captured.shape) == tuple(expect.shape)
            and raw_sha(captured) == raw_sha(expect)
        ),
    }
    # The reverse direction too: the captured fp32 tensor rounds back to the dump's exact bits.
    back = captured.to(injected_bf16.dtype)
    rec["roundtrip_to_dump_dtype_bit_identical"] = raw_sha(back) == raw_sha(injected_bf16)
    return rec


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def add_dtype_args(p) -> None:
    """--dtype (DiT + text encoders) and --vae-dtype (decoders); both default to float32.

    With the defaults every script runs exactly its fp32 code path (the as-used copies
    ``cpu_ref_*.as-used-fp32-refs-2026-10-27.py`` are the fp32 references' scripts).
    """
    p.add_argument("--dtype", choices=tuple(DTYPES), default="float32",
                   help="compute dtype of the DiT and the text encoders (bf16 baseline: bfloat16)")
    p.add_argument("--vae-dtype", choices=tuple(DTYPES), default="float32",
                   help="dtype of the VAE / decoders (default float32 also in the bf16 baseline)")


def param_dtypes(module: torch.nn.Module) -> dict:
    """Parameter count per dtype (diffusers keeps some modules fp32 in a bf16 load)."""
    out: dict = {}
    for x in module.parameters():
        out[str(x.dtype)] = out.get(str(x.dtype), 0) + x.numel()
    return out


def injection_check_cast(captured: torch.Tensor, injected: torch.Tensor,
                         loop_initial: torch.Tensor | None) -> dict:
    """Injection check for a reduced-precision run (``--dtype bfloat16``).

    * ``loop_initial`` (what ``prepare_latents`` returned, i.e. the latents the denoise loop
      starts from) must equal the dump cast to the loop dtype bit for bit, and that cast must
      be lossless (the loop starts from the device's exact noise);
    * the first DiT ``hidden_states`` must equal the dump cast to the DiT input dtype bit for
      bit. That cast is lossless for a bf16 dump; for an fp32 dump (Wan2.1, HunyuanVideo
      native) it is the bf16 rounding every bf16 DiT input goes through (recorded as such).
    """

    def lossless(src: torch.dtype, dst: torch.dtype) -> bool:
        return src == dst or (src == torch.bfloat16 and dst == torch.float32)

    exp_dit = injected.to(captured.dtype)
    rec = {
        "dump": tensor_rec(injected),
        "dit_input": tensor_rec(captured),
        "dit_input_equals_dump_cast": bool(tuple(captured.shape) == tuple(exp_dit.shape)
                                           and raw_sha(captured) == raw_sha(exp_dit)),
        "dit_input_cast_lossless": lossless(injected.dtype, captured.dtype),
    }
    ok = rec["dit_input_equals_dump_cast"]
    if loop_initial is not None:
        exp_loop = injected.to(loop_initial.dtype)
        rec["loop_initial"] = tensor_rec(loop_initial)
        rec["loop_initial_equals_dump_cast"] = bool(tuple(loop_initial.shape) == tuple(exp_loop.shape)
                                                    and raw_sha(loop_initial) == raw_sha(exp_loop))
        rec["loop_initial_cast_lossless"] = lossless(injected.dtype, loop_initial.dtype)
        ok = ok and rec["loop_initial_equals_dump_cast"] and rec["loop_initial_cast_lossless"]
    rec["ok"] = bool(ok)
    return rec


class PrepareSpy:
    """Wraps ``pipe.prepare_latents`` (instance attribute) and keeps a clone of its return."""

    def __init__(self, pipe) -> None:
        self.value: torch.Tensor | None = None
        orig = pipe.prepare_latents

        def spy(*args, **kw):
            out = orig(*args, **kw)
            if self.value is None:
                self.value = out.detach().clone()
            return out

        pipe.prepare_latents = spy


class StepTimer:
    """callback_on_step_end: wall time per step (end-to-end delta) and the last latents."""

    def __init__(self, num_steps: int) -> None:
        self.num_steps = num_steps
        self.step_s: list[float] = []
        self.final: torch.Tensor | None = None
        self._last: float | None = None
        self.on_first = None  # optional no-arg callable run after step 1 (bf16 runs: early injection check)

    def start(self) -> None:
        self._last = time.perf_counter()

    def __call__(self, pipe, step, timestep, callback_kwargs):
        now = time.perf_counter()
        self.step_s.append(now - self._last)
        self._last = now
        print(f"[cpu-ref] step {step + 1}/{self.num_steps} t={float(timestep):.3f} {self.step_s[-1]:.1f} s "
              f"{utc()}", flush=True)
        if step == 0 and self.on_first is not None:
            self.on_first()
        if step == self.num_steps - 1:
            self.final = callback_kwargs["latents"].detach().clone()
        return callback_kwargs


def early_injection(out_dir: Path, checks) -> None:
    """Runs ``checks()`` -> {name: injection_check_cast record}, writes injection_early.json, prints one line.

    Lets a supervisor confirm the injection right after step 1 instead of at the end of a
    multi-hour run (the final reference.json repeats the check).
    """
    recs = checks()
    ok = all(r["ok"] for r in recs.values())
    write_json(out_dir / "injection_early.json", {"utc": utc(), "ok": ok, **recs})
    print(f"[cpu-ref] injection-early ok={ok} " + " ".join(
        f"{k}: dit_input {r['dit_input']['dtype']} equals_dump_cast={r['dit_input_equals_dump_cast']} "
        f"lossless={r['dit_input_cast_lossless']} loop_initial "
        f"{r.get('loop_initial', {}).get('dtype')} equals={r.get('loop_initial_equals_dump_cast')}"
        for k, r in recs.items()), flush=True)


def free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def step_summary(step_s: list[float], forward_s: list[float], steps_total: int) -> dict:
    import statistics

    out = {"step_s": step_s, "forward_s": forward_s}
    if step_s:
        out["step_median_s"] = statistics.median(step_s)
        out["step_last_s"] = step_s[-1]
        # The per-step estimate uses the steady-state steps (step 0 excluded when there are more).
        steady = step_s[1:] if len(step_s) > 1 else step_s
        out["steady_step_s"] = statistics.median(steady)
        out["denoise_estimate_s_for_steps"] = {str(steps_total): out["steady_step_s"] * steps_total}
    return out
