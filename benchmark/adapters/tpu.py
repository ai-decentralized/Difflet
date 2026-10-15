"""TPU backend adapter for the difflet benchmark harness.

Drives the same code path production serving uses: one worker process per
chip, difflet's Qwen-Image stage adapter, difflet's sharded DiT. That matters
for comparability — a hand-rolled loop would measure something the server does
not run.

The TPU backend is *eager*, so ``compile_seconds`` is 0 in the harness sense
(no AOT artifact is built or reused). It is not free, though: XLA compiles the
graph on the first execution of each process, which lands inside
``load_seconds`` here and is called out in the report note. difflet's
``TpuApplicationBase`` can export a StableHLO artifact, but serving does not
depend on it, because torch_xla cannot persist compiled executables at all
("UNIMPLEMENTED: Deserializing serialized executable not supported"), so an
artifact would save tracing and not compilation.

Per-step latency follows the harness rule: inter-step deltas of a *real*
generate loop, device-synced, step 0 dropped.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from benchmark.harness import BackendAdapter

_REPLICA_TIMEOUT = 3600


def _env_number(name: str, kind):
    """``kind(os.environ[name])`` or None when unset/empty.

    The two TeaCache knobs (``DIFFLET_BENCH_TEACACHE_CADENCE``,
    ``DIFFLET_BENCH_TEACACHE_ONLINE_DELTA``) are A/B switches layered on the
    frozen MATRIX row, the same way ``DIFFLET_BENCH_SYNC_STEPS`` is; unset means
    the baseline the other device folders measured.
    """
    raw = os.environ.get(name, "").strip()
    return kind(raw) if raw else None


def _reported_steps(deltas, denoise_seconds, sync_steps):
    """Per-step latencies the report should quote.

    ``deltas`` comes from ``RealLoopStepTimer.deltas()`` and already excludes
    step 0.

    With ``sync_steps`` -- the default, and the cross-device method -- they are
    real device time and are used as-is. Without it they are enqueue times, so
    the loop is only resolvable in aggregate: every entry becomes the denoise
    wall clock divided by the step count. That distribution is deliberately
    degenerate -- identical mean/median/p90 -- because it is exactly as much as
    the method can resolve, and a plausible-looking spread would be fiction.
    Note it also absorbs step 0's share, including first-execution compile on a
    cold process, so read it on warm iterations only.
    """
    if sync_steps or not deltas:
        return list(deltas)
    per_step = denoise_seconds / (len(deltas) + 1)
    return [per_step] * len(deltas)


def _worker(rank, world, spec_payload, cmd_q, reply_q):
    """One rank: bring up its chip, load the stages, then serve commands."""
    from torch_xla._internal import pjrt

    pjrt.initialize_multiprocess(rank, world)

    import torch  # noqa: F401
    import torch_xla
    import torch_xla.core.xla_model as xm

    # Threads are shared across replicas; the default is one per core *per
    # process*, which oversubscribes the host N-fold and dominates the
    # host-side text encode.
    cores = os.cpu_count() or world
    torch.set_num_threads(max(1, cores // world))

    from types import SimpleNamespace

    from difflet.serving.orchestrators.qwen_image import QwenImageServingStageAdapter
    from difflet.serving.types import ParallelTopology

    spec = SimpleNamespace(**spec_payload)
    profile = SimpleNamespace(
        height=spec.height, width=spec.width, num_frames=None,
        parallel=ParallelTopology(tp_degree=world, cp_degree=1, world_size=world),
        world_size=world, teacache_speedup=None, teacache_calibration_data=None,
        # Probe-free TeaCache A/B knobs (off by default so the frozen MATRIX
        # row stays the baseline). The adapter reads them exactly as `difflet
        # serve --teacache-cadence/--teacache-online-delta` would.
        teacache_cadence=spec_payload.get("teacache_cadence"),
        teacache_online_delta=spec_payload.get("teacache_online_delta"),
        shape_dict=lambda: {"height": spec.height, "width": spec.width, "num_frames": None},
    )

    adapter = QwenImageServingStageAdapter()
    adapter._tpu = True
    adapter.model_dir = spec.model_dir
    adapter.active_profile = profile

    device = torch_xla.device()
    started = time.monotonic()
    adapter._load_text_stage(profile)
    adapter._load_denoiser_stage(profile)
    adapter._load_vae_stage(profile)
    load_seconds = time.monotonic() - started

    # Time each DiT call by wrapping the module, so per-step numbers come from
    # the real denoise loop rather than a separate synthetic one.
    #
    # DIFFLET_BENCH_SYNC_STEPS forces a device sync per step. The harness asks
    # for "device-synced" inter-step deltas, which is the right rule for eager
    # backends, but under XLA's lazy execution that extra sync breaks
    # pipelining and slows the loop it is measuring — it is a different
    # workload, not just a different clock. Default off; set it to compare.
    inner = adapter._tpu_module
    # Device-synced per step, DEFAULT ON, because that is the cross-device
    # metric. benchmark/step_realloop.py defines it: "inter-step deltas of a
    # real generate loop, cuda-synced, step 0 excluded", and trn2/trn3 were
    # re-measured to match the H100 that way. A TPU number produced by any
    # other rule is not comparable to the rows beside it, whatever else it
    # might be worth.
    #
    # This was previously OFF, on the reasoning that forcing a sync breaks
    # XLA's pipelining and so measures a different workload. That reasoning is
    # sound, but the conclusion drawn from it was not: the unsynced deltas are
    # not a different measure of device time, they are not device time at all.
    # Under lazy XLA the Python loop enqueues faster than the chips execute, so
    # an unsynced delta is the *enqueue rate* and the backlog is paid at the
    # loop's final sync -- measured on Qwen-Image, 274.9 ms enqueued against
    # 619.6 ms of real device time, which is how a step figure ended up in this
    # report that was smaller than the model's own attention.
    #
    # The overlap XLA loses to the sync is real, and larger than an eager
    # backend loses, so both numbers are recorded: `step_seconds` is the
    # comparable synced one, `throughput_step_seconds` is the denoise wall
    # clock divided by steps, and `enqueue_step_seconds` is the raw deltas.
    # DIFFLET_BENCH_SYNC_STEPS=0 restores the old behaviour.
    sync_steps = os.environ.get("DIFFLET_BENCH_SYNC_STEPS", "1") not in ("0", "false", "no")

    from benchmark.harness import RealLoopStepTimer

    # Mutable so a single worker can serve both bases without a reload: the
    # comparable synced pass and the natural pass a real serving loop runs.
    sync_now = {"on": sync_steps}

    def _tpu_sync():
        xm.mark_step()
        if sync_now["on"]:
            xm.wait_device_ops()

    timer = RealLoopStepTimer(sync=_tpu_sync)

    class _Timed:
        def __call__(self, *args, **kwargs):
            out = inner(*args, **kwargs)
            timer.step()
            return out

        def __getattr__(self, item):
            return getattr(inner, item)

    adapter._tpu_module = _Timed()

    def peak_gb():
        try:
            info = xm.get_memory_info(device)
            return info.get("bytes_used", 0) / 2**30
        except Exception:  # noqa: BLE001
            return None

    if rank == 0:
        reply_q.put({"type": "ready", "rank": rank, "load_seconds": load_seconds})
    else:
        reply_q.put({"type": "ready", "rank": rank})

    while True:
        cmd = cmd_q.get()
        if cmd["type"] == "shutdown":
            return
        sync_now["on"] = bool(cmd.get("sync_steps", sync_steps))
        timer.stamps.clear()
        request = SimpleNamespace(
            prompt=spec.prompt, num_inference_steps=spec.steps,
            seed=spec.seed, guidance_scale=spec.guidance_scale, request_id="bench",
            # _decode reads the shape off the request since f31e433 (a profile
            # may carry a whole shape set; the request says which one ran).
            height=spec.height, width=spec.width,
        )
        wall = time.monotonic()
        text = adapter._encode_prompt(request.prompt)
        encode_s = time.monotonic() - wall
        mark = time.monotonic()
        # _denoise ends by pulling the latents to the host, so this wall clock
        # already includes every queued step -- it is real, unlike the deltas.
        latents = adapter._denoise(text, request)
        denoise_s = time.monotonic() - mark
        mark = time.monotonic()
        png = adapter._decode(latents, request)
        decode_s = time.monotonic() - mark
        # A cheap visual check for changes that alter numerics: the metrics
        # above stay finite and in range even when an image is wrong.
        save_to = os.environ.get("DIFFLET_BENCH_SAVE_PNG")
        if save_to and rank == 0:
            Path(save_to).write_bytes(png)
        total = time.monotonic() - wall
        # With TeaCache on, a skipped step never calls the DiT, so the timer
        # sees only the FULL steps: `step_seconds` is then the per-full-step
        # latency and `denoise_seconds` carries the whole saving. The skip
        # counts ride along so a report can say which steps were real.
        steps = timer.deltas()
        teacache = getattr(adapter, "_tpu_teacache_last_stats", None)
        if rank == 0:
            reply_q.put({
                "type": "result", "wall_seconds": total, "load_seconds": load_seconds,
                "teacache": teacache,
                "encode_seconds": encode_s, "denoise_seconds": denoise_s,
                "decode_seconds": decode_s,
                # deltas() already drops step 0, so no further slicing here
                "step_seconds": _reported_steps(steps, denoise_s, sync_now["on"]),
                "enqueue_step_seconds": list(steps),
                "throughput_step_seconds": denoise_s / max(len(steps) + 1, 1),
                "step_basis": "synced" if sync_now["on"] else "natural",
                "peak_mem_gb": peak_gb(), "png_bytes": len(png),
                "latent_shape": list(latents.shape),
                "finite": bool(latents.isfinite().all()),
                "min": float(latents.min()), "max": float(latents.max()),
                # mean/std are part of harness.OutputInfo; the report renderer
                # prints them unconditionally alongside the range.
                "mean": float(latents.mean()), "std": float(latents.std()),
                "dtype": str(latents.dtype),
            })


class TpuAdapter(BackendAdapter):
    name = "tpu"

    def __init__(self, world: int | None = None) -> None:
        self._world = world
        self._procs: list = []
        self._cmd_qs: list = []
        self._reply_q = None
        self._load_seconds = 0.0

    # ------------------------------------------------------------- provenance

    def _tpu_env(self) -> dict[str, str]:
        try:
            import urllib.request

            request = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/"
                "attributes/tpu-env",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                body = response.read().decode()
            return dict(
                line.split(": ", 1) for line in body.splitlines() if ": " in line
            )
        except Exception:  # noqa: BLE001
            return {}

    def world_size(self) -> int:
        if self._world:
            return self._world
        chips = len(list(Path("/dev/vfio").glob("[0-9]*")))
        self._world = chips or 1
        return self._world

    def device_info(self) -> str:
        env = self._tpu_env()
        accelerator = env.get("ACCELERATOR_TYPE", "unknown").strip("'")
        topology = env.get("TOPOLOGY", "?").strip("'")
        return (
            f"Cloud TPU {accelerator} / {self.world_size()} chips / "
            f"topology {topology} / 16 GB HBM per chip"
        )

    def toolchain(self) -> dict[str, str]:
        from importlib import metadata

        versions = {}
        for package in ("torch", "torch-xla", "libtpu", "diffusers", "transformers"):
            try:
                versions[package] = metadata.version(package)
            except Exception:  # noqa: BLE001
                versions[package] = "absent"
        env = self._tpu_env()
        versions["accelerator_type"] = env.get("ACCELERATOR_TYPE", "?").strip("'")
        return versions

    # ------------------------------------------------------------- lifecycle

    _PATTERNS = ["transformer/*", "vae/*", "scheduler/*", "tokenizer/*",
                 "text_encoder/*", "model_index.json"]

    def _resolve_model_dir(self, spec, *, local_only: bool) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            spec.model_id,
            revision=getattr(spec, "revision", None),
            allow_patterns=self._PATTERNS,
            local_files_only=local_only,
        )

    def prepare(self, spec) -> None:
        self._model_dir = self._resolve_model_dir(spec, local_only=False)

    def compile(self, spec) -> tuple[float, dict[str, float]]:
        # Eager backend: no AOT artifact is produced or reused. XLA's own
        # compilation happens on first execution and is therefore counted in
        # load/e2e-cold, not here.
        return 0.0, {"eager": 0.0}

    def _ensure_started(self, spec) -> None:
        if self._procs:
            return
        world = self.world_size()
        # --skip-download never calls prepare(), so resolve the cached snapshot
        # here rather than passing a repo id where a path is expected.
        if not getattr(self, "_model_dir", None):
            self._model_dir = self._resolve_model_dir(spec, local_only=True)
        payload = {
            "model_dir": self._model_dir,
            "height": spec.height, "width": spec.width, "steps": spec.steps,
            "seed": getattr(spec, "seed", 42),
            "guidance_scale": getattr(spec, "guidance_scale", 1.0),
            "prompt": getattr(spec, "prompt", None) or "a red apple on a wooden table",
            "teacache_cadence": _env_number("DIFFLET_BENCH_TEACACHE_CADENCE", int),
            "teacache_online_delta": _env_number("DIFFLET_BENCH_TEACACHE_ONLINE_DELTA", float),
        }
        ctx = mp.get_context("spawn")
        self._cmd_qs = [ctx.Queue() for _ in range(world)]
        self._reply_q = ctx.Queue()
        self._procs = [
            ctx.Process(target=_worker,
                        args=(rank, world, payload, self._cmd_qs[rank], self._reply_q),
                        daemon=True)
            for rank in range(world)
        ]
        for process in self._procs:
            process.start()
        pending = set(range(world))
        while pending:
            reply = self._wait_reply()
            pending.discard(reply["rank"])
            if "load_seconds" in reply:
                self._load_seconds = reply["load_seconds"]

    def _wait_reply(self) -> dict:
        """Next worker reply, or RuntimeError as soon as a rank has died.

        A plain ``reply_q.get(timeout=_REPLICA_TIMEOUT)`` sat for the full hour
        after every rank had already crashed (the ``_decode`` TypeError of
        f9689ec left four tracebacks in the log and a parent that never
        returned). Poll instead, and check the ranks between polls.
        """
        import queue as _queue

        deadline = time.monotonic() + _REPLICA_TIMEOUT
        while True:
            try:
                return self._reply_q.get(timeout=5)
            except _queue.Empty:
                pass
            dead = [(rank, p.exitcode) for rank, p in enumerate(self._procs) if not p.is_alive()]
            if dead:
                self.shutdown()
                codes = ", ".join(f"rank{rank}={code}" for rank, code in dead)
                raise RuntimeError(f"TPU worker exited before replying ({codes}); see the log above")
            if time.monotonic() > deadline:
                self.shutdown()
                raise TimeoutError(f"no reply from TPU workers within {_REPLICA_TIMEOUT}s")

    #: This adapter can run the denoise loop without the per-step sync, so the
    #: harness can report the natural basis alongside the comparable one.
    supports_natural_mode = True

    def run_generate(self, spec, sync_steps: bool = True) -> dict:
        self._ensure_started(spec)
        for queue in self._cmd_qs:
            queue.put({"type": "generate", "sync_steps": sync_steps})
        reply = self._wait_reply()
        return {
            "wall_seconds": reply["wall_seconds"],
            "load_seconds": reply["load_seconds"],
            # RealLoopStepTimer.deltas() already excludes step 0.
            "step_seconds": reply["step_seconds"],
            "peak_mem_gb": reply.get("peak_mem_gb"),
            "output": {
                "shape": reply["latent_shape"], "dtype": reply["dtype"],
                "finite": reply["finite"], "min": reply["min"], "max": reply["max"],
                "mean": reply["mean"], "std": reply["std"],
                "note": f"packed latents; {reply['png_bytes']} byte PNG after decode",
            },
            "stage_seconds": {
                "text_encode": reply["encode_seconds"],
                "denoise": reply["denoise_seconds"],
                "vae_decode": reply["decode_seconds"],
            },
            # Kept alongside the comparable synced figure so the XLA-specific
            # tracing/execution overlap it gives up stays visible.
            "step_basis": reply.get("step_basis"),
            "throughput_step_seconds": reply.get("throughput_step_seconds"),
            "enqueue_step_seconds": reply.get("enqueue_step_seconds", []),
            # None unless DIFFLET_BENCH_TEACACHE_* was set: {full_steps,
            # skipped_steps, ...} from the controller, for the A/B report.
            "teacache": reply.get("teacache"),
        }

    def shutdown(self) -> None:
        for queue in self._cmd_qs:
            with_suppress = getattr(queue, "put", None)
            if with_suppress:
                try:
                    queue.put({"type": "shutdown"})
                except Exception:  # noqa: BLE001
                    pass
        for process in self._procs:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
        self._procs = []


__all__ = ["TpuAdapter"]
