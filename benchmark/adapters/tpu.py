"""TPU backend adapter for the difflet benchmark harness.

Drives the code path production serving uses -- one worker process per chip,
difflet's own application factories and orchestrators, the host encoders and
VAEs the serving adapters load -- through one per-model driver
(``benchmark/adapters/tpu_models.py``). That is what makes a row comparable:
a hand-rolled loop measures something the server does not run.

The protocol is the trn2 one (``benchmark/trn2/RESULTS.md``), applied without
adjustment:

* ``run_generate`` is a **fresh process**: the workers are shut down, spawned
  again, load the weights, serve exactly one request and stay resident. Its
  wall clock runs from spawn to the decoded output, the way the trn2
  adapter's runs from ``difflet generate``'s process start to its output
  file. The harness drops the OS page cache before the cold one.
* ``run_request`` is one more request on the resident workers -- the
  served-request cost, which trn2's CLI cannot measure (it reloads every
  process; its counterpart is warm e2e minus the warm load).
* per-step latency is ``RealLoopStepTimer``'s rule: inter-step deltas of a
  real generate, device-synced (``xm.wait_device_ops`` after each DiT call),
  step 0 excluded. The unsynced "natural" pass is reported beside it.

The backend is eager, so ``compile_seconds`` is 0 in the harness sense (no AOT
artifact is built or reused). It is not free: XLA compiles on each process's
first execution -- inside the load stages where a model's ``load_eager``
warms up, inside the first request otherwise -- and torch_xla cannot persist
the executables ("UNIMPLEMENTED: Deserializing serialized executable not
supported"), so every fresh process pays it again. That is the TPU analog of
trn2's one-time AOT compile and it is what separates e2e warm (fresh process)
from the resident request here.
"""

from __future__ import annotations

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
    frozen MATRIX row; unset means the baseline the other device folders
    measured.
    """
    raw = os.environ.get(name, "").strip()
    return kind(raw) if raw else None


def _reported_steps(deltas, denoise_seconds, sync_steps):
    """Per-step latencies the report should quote.

    ``deltas`` comes from ``RealLoopStepTimer.deltas()`` and already excludes
    step 0.

    With ``sync_steps`` -- the cross-device method -- they are real device time
    and are used as-is. Without it they are enqueue times, so the loop is only
    resolvable in aggregate: every entry becomes the denoise wall clock divided
    by the step count. That distribution is deliberately degenerate --
    identical mean/median/p90 -- because it is exactly as much as the method
    can resolve, and a plausible-looking spread would be fiction.
    """
    if sync_steps or not deltas:
        return list(deltas)
    per_step = denoise_seconds / (len(deltas) + 1)
    return [per_step] * len(deltas)


def _worker(rank, world, payload, cmd_q, reply_q):
    try:
        _worker_body(rank, world, payload, cmd_q, reply_q)
    except BaseException as exc:  # noqa: BLE001 - the parent must see every death
        import traceback

        reply_q.put({"type": "error", "rank": rank, "error": repr(exc),
                     "traceback": traceback.format_exc()})
        raise


def _worker_body(rank, world, payload, cmd_q, reply_q):
    """One rank: bring up its chip, load the model, then serve requests."""
    from torch_xla._internal import pjrt

    pjrt.initialize_multiprocess(rank, world)

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    # The serving adapters decode on the primary replica only; the drivers
    # read the same variable difflet/serving/models/* read.
    os.environ["DIFFLET_REPLICA_RANK"] = str(rank)
    # Threads are shared across replicas; the default is one per core *per
    # process*, which oversubscribes the host N-fold and dominates the
    # host-side text encode.
    torch.set_num_threads(max(1, (os.cpu_count() or world) // world))

    from benchmark.adapters.tpu_models import frames_uint8, make_driver, output_info, save_media
    from benchmark.harness import RealLoopStepTimer

    device = torch_xla.device()

    # Mutable so one worker serves both bases without a reload: the
    # comparable synced pass and the natural pass a serving loop runs. Under
    # lazy XLA an unsynced delta is the *enqueue* rate, not device time
    # (measured on Qwen-Image: 275 ms enqueued against 620 ms real), so the
    # synced figure is the one the cross-device table quotes.
    #
    # The sync is ``wait_device_ops`` alone, as RealLoopStepTimer documents:
    # every loop here already ``mark_step``s once per step, so the wait
    # measures the step the device is executing without cutting the graph.
    # Adding a ``mark_step`` inside the timer splits the per-step graph at the
    # DiT output, which is a different executable from the one the natural
    # pass (and serving) runs -- measured as a 24 s recompile on FLUX's first
    # natural request after the synced ones.
    sync_now = {"on": True}

    def _sync():
        if sync_now["on"]:
            xm.wait_device_ops()

    timer = RealLoopStepTimer(sync=_sync)
    driver = make_driver(payload, rank, world, timer)

    started = time.monotonic()
    stages = driver.load()
    load_seconds = time.monotonic() - started

    def mem_gb():
        try:
            info = xm.get_memory_info(device)
            used = info.get("peak_bytes_used") or info.get("bytes_used") or 0
            return used / 2**30
        except Exception:  # noqa: BLE001
            return None

    reply_q.put({
        "type": "ready", "rank": rank, "load_seconds": load_seconds,
        "load_stages": {name: round(seconds, 3) for name, seconds in stages.items()},
        "mem_after_load_gb": mem_gb(),
    })

    while True:
        cmd = cmd_q.get()
        if cmd["type"] == "shutdown":
            return
        sync_now["on"] = bool(cmd.get("sync_steps", True))
        timer.stamps.clear()
        prompt = payload["prompt"]
        wall = time.monotonic()
        text = driver.encode(prompt)
        encode_s = time.monotonic() - wall
        mark = time.monotonic()
        # Every driver's denoise ends with the latents on the host, so this
        # wall clock already includes every queued step -- it is real, unlike
        # the unsynced deltas.
        latents = driver.denoise(text)
        denoise_s = time.monotonic() - mark
        decode_s = 0.0
        decoded = None
        if driver.primary:
            mark = time.monotonic()
            decoded = driver.decode(latents)
            decode_s = time.monotonic() - mark
        total = time.monotonic() - wall
        if rank != 0:
            continue
        # With TeaCache on, a skipped step never calls the DiT, so the timer
        # sees only the FULL steps; the skip counts ride along.
        steps = timer.deltas()
        output = None
        saved: list[str] = []
        if decoded is not None:
            raw, layout, value_range = decoded
            output = output_info(raw, note=f"decoded {driver.output_kind} ({layout}, {value_range})")
            if cmd.get("save_stem"):
                frames = frames_uint8(raw, layout, value_range)
                saved = save_media(frames, cmd["save_stem"], kind=driver.output_kind, fps=driver.fps)
                output["note"] += "; saved " + ", ".join(Path(p).name for p in saved)
        reply_q.put({
            "type": "result", "wall_seconds": total, "load_seconds": load_seconds,
            "encode_seconds": encode_s, "denoise_seconds": denoise_s, "decode_seconds": decode_s,
            "step_seconds": _reported_steps(steps, denoise_s, sync_now["on"]),
            "enqueue_step_seconds": list(steps),
            "throughput_step_seconds": denoise_s / max(len(steps) + 1, 1),
            "step_basis": "synced" if sync_now["on"] else "natural",
            "peak_mem_gb": mem_gb(),
            "output": output,
            "latents": output_info(latents, note="denoised latents before the decode"),
            "teacache": driver.teacache_stats(),
            "saved": saved,
        })


class TpuAdapter(BackendAdapter):
    name = "tpu"
    supports_natural_mode = True
    supports_resident_mode = True

    def __init__(self, world: int | None = None, save_dir: str | None = None) -> None:
        self._world = world
        self._procs: list = []
        self._cmd_qs: list = []
        self._reply_q = None
        self._load_seconds = 0.0
        self._load_stages: dict[str, float] = {}
        self._mem_after_load_gb = None
        self._tag = "run"
        self.save_dir = save_dir or os.environ.get("DIFFLET_BENCH_SAVE_DIR") or None

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
        for package in ("torch", "torch-xla", "libtpu", "jax", "diffusers", "transformers"):
            try:
                versions[package] = metadata.version(package)
            except Exception:  # noqa: BLE001
                versions[package] = "absent"
        env = self._tpu_env()
        versions["accelerator_type"] = env.get("ACCELERATOR_TYPE", "?").strip("'")
        return versions

    # ------------------------------------------------------------- lifecycle

    def _resolve_model_dir(self, spec, *, local_only: bool) -> str:
        """The pinned snapshot directory.

        Every MATRIX row pins a revision, so with ``local_only`` the directory
        is addressed straight in the hub cache: ``snapshot_download(...,
        local_files_only=True)`` refuses a snapshot that was fetched with
        ``allow_patterns`` (it reports ``.gitattributes`` missing), and every
        checkpoint on this host was.
        """
        from huggingface_hub import constants, snapshot_download

        revision = getattr(spec, "revision", None)
        if local_only and revision:
            repo_dir = Path(constants.HF_HUB_CACHE) / f"models--{spec.model_id.replace('/', '--')}"
            snapshot = repo_dir / "snapshots" / revision
            if (snapshot / "model_index.json").exists():
                return str(snapshot)
        return snapshot_download(spec.model_id, revision=revision, local_files_only=local_only)

    def prepare(self, spec) -> None:
        self._model_dir = self._resolve_model_dir(spec, local_only=False)

    def compile(self, spec) -> tuple[float, dict[str, float]]:
        # Eager backend: no AOT artifact is produced or reused. XLA's own
        # compilation happens on first execution and is therefore counted in
        # load / e2e, not here.
        return 0.0, {"eager": 0.0}

    def tag(self, name: str) -> None:
        self._tag = name

    def _payload(self, spec) -> dict:
        if not getattr(self, "_model_dir", None):
            # --skip-download never calls prepare(), so resolve the cached
            # snapshot here rather than passing a repo id where a path is expected.
            self._model_dir = self._resolve_model_dir(spec, local_only=True)
        return {
            "model_type": spec.model_type,
            "model_dir": self._model_dir,
            "height": spec.height, "width": spec.width,
            "num_frames": getattr(spec, "num_frames", None),
            "steps": spec.steps,
            "seed": getattr(spec, "seed", 42),
            "guidance_scale": getattr(spec, "guidance_scale", None) or 1.0,
            "prompt": getattr(spec, "prompt", None) or "a red apple on a wooden table",
            "teacache_cadence": _env_number("DIFFLET_BENCH_TEACACHE_CADENCE", int),
            "teacache_online_delta": _env_number("DIFFLET_BENCH_TEACACHE_ONLINE_DELTA", float),
        }

    def _start(self, spec) -> None:
        world = self.world_size()
        payload = self._payload(spec)
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
            if reply["rank"] == 0:
                self._load_seconds = reply["load_seconds"]
                self._load_stages = dict(reply.get("load_stages") or {})
                self._mem_after_load_gb = reply.get("mem_after_load_gb")

    def _ensure_started(self, spec) -> None:
        if not self._procs:
            self._start(spec)

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
                reply = self._reply_q.get(timeout=5)
            except _queue.Empty:
                reply = None
            if reply is not None:
                if reply.get("type") == "error":
                    self.shutdown()
                    raise RuntimeError(
                        f"TPU worker rank{reply['rank']} failed: {reply['error']}\n{reply['traceback']}"
                    )
                return reply
            dead = [(rank, p.exitcode) for rank, p in enumerate(self._procs) if not p.is_alive()]
            if dead:
                self.shutdown()
                codes = ", ".join(f"rank{rank}={code}" for rank, code in dead)
                raise RuntimeError(f"TPU worker exited before replying ({codes}); see the log above")
            if time.monotonic() > deadline:
                self.shutdown()
                raise TimeoutError(f"no reply from TPU workers within {_REPLICA_TIMEOUT}s")

    # --------------------------------------------------------------- measure

    def _request(self, spec, sync_steps: bool) -> dict:
        save_stem = None
        if self.save_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            save_stem = str(Path(self.save_dir) / self._tag)
        for queue in self._cmd_qs:
            queue.put({"type": "generate", "sync_steps": sync_steps, "save_stem": save_stem})
        reply = self._wait_reply()
        return {
            "wall_seconds": reply["wall_seconds"],
            "load_seconds": self._load_seconds,
            # RealLoopStepTimer.deltas() already excludes step 0.
            "step_seconds": reply["step_seconds"],
            "peak_mem_gb": reply.get("peak_mem_gb"),
            "output": reply.get("output"),
            "latents": reply.get("latents"),
            "saved": reply.get("saved") or [],
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
            # None unless DIFFLET_BENCH_TEACACHE_* was set.
            "teacache": reply.get("teacache"),
        }

    def run_generate(self, spec) -> dict:
        """A fresh process: spawn the workers, load, one request; workers stay
        resident for ``run_request``. Wall clock = spawn -> decoded output."""
        self.shutdown()
        started = time.perf_counter()
        self._start(spec)
        request = self._request(spec, sync_steps=True)
        wall = time.perf_counter() - started
        load = sum(self._load_stages.values()) or self._load_seconds
        stages = [{"stage": name, "shard_s": None, "load_s": seconds}
                  for name, seconds in self._load_stages.items()]
        request["request_seconds"] = request["wall_seconds"]
        request["wall_seconds"] = wall
        request["load_seconds"] = load
        request["e2e_breakdown"] = {
            "stages": stages,
            "weights_shard_total_s": 0.0,
            "weights_load_total_s": round(load, 3),
            "wall_total_s": round(wall, 3),
            "compute_and_overhead_s": round(wall - load, 3),
            "note": (
                "fresh worker processes, one per chip. The residual is process start + "
                "imports + one request (text encode + denoise + decode on the primary "
                "replica). XLA's first-execution compile is inside the load stages where "
                "the model's load_eager warms up (HunyuanVideo, LTX-2, FLUX) and inside "
                "the request otherwise (Qwen-Image, Wan)."
            ),
        }
        return request

    def run_request(self, spec, sync_steps: bool = True) -> dict:
        self._ensure_started(spec)
        return self._request(spec, sync_steps=sync_steps)

    def shutdown(self) -> None:
        for queue in self._cmd_qs:
            try:
                queue.put({"type": "shutdown"})
            except Exception:  # noqa: BLE001
                pass
        for process in self._procs:
            process.join(timeout=60)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        self._procs = []
        self._cmd_qs = []


__all__ = ["TpuAdapter"]
