"""Manual check (W4): Wan2.1 T2V end to end on the neuron backend, TP4 on 4 ranks.

The first full video path: umT5 in fp32 on rank 0's host (loaded in its own status-synced
phase, "text encoder load") with the embeddings broadcast to every rank in bf16, the DiT on
each rank's core (compile mode: the 40 blocks compiled in place into one shared graph, the
warm-up compiled or loaded before the step loop), ``WanOrchestrator`` (fp32 latents, UniPC on
the host, classifier-free guidance as two batch-1 forwards per step, the empty negative prompt
encoded lazily through the same rank-0 encoder) on every rank, and the VAE decode on rank 0's
host (``difflet.cli.orchestrators.wan._decode_latents_host``, the ``--host-vae`` path) writing
the mp4. Every rank builds ``TorchNeuronWanApplication`` directly, loads it with
``skip_warmup=False`` and calls it as the CLI's transformer stage does
(``app(generator=Generator().manual_seed(seed), prompt=..., height, width, num_frames,
num_inference_steps, guidance_scale, output_type="latent")``), so the init noise comes from a
CPU generator (no device generator).

Two uses:

* ``--ref <ref_2steps.pt>`` (``tests/numerical/wan_pipeline_reference.py``: diffusers
  ``WanPipeline`` on CPU, fp32 reference + bf16 control + a bf16 control with bf16-rounded
  timesteps, same seed, prompt, steps and guidance): every rank compares its own results.
* ``--mp4 <path>`` (mandatory without ``--ref``): rank 0 decodes the final latents and writes
  the video after every rank has passed its checks and the process group is gone.

Launch (one device job at a time; cores free; absolute paths):

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=3 DIFFLET_NO_JEMALLOC=1 DIFFLET_DISABLE_PREWARM=1 \\
    DIFFLET_NEURON_PG_TIMEOUT_S=3600 PYTHONSAFEPATH=1 PYTHONPATH=<worktree> timeout 3600 \\
    <venv>/bin/torchrun --standalone --nproc-per-node 4 \\
        <worktree>/tests/manual/check_neuron_wan_w4.py --model-dir <snapshot> \\
        --ref <records>/ref_2steps.pt --steps 2 --guidance-scale 5.0 --seed 42 \\
        --cache-dir <records>/compile_cache --out-dir <records>/run-2steps

then, in the driver (no torchrun, no device), ``--summarize <out-dir> --tag 2steps`` asserts
that the four ranks hold identical latents (init, after every step, final; also the prompt and
negative embeddings), reads the mp4 back (frame count, fps, codec, size) and writes
``<records>/result-<tag>.json`` with the verdict, the latent hashes, the parity table, the
stage clocks and the memory figures. Exit 0 iff everything holds.

What each rank records (``<out-dir>/result-rank<r>.json``), and the checks (rank 0 prints one
PASS/FAIL line per check, each rank ``[rank r] PASS``):

* the measurement hygiene of the W2/W3 gates: ``torch_neuronx.metrics`` enabled and the NEFF
  cache listed before ``prepare_runtime`` on a fresh runtime (instrumentation), the compilation
  counters around every DiT forward (no_compile_in_forwards: ``TotalCompilations +
  PersistentHits == 0`` summed over the forwards, no new Dynamo graph), ``track_fallbacks()``
  around the load and around the generate (fallbacks), ``unwarmed_shapes == []`` and every
  forward's device inputs at the warm-up's shapes (forward_shapes), compiled_blocks /
  one_graph in compile mode, hbm_d48 (``max_memory_allocated`` <= 18 GiB);
* steps: one ``step_hook`` stamp per denoise iteration and ``steps x (2 at guidance > 1)``
  forwards; text_encoder_load: the umT5 load ran in its own phase;
* negative_embeds: the lazily encoded negative embeddings are not all zeros (the encoder ran,
  not the orchestrator's zeros fallback);
* rank 0: vmhwm_budget, its ``VmHWM`` after the generate at most 40 GB (umT5 fp32 + loader;
  Review Focus 2);
* with ``--ref``: init_latents (the generator's noise equals the reference's bit for bit),
  prompt_embeds / negative_prompt_embeds (``rel_max <= 1e-2`` against diffusers' fp32 umT5,
  rows past the valid tokens zero in both), parity_latents_final (``parity.check`` of the final
  latents against the fp32 reference with the bf16 control, same rule as the W2/W3 gates,
  D68), update_final (the update ``final - init``: cosine >= 0.98, the catastrophe floor, and
  the same ratio rule); reported, not gated: the same comparison after every step, the
  transformer outputs per forward, and every comparison against the control with
  bf16-rounded timesteps (decision D44: the orchestrator feeds a bf16 DiT 1000 for 999);
* with ``--expect warm``: the load compiled no NEFF (``TotalCompilations == 0``, at least one
  persistent hit); the counters of every phase and the new ``.neff`` files are recorded.

umT5 threads: rank 0 encodes with a fixed ``--encoder-threads`` (default 12, this host's
vCPUs). It is pinned, not picked per launch, because umT5's fp32 output bits depend on the
intra-op thread count: the embeddings, and with them every latent hash below, are reproducible
across launches only at the same count. ``--probe-encoder-threads 12,3,12,3`` (off by default)
times rank 0's encode at each listed count before the generate (errata P1-7), as a record
only; the generate still runs at ``--encoder-threads``. W4 measured 12 vs 3 as a tie within
noise (2.17 / 2.28 s, then 2.88 / 2.17 s): the encode is memory-bound.

Timing (informational; W6 owns the benchmark): ``phase_seconds`` (incl. "text encoder load"
and "warmup forward"), the optional thread probe, every encode inside the generate, per-iteration
seconds from ``step_hook`` (iteration 0 from the start of its first forward), per-forward
seconds from ``forward_hook``, the generate wall, the VAE decode (rank 0, ``--decode-threads``)
and the process total. Every rank hashes the latents after each step (``--no-hash-steps`` to
skip) and at the end.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ["DIFFLET_BACKEND"] = "neuron"

import torch

from tests.numerical import neuron_parity as parity

MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
#: benchmark/models.py BenchConfig.prompt (the trn2 benchmark prompt).
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
#: Review Focus 2: rank 0 holds umT5 fp32 (22.7 GB) next to its DiT shards' loader.
VMHWM_BUDGET_GB = 40.0
#: Decision D48: above this per-rank peak the 14B runs fall back to 480x832x5.
HBM_LIMIT_GIB = 18.0
#: Both embeddings come from fp32 umT5 (the device's arrive in bf16).
EMBEDS_MAX_REL_MAX = 1e-2
#: The update final - init must point the reference's way (brief W4 step 2(c)).
UPDATE_MIN_COSINE = 0.98
#: A decoded clip whose frames are this flat (std in [-1, 1] units) is not a video.
MIN_FRAME_STD = 0.05
GIB = 2**30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summarize", type=Path, default=None,
                   help="driver mode: read <dir>/result-rank*.json, assert, write the result")
    p.add_argument("--tag", default=None, help="driver mode: result-<tag>.json")
    p.add_argument("--result-dir", type=Path, default=None,
                   help="driver mode: where result-<tag>.json goes (default: the parent of "
                        "--summarize)")
    p.add_argument("--launch-times", type=Path, default=None,
                   help="driver mode: a file with the launch's start and end epoch seconds")
    p.add_argument("--model-dir", type=Path, help="the diffusers snapshot")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--text-seq-len", type=int, default=512)
    p.add_argument("--exec-mode", choices=("eager", "compile"), default="compile")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="NEFF cache directory (else the Difflet compile cache's)")
    p.add_argument("--expect", choices=("warm",), default=None)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--ref", type=Path, default=None,
                   help="tests/numerical/wan_pipeline_reference.py output to compare with")
    p.add_argument("--mp4", type=Path, default=None,
                   help="rank 0 decodes on the host and writes this video (required "
                        "without --ref)")
    p.add_argument("--no-hash-steps", dest="hash_steps", action="store_false",
                   help="do not hash the latents after every step (the final ones still are)")
    p.add_argument("--probe-encoder-threads", default="",
                   help="record only: rank 0's umT5 intra-op thread counts to time before the "
                        "generate, in order (e.g. 12,3,12,3); off by default")
    p.add_argument("--encoder-threads", type=int, default=12,
                   help="rank 0's umT5 intra-op threads for the real encodes; pinned because "
                        "the embeddings' bits depend on it")
    p.add_argument("--decode-threads", type=int, default=os.cpu_count() or 1,
                   help="rank 0's torch threads for the VAE decode (the peers have exited)")
    args = p.parse_args(argv)
    if args.summarize is not None:
        if args.tag is None:
            p.error("--summarize needs --tag")
        return args
    if args.model_dir is None or args.out_dir is None:
        p.error("--model-dir and --out-dir are required (or --summarize)")
    if args.ref is None and args.mp4 is None:
        p.error("--mp4 is required without --ref: the goal run writes the video")
    if args.expect is not None and args.cache_dir is None:
        p.error("--expect needs --cache-dir (the directory whose warm state is checked)")
    try:
        args.probe_encoder_threads = [int(t) for t in args.probe_encoder_threads.split(",")
                                      if t.strip()]
    except ValueError:
        p.error(f"--probe-encoder-threads must be comma-separated integers, got "
                f"{args.probe_encoder_threads!r}")
    if args.encoder_threads < 1:
        p.error(f"--encoder-threads must be at least 1, got {args.encoder_threads}")
    return args


# ------------------------------------------------------------------------- measurements


def sha256(tensor: torch.Tensor) -> str:
    """Of the raw bytes; bf16 goes through an int16 view (numpy has no bf16)."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
    return {"n": len(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "p90": p90, "min": ordered[0],
            "max": ordered[-1]}


def host_memory() -> dict[str, float]:
    """This process's VmHWM / VmRSS and the host's MemAvailable / Cached, in GiB."""
    values: dict[str, float] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            key, value = line.split(":", 1)
            values[f"{key}_gib"] = int(value.split()[0]) / 2**20  # kB
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "Cached:")):
            key, value = line.split(":", 1)
            values[f"host_{key}_gib"] = int(value.split()[0]) / 2**20
    return values


def compilation_counters() -> dict[str, int]:
    import torch_neuronx.metrics as metrics

    return {name: metrics.get_counter_value("CompilationCache." + name) or 0
            for name in parity.COUNTER_NAMES}


def neff_files() -> dict[str, int]:
    from difflet.backends.neuron.compile import NEFF_CACHE_ENV

    raw = os.environ.get(NEFF_CACHE_ENV)
    directory = Path(raw) if raw else None
    if directory is None or not directory.is_dir():
        return {}
    files = {}
    for path in directory.rglob("*.neff"):
        try:
            files[str(path.relative_to(directory))] = path.stat().st_size
        except OSError:
            continue
    return files


def dynamo_stats() -> dict[str, int]:
    from torch._dynamo.utils import counters

    return {"unique_graphs": int(counters["stats"]["unique_graphs"]),
            "graph_breaks": int(sum(counters["graph_break"].values()))}


def device_memory_gib() -> dict[str, float]:
    import torch_neuronx

    return {"allocated_gib": torch_neuronx.memory_allocated() / GIB,
            "max_allocated_gib": torch_neuronx.max_memory_allocated() / GIB}


def snapshot(t0: float, *, device: bool = True) -> dict[str, Any]:
    return {"t_s": time.perf_counter() - t0, "counters": compilation_counters(),
            "dynamo": dynamo_stats(), "neff_count": len(neff_files()),
            "device_memory": device_memory_gib() if device else None,
            "host_memory": host_memory()}


def delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    return {k: after["counters"][k] - before["counters"][k] for k in after["counters"]}


def comparison(actual: torch.Tensor, runs: dict[str, Any], select) -> dict[str, Any]:
    """``parity.check`` of ``actual`` against the fp32 reference with the bf16 control (the
    W2/W3 rule, D68), plus the same against the bf16 control with bf16-rounded timesteps when
    the reference has it (diagnostic, D44). ``select(run)`` picks the tensor out of a run."""
    ref, ctl = select(runs["fp32"]), select(runs["bf16"])
    m = parity.metrics(actual, ref, control=ctl)
    m["vs_bf16"] = parity.metrics(actual, ctl)
    control = parity.metrics(ctl, ref)
    notes: list[str] = []
    ok, reasons = parity.check(m, control=control, notes=notes)
    entry: dict[str, Any] = {
        "vs_fp32": m, "control": control, "ok": ok, "reasons": reasons, "notes": notes,
        "ratio_vs_control": m["rel_l1"] / control["rel_l1"],
        "same_dtype_gated": parity.same_dtype_gated(m),
        "same_dtype": parity.same_dtype_label(m, control),
    }
    if "bf16_tsround" in runs:
        rounded = select(runs["bf16_tsround"])
        mr = parity.metrics(actual, ref, control=rounded)
        mr["vs_bf16"] = parity.metrics(actual, rounded)
        control_r = parity.metrics(rounded, ref)
        ok_r, reasons_r = parity.check(mr, control=control_r)
        entry["tsround"] = {
            "vs_fp32": mr, "control": control_r, "ok": ok_r, "reasons": reasons_r,
            "ratio_vs_control": mr["rel_l1"] / control_r["rel_l1"],
            "same_dtype": parity.same_dtype_label(mr, control_r),
            "control_vs_bf16_control": parity.metrics(rounded, ctl),
        }
    return entry


def describe(entry: dict[str, Any]) -> str:
    m, control = entry["vs_fp32"], entry["control"]
    text = (f"cos={m['cosine']:.6f} rel_l1={m['rel_l1']:.3e} ({entry['ratio_vs_control']:.3f}x "
            f"control {control['rel_l1']:.3e}, cos {control['cosine']:.6f}) "
            f"{entry['same_dtype']}")
    if "tsround" in entry:
        t = entry["tsround"]
        text += (f"; vs the bf16-timestep control {t['control']['rel_l1']:.3e}: "
                 f"{t['ratio_vs_control']:.3f}x ({t['same_dtype']})")
    if entry["reasons"]:
        text += f" {entry['reasons']}"
    if entry["notes"]:
        text += f" {entry['notes']}"
    return text


def embeds_check(actual: torch.Tensor, reference: torch.Tensor, valid: int) -> tuple[bool, str]:
    m = parity.metrics(actual.float(), reference.float())
    pad_actual = float(actual[:, valid:].abs().max()) if actual.shape[1] > valid else 0.0
    pad_ref = float(reference[:, valid:].abs().max()) if reference.shape[1] > valid else 0.0
    ok = m["rel_max"] <= EMBEDS_MAX_REL_MAX and pad_actual == 0.0 and pad_ref == 0.0
    return ok, (f"rel_max={m['rel_max']:.3e} (max {EMBEDS_MAX_REL_MAX}) cos={m['cosine']:.8f} "
                f"rel_l1={m['rel_l1']:.3e}; {valid} valid tokens, padding absmax "
                f"{pad_actual} / {pad_ref} (device / reference)")


# ---------------------------------------------------------------------------- the run


def main(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    started_epoch = time.time()
    ref = None
    if args.ref is not None:
        # The reference first (host only, seconds): a wrong --ref must not cost the 14B load.
        ref = torch.load(args.ref, weights_only=False)
        meta = ref["meta"]
        expected = {"prompt": args.prompt, "steps": args.steps,
                    "guidance_scale": args.guidance_scale, "seed": args.seed,
                    "height": args.height, "width": args.width,
                    "num_frames": args.num_frames, "max_sequence_length": args.text_seq_len,
                    "negative_prompt": ""}
        wrong = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
        if wrong:
            raise ValueError(f"{args.ref} was made for other settings (reference, this run): "
                             f"{wrong}")
        if "bf16" not in ref["runs"]:
            raise ValueError(f"{args.ref} has no bf16 control run")

    from difflet.backends.neuron import runtime
    from difflet.backends.neuron.compile import configure_compile_cache
    from difflet.backends.neuron.runtime import track_fallbacks
    from difflet.backends.registry import get_backend
    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    if args.cache_dir is not None:
        cache_dir = args.cache_dir.expanduser().absolute()
        configure_compile_cache(cache_dir,
                                local_cache_dir=cache_dir.with_name(cache_dir.name + "_local"))
    else:
        configure_compile_cache()

    # Before prepare_runtime (which compiles or loads a NEFF itself): the counters on, the
    # NEFF cache listed; neither starts the runtime.
    import torch_neuronx
    import torch_neuronx.metrics as metrics

    metrics.set_enabled(True)
    metrics_setup = {"enabled_before_prepare_runtime": bool(metrics.is_enabled()),
                     "runtime_initialized_when_enabled":
                         bool(torch_neuronx.is_neuron_runtime_initialized())}
    neffs_before = neff_files()
    snaps = {"pre_prepare": snapshot(t0, device=False)}
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parallel = DiffletParallelConfig(tp_degree=world)
    get_backend("neuron").prepare_runtime(parallel)
    prepare_s = time.perf_counter() - t0

    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snaps["start"] = snapshot(t0)

    app = TorchNeuronWanApplication(
        model_path=args.model_dir, parallel=parallel, dtype=torch.bfloat16,
        shape={"height": args.height, "width": args.width, "num_frames": args.num_frames},
        text_seq_len=args.text_seq_len, exec_mode=args.exec_mode, device="neuron",
        encoder_threads=args.encoder_threads,
    )
    start = time.perf_counter()
    with track_fallbacks() as load_fallbacks:
        app.load(skip_warmup=False)  # DiT load + warm-up, then umT5 on rank 0
    load_s = time.perf_counter() - start
    load_fallbacks = list(load_fallbacks)
    snaps["after_load"] = snapshot(t0)
    print(f"[rank {rank}] loaded in {load_s:.1f}s: phases="
          f"{ {k: round(v, 1) for k, v in app.phase_seconds.items()} } HBM max="
          f"{snaps['after_load']['device_memory']['max_allocated_gib']:.2f} GiB VmHWM="
          f"{snaps['after_load']['host_memory']['VmHWM_gib']:.2f} GiB counters="
          f"{delta(snaps['after_load'], snaps['start'])}", flush=True)

    # ------------------------------------------------------------- instrumentation
    pipeline, adapter, encoder = app.pipeline, app.transformer_adapter, app.text_encoder
    phase = {"name": "probe"}
    encodes: list[dict[str, Any]] = []  # rank 0: every umT5 encode
    original_encode = encoder._encode

    def timed_encode(input_ids, attention_mask):
        begin = time.perf_counter()
        embeds = original_encode(input_ids, attention_mask)
        encodes.append({"phase": phase["name"], "threads": encoder.threads,
                        "tokens": int(attention_mask.sum()),
                        "seconds": time.perf_counter() - begin,
                        "vmhwm_gib": host_memory()["VmHWM_gib"]})
        return embeds

    encoder._encode = timed_encode

    # P1-7, record only (--probe-encoder-threads): rank 0's encode at each listed thread count
    # while the peers wait in the status all-reduce. The generate runs at the pinned
    # --encoder-threads whatever the probe finds: the embeddings' bits depend on the count.
    probe: list[dict[str, Any]] = []
    probe_fastest = None
    if args.probe_encoder_threads:
        tokenizer = pipeline._get_tokenizer()
        tokens = tokenizer([args.prompt], padding="max_length", truncation=True,
                           max_length=args.text_seq_len, return_tensors="pt")
        for threads in args.probe_encoder_threads:
            encoder.threads = threads
            begin = time.perf_counter()
            embeds = encoder(tokens["input_ids"].to(torch.int64),
                             tokens["attention_mask"].to(torch.int32))
            probe.append({"threads": threads, "call_s": time.perf_counter() - begin,
                          "sha256": sha256(embeds),
                          "encode_s": encodes[-1]["seconds"] if rank == 0 else None})
        if rank == 0:
            best: dict[int, float] = {}
            for entry in probe:
                best[entry["threads"]] = min(best.get(entry["threads"], math.inf),
                                             entry["encode_s"])
            probe_fastest = min(best, key=best.get)
    encoder.threads = args.encoder_threads
    snaps["after_probe"] = snapshot(t0)
    if probe:
        print(f"[rank {rank}] encoder thread probe (record only): "
              f"{[(e['threads'], round(e['call_s'], 3), e['encode_s'] and round(e['encode_s'], 3)) for e in probe]}"
              f"; fastest {probe_fastest}; the generate uses the pinned {args.encoder_threads}",
              flush=True)

    phase["name"] = "generate"
    step_stamps: list[float] = []
    app.step_hook = lambda: step_stamps.append(time.perf_counter())
    forward_s: list[float] = []
    adapter.forward_hook = forward_s.append

    dit = adapter.module
    forward_counters: list[dict[str, int]] = []
    forward_shapes: list[list[list[int]]] = []

    def counted_forward(*device_args):
        forward_shapes.append([list(a.shape) for a in device_args])
        before = compilation_counters()
        out = dit(*device_args)
        after = compilation_counters()
        forward_counters.append({k: after[k] - before[k] for k in after})
        return out

    adapter.module = counted_forward

    class Observed:
        """The orchestrator's DiT callable around the adapter: call starts, timesteps, outputs."""

        def __init__(self, inner, capture: bool):
            self.inner, self.capture = inner, capture
            self.dtype, self.config = inner.dtype, inner.config
            self.starts: list[float] = []
            self.timesteps: list[float] = []
            self.outputs: list[torch.Tensor] = []

        def __call__(self, hidden_states, timestep, encoder_hidden_states):
            self.starts.append(time.perf_counter())
            self.timesteps.append(float(timestep.flatten()[0]))
            out = self.inner(hidden_states, timestep, encoder_hidden_states)
            if self.capture:
                self.outputs.append(out.detach().to(torch.float32).clone())
            return out

    observed = Observed(adapter, capture=ref is not None)
    pipeline.transformer = observed

    stamped_step = pipeline._scheduler_step
    step_hashes: list[str] = []
    step_latents: list[torch.Tensor] = []

    def observed_step(*step_args, **step_kwargs):
        latents = stamped_step(*step_args, **step_kwargs)  # step_hook has stamped already
        if args.hash_steps:
            step_hashes.append(sha256(latents))
        if ref is not None:
            step_latents.append(latents.detach().clone())
        return latents

    pipeline._scheduler_step = observed_step

    init: list[torch.Tensor] = []
    original_prepare = pipeline.prepare_latents

    def observed_prepare(*prepare_args, **prepare_kwargs):
        latents = original_prepare(*prepare_args, **prepare_kwargs)
        init.append(latents.detach().clone())
        return latents

    pipeline.prepare_latents = observed_prepare

    negatives: list[torch.Tensor] = []
    original_negative = pipeline._negative_prompt_embeds

    def observed_negative(prompt_embeds):
        embeds = original_negative(prompt_embeds)
        negatives.append(embeds.detach().clone())
        return embeds

    pipeline._negative_prompt_embeds = observed_negative

    prompt_encodes: list[float] = []
    original_encode_prompt = pipeline.encode_prompt

    def timed_encode_prompt(**kwargs):
        begin = time.perf_counter()
        embeds = original_encode_prompt(**kwargs)
        if kwargs.get("prompt") is not None or kwargs.get("input_ids") is not None:
            prompt_encodes.append(time.perf_counter() - begin)
        return embeds

    pipeline.encode_prompt = timed_encode_prompt

    # --------------------------------------------------------------- the generate
    snaps["before_generate"] = snapshot(t0)
    begin = time.perf_counter()
    with track_fallbacks() as fallbacks:
        out = app(
            generator=torch.Generator().manual_seed(args.seed),
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            output_type="latent",
        )
    generate_s = time.perf_counter() - begin
    fallbacks = list(fallbacks)
    snaps["end"] = snapshot(t0)
    latents = out.latents.detach().to(torch.float32)
    evidence = parity.device_evidence(app, fallbacks)
    neffs_after = neff_files()

    iterations = []
    if step_stamps and observed.starts:
        iterations = [step_stamps[0] - observed.starts[0]] + [
            b - a for a, b in zip(step_stamps, step_stamps[1:])]
    per_forward = list(forward_s)

    # ------------------------------------------------------------------ checks
    checks: list[tuple[str, bool, str]] = []
    dit_app = app.transformer
    config = app.config
    report = dit_app.load_report or {}
    checks.append(("checkpoint", not report.get("missing") and not report.get("unexpected"),
                   f"missing={len(report.get('missing', []))} "
                   f"unexpected={len(report.get('unexpected', []))}"))
    attn = dit_app.module.blocks[0].attn1
    checks.append(("heads_per_rank", attn.local_heads == config.num_attention_heads // world,
                   f"local_heads={attn.local_heads} (={config.num_attention_heads}/{world})"))
    checks.append(("fallbacks", load_fallbacks == [] and fallbacks == [],
                   f"load+warm-up={load_fallbacks} generate={fallbacks}"))
    checks.append(("unwarmed_shapes", evidence["unwarmed_shapes"] == [],
                   f"{evidence['unwarmed_shapes']} warmup={dit_app.warmup_shapes}"))
    forwards_expected = args.steps * (2 if args.guidance_scale > 1.0 else 1)
    warm_shapes = [list(s) for s in (dit_app.warmup_shapes or [])]
    checks.append(("forward_shapes",
                   len(forward_shapes) == forwards_expected
                   and all(shapes == warm_shapes for shapes in forward_shapes),
                   f"{len(forward_shapes)} forwards (expected {forwards_expected}) at "
                   f"{sorted(set(map(str, forward_shapes)))} warmup={warm_shapes}"))
    checks.append(("steps", len(step_stamps) == args.steps and len(per_forward)
                   == forwards_expected,
                   f"{len(step_stamps)} step_hook stamps (expected {args.steps}), "
                   f"{len(per_forward)} forward_hook calls (expected {forwards_expected})"))
    forwards_delta = {k: sum(c[k] for c in forward_counters) for k in parity.COUNTER_NAMES}
    forward_neffs = forwards_delta["TotalCompilations"] + forwards_delta["PersistentHits"]
    graphs_load, graphs_end = snaps["after_load"]["dynamo"], snaps["end"]["dynamo"]
    no_new_graph = (args.exec_mode != "compile"
                    or graphs_end["unique_graphs"] == graphs_load["unique_graphs"])
    checks.append(("no_compile_in_forwards", forward_neffs == 0 and no_new_graph,
                   f"summed over {len(forward_counters)} DiT forwards {forwards_delta}, "
                   f"unique_graphs {graphs_load['unique_graphs']} -> "
                   f"{graphs_end['unique_graphs']}"))
    peak_gib = snaps["end"]["device_memory"]["max_allocated_gib"]
    enabled = bool(metrics_setup["enabled_before_prepare_runtime"])
    fresh = metrics_setup["runtime_initialized_when_enabled"] is False
    hits = int(forwards_delta["InMemoryHits"])
    checks.append(("instrumentation", enabled and fresh and hits > 0 and peak_gib > 0,
                   f"metrics enabled before prepare_runtime={enabled} on a fresh runtime="
                   f"{fresh}; forward InMemoryHits={hits} (> 0); max_memory_allocated="
                   f"{peak_gib:.2f} GiB (> 0)"))
    checks.append(("hbm_d48", peak_gib <= HBM_LIMIT_GIB,
                   f"max_memory_allocated after load="
                   f"{snaps['after_load']['device_memory']['max_allocated_gib']:.2f} end="
                   f"{peak_gib:.2f} GiB (limit {HBM_LIMIT_GIB})"))
    if args.exec_mode == "compile":
        checks.append(("compiled_blocks", len(app.compiled_blocks) == config.num_layers,
                       f"{len(app.compiled_blocks)} of {config.num_layers}"))
        checks.append(("one_graph", graphs_load["unique_graphs"] == 1
                       and graphs_end["unique_graphs"] == 1 and graphs_end["graph_breaks"] == 0,
                       f"unique_graphs after load={graphs_load['unique_graphs']} end="
                       f"{graphs_end['unique_graphs']} graph_breaks={graphs_end['graph_breaks']}"))
    checks.append(("text_encoder_load", "text encoder load" in app.phase_seconds,
                   f"{app.phase_seconds.get('text encoder load', math.nan):.1f}s"))
    negative = negatives[0] if negatives else None
    checks.append(("negative_embeds",
                   args.guidance_scale <= 1.0
                   or (negative is not None and bool(negative.abs().max() > 0)),
                   "not encoded" if negative is None else
                   f"absmax {float(negative.abs().max()):.4f}, shape {list(negative.shape)}"))
    vmhwm_gb = host_memory()["VmHWM_gib"] * GIB / 1e9
    if rank == 0:
        checks.append(("vmhwm_budget", vmhwm_gb <= VMHWM_BUDGET_GB,
                       f"rank-0 VmHWM {vmhwm_gb:.2f} GB ({vmhwm_gb * 1e9 / GIB:.2f} GiB) after "
                       f"the generate (budget {VMHWM_BUDGET_GB} GB)"))
    load_delta = delta(snaps["after_load"], snaps["start"])
    if args.expect == "warm":
        checks.append(("warm_load", load_delta["TotalCompilations"] == 0
                       and load_delta["PersistentHits"] >= 1,
                       f"load counters {load_delta}"))

    comparisons: dict[str, Any] = {}
    if ref is not None:
        runs = ref["runs"]
        init_ok = bool(init) and torch.equal(init[0], ref["latents_init"])
        checks.append(("init_latents", init_ok,
                       f"generator(seed {args.seed}) noise == reference latents_init: {init_ok}"))
        ok_p, detail_p = embeds_check(out.prompt_embeds, ref["prompt_embeds"],
                                      ref["prompt_tokens"])
        checks.append(("prompt_embeds", ok_p, detail_p))
        if negative is not None and ref["negative_prompt_embeds"] is not None:
            ok_n, detail_n = embeds_check(negative, ref["negative_prompt_embeds"],
                                          ref["negative_prompt_tokens"])
            checks.append(("negative_prompt_embeds", ok_n, detail_n))
        comparisons["steps"] = [
            comparison(step_latents[i], runs, lambda run, i=i: run["latents_steps"][i])
            for i in range(min(len(step_latents), args.steps))]
        comparisons["forwards"] = [
            comparison(observed.outputs[k], runs, lambda run, k=k: run["noise_preds"][k])
            for k in range(min(len(observed.outputs), forwards_expected))]
        final = comparison(latents, runs, lambda run: run["latents_final"])
        comparisons["final"] = final
        checks.append(("parity_latents_final", final["ok"], describe(final)))
        update = comparison(latents - ref["latents_init"], runs,
                            lambda run: run["latents_final"] - ref["latents_init"])
        comparisons["update"] = update
        update_ok = update["ok"] and update["vs_fp32"]["cosine"] >= UPDATE_MIN_COSINE
        checks.append(("update_final", update_ok,
                       f"cosine {update['vs_fp32']['cosine']:.6f} (min {UPDATE_MIN_COSINE}); "
                       + describe(update)))
        comparisons["timesteps"] = {"device": observed.timesteps,
                                    **{name: run["timesteps_seen"] for name, run in runs.items()}}

    ok = all(passed for _, passed, _ in checks)
    if rank == 0:
        torch.save(latents, args.out_dir / "latents.pt")
        if ref is not None:
            torch.save({"latents_steps": step_latents, "noise_preds": observed.outputs,
                        "prompt_embeds": out.prompt_embeds, "negative_prompt_embeds": negative,
                        "timesteps": observed.timesteps},
                       args.out_dir / "device_steps.pt")

    run_delta = delta(snaps["end"], snaps["pre_prepare"])
    added_neffs = sorted(set(neffs_after) - set(neffs_before))
    result: dict[str, Any] = {
        "rank": rank, "world_size": world, "exec_mode": args.exec_mode, "expect": args.expect,
        "ok": ok, "started_epoch": started_epoch,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "HF_HUB_OFFLINE", "DIFFLET_NO_JEMALLOC", runtime.PG_TIMEOUT_ENV,
            runtime.VISIBLE_CORES_ENV, "DIFFLET_DISABLE_PREWARM", "DIFFLET_COMPILE_CACHE",
            "TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_HLO_CACHE_DIR")},
        "torch_threads": torch.get_num_threads(),
        "metrics_setup": metrics_setup,
        "checks": [{"name": n, "ok": p, "detail": d} for n, p, d in checks],
        "comparisons": comparisons,
        "device_evidence": evidence,
        "load_fallbacks": load_fallbacks,
        "hashes": {
            "init_latents": sha256(init[0]) if init else None,
            "latents_final": sha256(latents),
            "latents_steps": step_hashes,
            "prompt_embeds": sha256(out.prompt_embeds),
            "negative_prompt_embeds": sha256(negative) if negative is not None else None,
        },
        "timesteps_seen": observed.timesteps,
        "phase_seconds": dict(app.phase_seconds),
        "timers_s": {
            "prepare_runtime": prepare_s, "load": load_s, "generate": generate_s,
            "iterations": iterations, "forwards": per_forward,
            "prompt_encodes": prompt_encodes, "first_forward_after_generate_start":
                (observed.starts[0] - begin) if observed.starts else None,
        },
        "iteration_stats_excl_first": stats(iterations[1:]),
        "forward_stats": stats(per_forward),
        "encodes": encodes,
        "encoder_threads": args.encoder_threads,
        "encoder_threads_probe": probe,
        "encoder_threads_probe_fastest": probe_fastest,
        "counters": {
            "prepare_runtime": delta(snaps["start"], snaps["pre_prepare"]),
            "load": load_delta,
            "probe": delta(snaps["after_probe"], snaps["after_load"]),
            "generate": delta(snaps["end"], snaps["before_generate"]),
            "forwards": forwards_delta, "run": run_delta,
        },
        "neffs_before_prepare_runtime": len(neffs_before),
        "new_neff_files": {path: neffs_after[path] for path in added_neffs},
        "max_memory_allocated_gib": peak_gib,
        "snapshots": snaps,
        "vmhwm_gb_after_generate": vmhwm_gb,
        "warmup_shapes": warm_shapes,
        "forward_shapes": forward_shapes,
        "compiled_blocks": len(app.compiled_blocks),
        "load_report": {k: len(v) for k, v in report.items()},
    }
    result_path = args.out_dir / f"result-rank{rank}.json"
    result_path.write_text(json.dumps(result, indent=2))

    if rank == 0:
        for name, passed, detail in checks:
            print(f"{'PASS' if passed else 'FAIL'} {name:24s} {detail}", flush=True)
        if ref is not None:
            for i, entry in enumerate(comparisons["steps"]):
                print(f"step {i + 1} latents: {describe(entry)}", flush=True)
            for k, entry in enumerate(comparisons["forwards"]):
                print(f"forward {k} (step {k // 2 + 1} {'uncond' if k % 2 else 'cond'}, t="
                      f"{observed.timesteps[k]:g}): {describe(entry)}", flush=True)
    failed = [name for name, passed, _ in checks if not passed]
    it = result["iteration_stats_excl_first"]
    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'} generate {args.steps} steps g="
          f"{args.guidance_scale:g}: load {load_s:.1f}s generate {generate_s:.1f}s iteration "
          f"median {it.get('median', math.nan):.3f}s forward median "
          f"{result['forward_stats'].get('median', math.nan):.3f}s latents "
          f"{result['hashes']['latents_final'][:16]} HBM {peak_gib:.2f}GiB VmHWM "
          f"{vmhwm_gb:.2f}GB forward_counters={forwards_delta} run_counters={run_delta} new_neffs="
          f"{len(added_neffs)}" + (f" failed={failed}" if failed else ""), flush=True)
    if not ok:
        return 1
    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0 and args.mp4 is not None:
        # The decode runs on the host after the peers are done (the CLI does it in the driver
        # with --host-vae); umT5 is not needed any more.
        encoder.model = None
        gc.collect()
        decode = decode_mp4(args, latents, args.out_dir / "latents.pt")
        result["decode"] = decode
        checks.append(("decode", decode["ok"], decode["detail"]))
        print(f"{'PASS' if decode['ok'] else 'FAIL'} {'decode':24s} {decode['detail']}",
              flush=True)
        result["ok"] = ok = ok and decode["ok"]
        result["checks"] = [{"name": n, "ok": p, "detail": d} for n, p, d in checks]
    result["timers_s"]["total"] = time.perf_counter() - t0
    result["host_memory_end"] = host_memory()
    result_path.write_text(json.dumps(result, indent=2))
    print(f"[rank {rank}] done in {result['timers_s']['total']:.1f}s, ok={ok}", flush=True)
    return 0 if ok else 1


def decode_mp4(args: argparse.Namespace, latents: torch.Tensor, latents_path: Path
               ) -> dict[str, Any]:
    """``_decode_latents_host`` (the --host-vae path) on rank 0, timed, with the frames checked
    before export."""
    from diffusers import AutoencoderKLWan

    from difflet.cli.orchestrators import wan as wan_cli

    saved = torch.load(latents_path, weights_only=True)
    timings: dict[str, float] = {}
    frames_info: dict[str, Any] = {}
    original_decode = AutoencoderKLWan.decode
    original_save = wan_cli._save_video

    def timed_decode(self, *decode_args, **decode_kwargs):
        begin = time.perf_counter()
        result = original_decode(self, *decode_args, **decode_kwargs)
        timings["vae_decode_s"] = time.perf_counter() - begin
        return result

    def checked_save(tensor, output_path):
        frames = tensor.detach().to(torch.float32)
        clipped = frames.clamp(-1, 1)
        frames_info.update({
            "shape": list(frames.shape), "finite": bool(torch.isfinite(frames).all()),
            "min": float(frames.min()), "max": float(frames.max()),
            "per_frame_std": [float(clipped[:, :, i].std()) for i in range(frames.shape[2])],
        })
        begin = time.perf_counter()
        saved_ok = original_save(tensor, output_path)
        timings["mp4_write_s"] = time.perf_counter() - begin
        frames_info["saved_mp4"] = bool(saved_ok)
        return saved_ok

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(args.decode_threads)
    AutoencoderKLWan.decode = timed_decode
    wan_cli._save_video = checked_save
    begin = time.perf_counter()
    try:
        wan_cli._decode_latents_host(str(latents_path), args.model_id, str(args.mp4),
                                     revision=args.revision)
    finally:
        AutoencoderKLWan.decode = original_decode
        wan_cli._save_video = original_save
        torch.set_num_threads(previous_threads)
    timings["decode_latents_host_s"] = time.perf_counter() - begin
    expected_shape = [1, 3, args.num_frames, args.height, args.width]
    stds = frames_info.get("per_frame_std", [])
    exists = args.mp4.is_file()
    ok = (torch.equal(saved, latents) and frames_info.get("shape") == expected_shape
          and frames_info.get("finite", False) and bool(stds) and min(stds) > MIN_FRAME_STD
          and frames_info.get("saved_mp4", False) and exists)
    detail = (f"latents.pt == final latents: {torch.equal(saved, latents)}; frames "
              f"{frames_info.get('shape')} (expected {expected_shape}) finite="
              f"{frames_info.get('finite')} range [{frames_info.get('min', math.nan):.3f}, "
              f"{frames_info.get('max', math.nan):.3f}] per-frame std min "
              f"{min(stds) if stds else math.nan:.3f} (> {MIN_FRAME_STD}); mp4 {args.mp4} "
              f"written={frames_info.get('saved_mp4')} exists={exists}; VAE decode "
              f"{timings.get('vae_decode_s', math.nan):.1f}s, whole host decode "
              f"{timings['decode_latents_host_s']:.1f}s ({args.decode_threads} threads)")
    return {"ok": bool(ok), "detail": detail, "timings": timings, "frames": frames_info,
            "mp4": str(args.mp4), "mp4_sha256": file_sha256(args.mp4) if exists else None,
            "threads": args.decode_threads,
            "vmhwm_gib_after_decode": host_memory()["VmHWM_gib"]}


# ------------------------------------------------------------------- the driver side


def read_back_mp4(path: Path) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np

    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        meta = reader.get_meta_data()
        frames = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    return {"frames": len(frames), "fps": meta.get("fps"), "codec": meta.get("codec"),
            "size": list(meta.get("size") or []),
            "frame_shape": list(frames[0].shape) if frames else None,
            "per_frame_std_uint8": [float(f.std()) for f in frames],
            "sha256": file_sha256(path), "bytes": path.stat().st_size}


def summarize(args: argparse.Namespace) -> int:
    directory = args.summarize
    paths = sorted(directory.glob("result-rank*.json"))
    if not paths:
        print(f"FAIL {args.tag}: no result-rank*.json under {directory} (did the launch run, "
              "and did every rank get as far as writing its result?)", flush=True)
        return 1
    results = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["rank"])
    problems: list[str] = []
    world = results[0]["world_size"] if results else 0
    if [r["rank"] for r in results] != list(range(world)) or world != 4:
        problems.append(f"expected 4 ranks' results, found ranks {[r['rank'] for r in results]}")
    for r in results:
        failed = [c["name"] for c in r["checks"] if not c["ok"]]
        if not r["ok"] or failed:
            problems.append(f"rank {r['rank']} failed {failed}")
    hashes: dict[str, Any] = {}
    for key in ("init_latents", "latents_final", "latents_steps", "prompt_embeds",
                "negative_prompt_embeds"):
        values = [r["hashes"][key] for r in results]
        same = all(v == values[0] for v in values)
        hashes[key] = {"equal_on_all_ranks": same,
                       "value": values[0] if same else values}
        if not same:
            problems.append(f"{key} differs across ranks")
    if not results or not results[0]["hashes"]["latents_steps"] and results[0]["args"].get(
            "hash_steps", True):
        problems.append("no per-step hashes recorded")
    rank0 = results[0] if results else {}
    decode = rank0.get("decode")
    mp4 = None
    if rank0.get("args", {}).get("mp4"):
        if decode is None:
            problems.append("rank 0 recorded no decode")
        else:
            path = Path(decode["mp4"])
            if not path.is_file():
                problems.append(f"{path} does not exist")
            else:
                mp4 = read_back_mp4(path)
                a = rank0["args"]
                expected = {"frames": a["num_frames"], "fps": 16.0, "codec": "h264",
                            "size": [a["width"], a["height"]]}
                got = {"frames": mp4["frames"], "fps": float(mp4["fps"] or 0),
                       "codec": mp4["codec"], "size": mp4["size"]}
                mp4["expected"], mp4["ok"] = expected, got == expected
                if got != expected:
                    problems.append(f"mp4 read-back {got} != {expected}")
                if mp4["sha256"] != decode["mp4_sha256"]:
                    problems.append("mp4 changed since rank 0 wrote it")
    launch = None
    if args.launch_times is not None and args.launch_times.is_file():
        values = [float(v) for v in args.launch_times.read_text().split()]
        if len(values) >= 2:
            launch = {"start_epoch": values[0], "end_epoch": values[1],
                      "wall_s": values[1] - values[0]}

    def rank_max(getter):
        values = [getter(r) for r in results]
        values = [v for v in values if v is not None]
        return max(values) if values else None

    phases = rank0.get("phase_seconds", {})
    probe = rank0.get("encoder_threads_probe", [])
    probe_best = {}
    for entry in probe:
        if entry.get("encode_s") is not None:
            probe_best[str(entry["threads"])] = min(probe_best.get(str(entry["threads"]),
                                                                   math.inf), entry["encode_s"])
    timing = {
        "launch_wall_s": launch and launch["wall_s"],
        "rank0_process_total_s": rank0.get("timers_s", {}).get("total"),
        "prepare_runtime_s_rank_max": rank_max(lambda r: r["timers_s"]["prepare_runtime"]),
        "load_s_rank_max": rank_max(lambda r: r["timers_s"]["load"]),
        "phase_seconds_rank0": phases,
        "warmup_forward_s_rank_max": rank_max(lambda r: r["phase_seconds"].get(
            "warmup forward")),
        "text_encoder_load_s": phases.get("text encoder load"),
        "encode_probe_best_s_by_threads": probe_best,
        "encoder_threads": rank0.get("encoder_threads"),
        "encoder_threads_probe_fastest": rank0.get("encoder_threads_probe_fastest"),
        "generate_encodes_rank0": [e for e in rank0.get("encodes", [])
                                   if e["phase"] == "generate"],
        "prompt_encode_calls_s_rank0": rank0.get("timers_s", {}).get("prompt_encodes"),
        "generate_s_rank_max": rank_max(lambda r: r["timers_s"]["generate"]),
        "iteration_stats_excl_first_rank0": rank0.get("iteration_stats_excl_first"),
        "iteration_median_s_rank_max": rank_max(
            lambda r: r["iteration_stats_excl_first"].get("median")),
        "first_iteration_s_rank0": (rank0.get("timers_s", {}).get("iterations") or [None])[0],
        "forward_stats_rank0": rank0.get("forward_stats"),
        "forward_median_s_rank_max": rank_max(lambda r: r["forward_stats"].get("median")),
        "decode": decode and decode["timings"],
    }
    memory = {
        "hbm_max_gib_by_rank": [r["max_memory_allocated_gib"] for r in results],
        "rank0_vmhwm_gb_after_generate": rank0.get("vmhwm_gb_after_generate"),
        "rank0_vmhwm_gib_by_phase": {name: snap["host_memory"]["VmHWM_gib"]
                                     for name, snap in rank0.get("snapshots", {}).items()},
        "rank0_vmhwm_gib_after_decode": decode and decode.get("vmhwm_gib_after_decode"),
        "vmhwm_gib_by_rank_end": [r["snapshots"]["end"]["host_memory"]["VmHWM_gib"]
                                  for r in results],
        "host_mem_available_gib_min_seen": min(
            (snap["host_memory"]["host_MemAvailable_gib"] for r in results
             for snap in r["snapshots"].values()), default=None),
    }
    comparisons = rank0.get("comparisons", {})
    if comparisons:
        for r in results[1:]:
            if json.dumps(r["comparisons"], sort_keys=True) != json.dumps(comparisons,
                                                                          sort_keys=True):
                problems.append(f"rank {r['rank']}'s comparisons differ from rank 0's")
    ok = not problems
    summary = {
        "tag": args.tag, "ok": ok, "problems": problems, "run_dir": str(directory),
        "head_note": "see HOW_TO_VERIFY for the commit the run used",
        "settings": {k: rank0.get("args", {}).get(k) for k in (
            "prompt", "steps", "guidance_scale", "seed", "height", "width", "num_frames",
            "exec_mode", "cache_dir", "ref", "mp4")},
        "checks_rank0": rank0.get("checks"),
        "hashes": hashes,
        "parity": {
            "final": comparisons.get("final"), "update": comparisons.get("update"),
            "steps": comparisons.get("steps"), "forwards": comparisons.get("forwards"),
            "timesteps": comparisons.get("timesteps"),
        } if comparisons else None,
        "timesteps_seen_rank0": rank0.get("timesteps_seen"),
        "timing": timing,
        "memory": memory,
        "counters_by_rank": [r["counters"] for r in results],
        "new_neff_files_by_rank": [r["new_neff_files"] for r in results],
        "fallbacks_by_rank": [r["device_evidence"]["fallbacks"] for r in results],
        "unwarmed_shapes_by_rank": [r["device_evidence"]["unwarmed_shapes"] for r in results],
        "decode": decode,
        "mp4_read_back": mp4,
        "launch": launch,
    }
    out_dir = args.result_dir or directory.parent
    out_path = out_dir / f"result-{args.tag}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"{'PASS' if ok else 'FAIL'} {args.tag}: 4 ranks ok="
          f"{all(r['ok'] for r in results)}; final latents "
          f"{'identical' if hashes['latents_final']['equal_on_all_ranks'] else 'DIFFER'} "
          f"({hashes['latents_final']['value'] if hashes['latents_final']['equal_on_all_ranks'] else ''}); "
          f"per-step hashes {'identical' if hashes['latents_steps']['equal_on_all_ranks'] else 'DIFFER'} "
          f"({len(results[0]['hashes']['latents_steps']) if results else 0} steps); "
          f"mp4 {mp4 and {k: mp4[k] for k in ('frames', 'fps', 'codec', 'size', 'sha256')}}"
          + (f"; problems: {problems}" if problems else "") + f" -> {out_path}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    parsed = parse_args()
    sys.exit(summarize(parsed) if parsed.summarize is not None else main(parsed))
