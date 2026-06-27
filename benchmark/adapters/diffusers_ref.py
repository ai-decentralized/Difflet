"""Reference backend adapter: stock Hugging Face diffusers on CPU/CUDA.

Exists to prove the harness is genuinely backend-generic — the same metrics and
runner drive a non-Trainium backend with no changes. It runs the upstream
diffusers pipeline eagerly (no AOT compile, so ``compile_seconds`` is 0) and is
the natural apples-to-apples reference for the Trainium numbers.

It populates the *same* universal metrics the Trainium adapter does, so an
H100/CUDA report is directly comparable to a trn2 one:

  * ``load_seconds``  — ``from_pretrained(...).to(device)`` weights load
  * ``e2e_breakdown`` — single ``pipeline load`` stage so the report shows the
                        cold load-vs-compute split
  * ``step_seconds``  — per denoise step wall (via ``callback_on_step_end``,
                        cuda-synchronized; the first step is dropped as warmup)
  * ``peak_mem_gb``   — ``torch.cuda.max_memory_allocated`` peak during generate
  * ``output``        — shape / dtype / finite / value-range of the result

The pinned HF ``revision`` from the MATRIX is honored so the exact same weights
are loaded as on Trainium. Eager/heavyweight: a real generate at the MATRIX
shape, one model at a time.
"""
from __future__ import annotations

import inspect
import os
import time

from benchmark.harness import BackendAdapter, OutputInfo


class DiffusersRefAdapter(BackendAdapter):
    name = "diffusers"

    def __init__(self, device: str = "cpu"):
        self.device = device

    # -- info ------------------------------------------------------------- #
    def device_info(self) -> str:
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                return f"CUDA / {torch.cuda.get_device_name(0)}"
            import platform
            return f"CPU / {platform.processor() or platform.machine()}"
        except Exception:
            return self.device

    def toolchain(self) -> dict[str, str]:
        try:
            import importlib.metadata as m
            tc = {p: m.version(p) for p in ("torch", "diffusers", "transformers",
                                            "accelerate") if _safe(m, p)}
            try:
                import torch
                if torch.cuda.is_available():
                    tc["cuda"] = torch.version.cuda or "?"
            except Exception:
                pass
            return tc
        except Exception:
            return {}

    # -- phases ----------------------------------------------------------- #
    def prepare(self, spec) -> None:
        from diffusers import DiffusionPipeline
        DiffusionPipeline.download(spec.model_id, revision=getattr(spec, "revision", None))

    def compile(self, spec) -> tuple[float, dict[str, float]]:
        return 0.0, {"eager": 0.0}  # eager backend: no AOT compile

    def run_generate(self, spec) -> dict:
        """One full GPU-resident generate at the MATRIX default config. If the
        model does not fit (CUDA OOM) it is reported as a failure by the runner —
        no offload fallback, so every reported number is a clean default-config run."""
        import torch
        from diffusers import DiffusionPipeline

        dtype = torch.bfloat16 if spec.dtype == "bf16" else torch.float32
        is_cuda = self.device == "cuda" and torch.cuda.is_available()
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ---- weights load (timed separately so e2e splits load vs compute) --
        t0 = time.perf_counter()
        pipe = DiffusionPipeline.from_pretrained(
            spec.model_id, revision=getattr(spec, "revision", None), torch_dtype=dtype,
        ).to(self.device)
        if is_cuda:
            torch.cuda.synchronize()
        load_s = time.perf_counter() - t0

        # ---- VAE tiling/slicing: the standard single-GPU setting for video VAE
        # decode (stock diffusers otherwise decodes all frames at once and OOMs an
        # 80GB GPU). Identical frames, DiT per-step latency unaffected; no-op for
        # the small image VAEs. Mirrors the trn2 path's segmented/tiled decoder.
        # Disable with DIFFLET_BENCH_VAE_TILING=0 — needed for LTX-2, whose small
        # latents tile into a degenerate (size-1) dim that crashes its decoder conv.
        tiled = (_enable_vae_tiling(pipe)
                 if os.environ.get("DIFFLET_BENCH_VAE_TILING", "1") != "0" else False)

        # ---- assemble call kwargs, keeping only what this pipeline accepts --
        sig = _call_params(pipe)
        kwargs: dict = {"prompt": spec.prompt, "num_inference_steps": spec.steps}
        if spec.guidance_scale is not None and "guidance_scale" in sig:
            kwargs["guidance_scale"] = spec.guidance_scale
        if spec.height and "height" in sig:
            kwargs["height"] = spec.height
        if spec.width and "width" in sig:
            kwargs["width"] = spec.width
        if spec.num_frames and "num_frames" in sig:
            kwargs["num_frames"] = spec.num_frames
        # HARD GUARANTEE: input size (H×W×F) and step count must match the trn2
        # MATRIX exactly — never silently fall back to a pipeline default size.
        for dim in ("height", "width", "num_frames"):
            if getattr(spec, dim) is not None and dim not in kwargs:
                raise ValueError(
                    f"{type(pipe).__name__} does not accept '{dim}'; cannot run at the "
                    f"trn2 input size {spec.height}x{spec.width}x{spec.num_frames}. "
                    "Refusing to benchmark a different shape than trn2.")
        assert kwargs["num_inference_steps"] == spec.steps  # steps == trn2
        if "generator" in sig:
            g = torch.Generator(device="cuda" if is_cuda else "cpu")
            kwargs["generator"] = g.manual_seed(int(getattr(spec, "seed", 42)))
        if "output_type" in sig:
            kwargs["output_type"] = "pt"  # tensors -> we can compute output stats

        # ---- per-step latency via callback (cuda-synchronized) -------------
        step_ts: list[float] = []
        if "callback_on_step_end" in sig:
            def _cb(p, step, timestep, cbk):
                if is_cuda:
                    torch.cuda.synchronize()
                step_ts.append(time.perf_counter())
                return cbk
            kwargs["callback_on_step_end"] = _cb

        # ---- generate ------------------------------------------------------
        t1 = time.perf_counter()
        out = pipe(**kwargs)
        if is_cuda:
            torch.cuda.synchronize()
        compute_s = time.perf_counter() - t1
        wall = load_s + compute_s

        note = ("eager diffusers (gpu-resident): one fused load of all components; residual "
                "is text-encode + denoise loop + VAE decode (no AOT compile).")
        if tiled:
            note += (" VAE tiling/slicing enabled (standard single-GPU video-decode "
                     "setting; identical frames, DiT per-step unaffected).")
        res: dict = {
            "wall_seconds": wall,
            "load_seconds": load_s,
            "e2e_breakdown": {
                "stages": [{"stage": "pipeline load (from_pretrained → device)",
                            "shard_s": None, "load_s": load_s}],
                "weights_shard_total_s": None,
                "weights_load_total_s": load_s,
                "wall_total_s": wall,
                "compute_and_overhead_s": compute_s,
                "note": note,
            },
        }
        # per-step: drop the first (kernel autotune / lazy-init warmup)
        deltas = [step_ts[i] - step_ts[i - 1] for i in range(1, len(step_ts))]
        if deltas:
            res["step_seconds"] = deltas
        if is_cuda:
            res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        res["output"] = _summarize_output(out, spec)

        del pipe, out
        if is_cuda:
            torch.cuda.empty_cache()
        return res


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _call_params(pipe) -> set[str]:
    try:
        return set(inspect.signature(pipe.__call__).parameters)
    except (ValueError, TypeError):
        return set()


def _enable_vae_tiling(pipe) -> bool:
    """Turn on VAE tiling + slicing wherever the pipeline/VAE exposes it. Returns
    True if anything was enabled. Standard single-GPU memory setting for video
    decode; a no-op (below tile threshold) for small image VAEs."""
    enabled = False
    for fn in ("enable_vae_tiling", "enable_vae_slicing"):
        m = getattr(pipe, fn, None)
        if callable(m):
            try:
                m(); enabled = True
            except Exception:
                pass
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        for fn in ("enable_tiling", "enable_slicing"):
            m = getattr(vae, fn, None)
            if callable(m):
                try:
                    m(); enabled = True
                except Exception:
                    pass
    return enabled


def _summarize_output(out, spec) -> dict:
    """Best-effort shape/dtype/finite/value-range over whatever the pipe returned."""
    import torch

    arr = getattr(out, "images", None)
    kind = "images"
    if arr is None:
        arr = getattr(out, "frames", None)
        kind = "frames"
    if arr is None:
        arr = out
        kind = "raw"

    # unwrap nested lists (e.g. frames -> [ [PIL, ...] ]) down to a leaf
    leaf = arr
    for _ in range(4):
        if isinstance(leaf, (list, tuple)):
            if not leaf:
                break
            leaf = leaf[0]
        else:
            break
    try:
        if isinstance(leaf, torch.Tensor):
            t = arr if isinstance(arr, torch.Tensor) else leaf
            t = t.detach().float().cpu()
        else:  # PIL.Image or ndarray
            import numpy as np
            if hasattr(leaf, "convert"):  # PIL
                t = torch.from_numpy(np.asarray(leaf)).float()
            else:
                t = torch.as_tensor(np.asarray(leaf)).float()
        return OutputInfo(
            shape=list(t.shape), dtype=str(t.dtype),
            finite=bool(torch.isfinite(t).all()),
            min=float(t.min()), max=float(t.max()),
            mean=float(t.mean()), std=float(t.std()),
            note=f"diffusers {kind} ({spec.output_kind})",
        ).__dict__
    except Exception as e:
        return OutputInfo(note=f"diffusers reference output ({kind}); stats failed: {e}").__dict__


def _safe(m, p) -> bool:
    try:
        m.version(p)
        return True
    except Exception:
        return False
