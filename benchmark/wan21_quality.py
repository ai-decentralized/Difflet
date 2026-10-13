"""Wan2.1 T2V 14B output quality, TorchNeuron (``neuron``) vs NxD AoT (``trainium``). CPU only.

Reads what ``benchmark/wan21_native_vs_nxd.py`` left under ``--out`` (``<backend>/<run>/``
with ``out.mp4`` and ``probe/latents-<pid>-<n>.pt`` = ``{"initial", "final"}`` written by
rank 0 of the denoise loop, see ``benchmark/wan_step_probe/sitecustomize.py``) and writes
``<out>/quality/``:

* ``metrics.json`` -- everything below, with the definitions used;
* ``sheets/<backend>_<run>.png`` -- all 9 frames of every run's video, 3x3, labelled;
* ``blind/{X,Y}_g{5,1}.png`` -- frames 0,2,4,6,8 in one 1664 px row, no labels; which
  backend is X/Y lives only in ``blind_key.json``.

Metrics:

1. **Initial noise**: sha256 of the ``initial`` latents of every run that has them.
2. **Final latents** (warm1, same guidance, neuron vs trainium): cosine, relative L2
   ``||n - t|| / ||t||``, max abs diff, PSNR with peak = range (max - min) of the trainium
   latents, cosine of the update ``final - initial``; per-run sha256 of ``final`` for
   within-backend determinism.
3. **Decoded videos** (warm1, neuron vs trainium per guidance): per-frame PSNR on RGB uint8
   (peak 255) and SSIM on BT.601 luma (Wang et al. 2004: 11x11 gaussian window, sigma 1.5,
   K1=0.01, K2=0.03, L=255, valid region, mean SSIM map). sha256 of the mp4 file and of the
   decoded RGB frames per run. The same metrics between guidance 5 and 1 of one backend are
   reported as a "different video, same prompt and seed" reference scale.
4. **Per-video stats**: per-channel mean/std/min/max, temporal smoothness (mean |frame[i+1]
   - frame[i]| over RGB), sharpness (variance of the 4-neighbour Laplacian of luma, mean
   over frames), saturation (fraction of RGB values <= 2 or >= 253).
5. **CLIP score**: openai/clip-vit-large-patch14 (fp32, CPU), cosine between the prompt
   embedding and each frame's image embedding (CLIPProcessor resize + 224 center crop),
   mean over frames; the same for three off-prompt control texts; margin = prompt score -
   best control score.
6. **CPU VAE reference** (separates DiT divergence from VAE-stage divergence): the warm1
   final latents of each backend decoded by the diffusers ``AutoencoderKLWan`` in fp32 on
   CPU (same de-normalisation and ``[-1, 1] -> uint8`` conversion as the CLI) and encoded
   with the CLI's own encoder (``difflet.serving.video_media.encode_tensor_to_mp4``, fps 16,
   libx264 crf 18) to ``ref/<backend>_<run>_cpuvae.mp4``. Reported: each backend's
   ``out.mp4`` vs the CPU decode of its own latents (the backend's VAE stage alone), and
   the two CPU decodes against each other (the latent divergence alone, in pixels).
   About 100 s and ~9 GB RSS per decode at 3 threads; ``--skip-cpu-vae`` turns it off.

Usage::

    python benchmark/wan21_quality.py --out /home/ubuntu/wan21-bench [--threads 3]
        [--skip-clip] [--skip-cpu-vae]
"""

from __future__ import annotations

import os

# Before torch/numpy load their thread pools: this runs next to device profiling.
_THREADS = os.environ.get("WAN_QUALITY_THREADS", "3")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, _THREADS)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
FPS = 16
PROMPT = "a cinematic shot of a red fox running through a snowy forest"
CONTROLS = ("a bowl of fruit on a kitchen table", "a city street at night with neon signs",
            "a portrait of an elderly man")
CLIP_ID = "openai/clip-vit-large-patch14"
CLIP_PATTERNS = ["*.json", "*.txt", "model.safetensors"]
BACKENDS = ("neuron", "trainium")
#: Neutral names for the blind sheets; the mapping is written to blind_key.json only.
BLIND = {"X": "trainium", "Y": "neuron"}
BLIND_FRAMES = (0, 2, 4, 6, 8)
BLIND_TILE = (332, 192)  # 5 * 332 + 4 * 1 px gap = 1664 px wide
SHEET_SCALE = 2  # per-run sheets: frames at half resolution
SAT_LOW, SAT_HIGH = 2, 253
LUMA = (0.299, 0.587, 0.114)  # BT.601


# --------------------------------------------------------------------------- inputs


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tensor(t: torch.Tensor) -> str:
    return _sha256_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def _guidance_by_run(out: Path) -> dict[tuple[str, str], float]:
    """(backend, run) -> guidance scale from the driver's runs.jsonl (last successful entry)."""
    guidance = {}
    runs = out / "runs.jsonl"
    if runs.exists():
        for line in runs.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("returncode") == 0 and rec.get("guidance") is not None:
                guidance[(rec["backend"], rec["name"])] = float(rec["guidance"])
    return guidance


def _discover(out: Path) -> list[dict]:
    """Every ``<backend>/<run>/out.mp4`` with its guidance and (optional) latents file."""
    guidance = _guidance_by_run(out)
    runs = []
    for backend in BACKENDS:
        for run_dir in sorted((out / backend).iterdir()):
            mp4 = run_dir / "out.mp4"
            if not mp4.is_file():
                continue
            g = guidance.get((backend, run_dir.name))
            if g is None:  # fall back to the driver's "-g<scale>" naming
                g = float(run_dir.name.rsplit("-g", 1)[1])
            latents = sorted((run_dir / "probe").glob("latents-*.pt"))
            runs.append({"backend": backend, "run": run_dir.name,
                         "key": f"{backend}/{run_dir.name}", "guidance": g, "mp4": mp4,
                         "latents": latents})
    return runs


def _decode(path: Path) -> np.ndarray:
    """All frames of an mp4 as uint8 RGB, shape (T, H, W, 3)."""
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


# --------------------------------------------------------------------------- metrics


def _latent_compare(a: torch.Tensor, b: torch.Tensor, a0: torch.Tensor | None,
                    b0: torch.Tensor | None) -> dict:
    """``a`` vs reference ``b`` (float64); a0/b0 the initial noise for the update cosine."""
    a, b = a.double(), b.double()
    diff = a - b
    mse = float(diff.pow(2).mean())
    peak = float(b.max() - b.min())
    res = {
        "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
        "rel_l2": float(diff.norm() / b.norm()),
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "psnr_db": (float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)),
        "psnr_peak": peak,
        # per latent frame (dim 2 of B,C,T,H,W)
        "cosine_per_latent_frame": [
            float(F.cosine_similarity(a[:, :, t].flatten(), b[:, :, t].flatten(), dim=0))
            for t in range(a.shape[2])],
    }
    if a0 is not None and b0 is not None:
        ua, ub = a - a0.double(), b - b0.double()
        res["update_cosine"] = float(F.cosine_similarity(ua.flatten(), ub.flatten(), dim=0))
        res["update_rel_l2"] = float((ua - ub).norm() / ub.norm())
    return res


def _luma(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 1, H, W) float64 luma in [0, 255]."""
    rgb = torch.from_numpy(frames).double()
    return (rgb @ torch.tensor(LUMA, dtype=torch.float64)).unsqueeze(1)


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def _ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 255.0) -> list[float]:
    """Per-image mean SSIM of (N, 1, H, W) tensors (separable gaussian, valid region)."""
    g = _gaussian_window()
    gh, gv = g.view(1, 1, 1, -1), g.view(1, 1, -1, 1)

    def blur(t):
        return F.conv2d(F.conv2d(t, gh), gv)

    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x * mu_x
    syy = blur(y * y) - mu_y * mu_y
    sxy = blur(x * y) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2))
    return ssim_map.mean(dim=(1, 2, 3)).tolist()


def _psnr(a: np.ndarray, b: np.ndarray) -> list[float]:
    """Per-frame PSNR (dB) of (T, H, W, 3) uint8 videos, peak 255."""
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(axis=(1, 2, 3))
    return [float("inf") if m == 0 else 10 * math.log10(255.0 ** 2 / m) for m in mse]


def _video_compare(a: np.ndarray, b: np.ndarray) -> dict:
    psnr = _psnr(a, b)
    ssim = _ssim(_luma(a), _luma(b))
    return {"psnr_db_per_frame": psnr, "psnr_db_mean": float(np.mean(psnr)),
            "psnr_db_min": float(np.min(psnr)),
            "ssim_luma_per_frame": ssim, "ssim_luma_mean": float(np.mean(ssim)),
            "ssim_luma_min": float(np.min(ssim)),
            "mean_abs_rgb_diff": float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())}


def _video_stats(frames: np.ndarray) -> dict:
    f = frames.astype(np.float64)
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                              dtype=torch.float64).view(1, 1, 3, 3)
    lap = F.conv2d(_luma(frames), lap_kernel)
    lap_var = lap.var(dim=(1, 2, 3)).tolist()
    temporal = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3)) if len(f) > 1 else np.zeros(0)
    return {
        "frames": int(frames.shape[0]), "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "channel_mean": f.mean(axis=(0, 1, 2)).tolist(),
        "channel_std": f.std(axis=(0, 1, 2)).tolist(),
        "channel_min": frames.min(axis=(0, 1, 2)).tolist(),
        "channel_max": frames.max(axis=(0, 1, 2)).tolist(),
        "temporal_mean_abs_diff": float(temporal.mean()) if temporal.size else None,
        "temporal_mean_abs_diff_per_pair": temporal.tolist(),
        "laplacian_var_mean": float(np.mean(lap_var)),
        "laplacian_var_per_frame": lap_var,
        "saturated_frac": float(((frames <= SAT_LOW) | (frames >= SAT_HIGH)).mean()),
        "saturated_low_frac": float((frames <= SAT_LOW).mean()),
        "saturated_high_frac": float((frames >= SAT_HIGH).mean()),
    }


# --------------------------------------------------------------------------- CLIP


def _clip_snapshot() -> tuple[str, str]:
    """Local CLIP snapshot dir and its commit; downloads once (online for that call only)."""
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(CLIP_ID, allow_patterns=CLIP_PATTERNS, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached yet: fetch in a child with the hub online
        code = ("import sys; from huggingface_hub import snapshot_download; "
                "print(snapshot_download(sys.argv[1], allow_patterns=sys.argv[2:]))")
        env = {**os.environ, "HF_HUB_OFFLINE": "0"}
        subprocess.run([sys.executable, "-c", code, CLIP_ID, *CLIP_PATTERNS], env=env,
                       check=True)
        path = snapshot_download(CLIP_ID, allow_patterns=CLIP_PATTERNS, local_files_only=True)
    return path, Path(path).name


class _Clip:
    def __init__(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.path, self.revision = _clip_snapshot()
        self.model = CLIPModel.from_pretrained(self.path, dtype=torch.float32).eval()
        self.processor = CLIPProcessor.from_pretrained(self.path)
        self.texts = [PROMPT, *CONTROLS]
        with torch.inference_mode():
            tok = self.processor(text=self.texts, return_tensors="pt", padding=True)
            self.text_emb = F.normalize(self.model.get_text_features(**tok), dim=-1)

    def score(self, frames: np.ndarray) -> dict:
        with torch.inference_mode():
            px = self.processor(images=list(frames), return_tensors="pt")["pixel_values"]
            img = F.normalize(self.model.get_image_features(pixel_values=px), dim=-1)
        cos = (img @ self.text_emb.T).double()  # (frames, texts)
        per_text = cos.mean(dim=0).tolist()
        prompt, controls = per_text[0], per_text[1:]
        return {
            "prompt_cosine_mean": prompt,
            "prompt_cosine_per_frame": cos[:, 0].tolist(),
            "control_cosine_mean": dict(zip(CONTROLS, controls)),
            "margin_vs_best_control": prompt - max(controls),
            "frames_prompt_is_top": int((cos.argmax(dim=1) == 0).sum()),
        }


# --------------------------------------------------------------------------- CPU VAE


class _CpuVae:
    """Reference decode of final latents: diffusers AutoencoderKLWan, fp32, CPU."""

    def __init__(self) -> None:
        from diffusers import AutoencoderKLWan
        from huggingface_hub import snapshot_download

        root = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=["vae/*"],
                                 local_files_only=True)
        self.vae = AutoencoderKLWan.from_pretrained(Path(root) / "vae",
                                                    torch_dtype=torch.float32).eval()
        cfg = self.vae.config
        shape = (1, int(cfg.z_dim), 1, 1, 1)
        self.mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).view(shape)
        self.std = torch.tensor(cfg.latents_std, dtype=torch.float32).view(shape)

    def decode(self, final: torch.Tensor, mp4: Path) -> tuple[np.ndarray, np.ndarray]:
        """(raw uint8 frames, frames decoded back from the CLI-encoded ``mp4``)."""
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from difflet.serving.video_media import encode_tensor_to_mp4, normalize_video_tensor

        with torch.inference_mode():
            video = self.vae.decode(final.float() * self.std + self.mean).sample
        video = video.float().clamp(-1.0, 1.0)  # BCTHW
        raw = normalize_video_tensor(video, layout="BCTHW", value_range="minus_one_to_one")
        encode_tensor_to_mp4(video, mp4, fps=FPS, layout="BCTHW",
                             value_range="minus_one_to_one")
        return raw, _decode(mp4)


# --------------------------------------------------------------------------- sheets


def _contact_sheet(frames: np.ndarray, title: str, cols: int = 3) -> Image.Image:
    t, h, w = frames.shape[:3]
    tw, th = w // SHEET_SCALE, h // SHEET_SCALE
    rows = math.ceil(t / cols)
    header, gap = 22, 2
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * gap, header + rows * (th + gap)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title, fill="black")
    for i, frame in enumerate(frames):
        x, y = (i % cols) * (tw + gap), header + (i // cols) * (th + gap)
        sheet.paste(Image.fromarray(frame).resize((tw, th), Image.LANCZOS), (x, y))
        draw.text((x + 4, y + 3), f"f{i}", fill="white", stroke_width=1, stroke_fill="black")
    return sheet


def _blind_strip(frames: np.ndarray) -> Image.Image:
    tw, th = BLIND_TILE
    gap = 1
    strip = Image.new("RGB", (len(BLIND_FRAMES) * tw + (len(BLIND_FRAMES) - 1) * gap, th),
                      "black")
    for i, idx in enumerate(BLIND_FRAMES):
        strip.paste(Image.fromarray(frames[idx]).resize((tw, th), Image.LANCZOS),
                    (i * (tw + gap), 0))
    return strip


# --------------------------------------------------------------------------- main


def _finite(x):
    """JSON-safe float (inf/nan -> string)."""
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def _group_equal(runs: list[dict], field: str) -> dict:
    hashes = {r["key"]: r.get(field) for r in runs if r.get(field)}
    return {"runs": hashes, "all_equal": len(set(hashes.values())) == 1 if hashes else None,
            "distinct": len(set(hashes.values()))}


def run(out: Path, threads: int, skip_clip: bool = False, skip_cpu_vae: bool = False) -> dict:
    torch.set_num_threads(threads)
    qdir = out / "quality"
    (qdir / "sheets").mkdir(parents=True, exist_ok=True)
    (qdir / "blind").mkdir(parents=True, exist_ok=True)
    started = time.time()

    runs = _discover(out)
    latents: dict[str, dict] = {}
    videos: dict[str, np.ndarray] = {}
    for r in runs:
        r["mp4_sha256"] = _sha256_file(r["mp4"])
        videos[r["key"]] = frames = _decode(r["mp4"])
        r["frames_sha256"] = _sha256_bytes(frames.tobytes())
        r["stats"] = _video_stats(frames)
        _contact_sheet(frames, f"{r['backend']}  {r['run']}  guidance={r['guidance']:g}"
                       ).save(qdir / "sheets" / f"{r['backend']}_{r['run']}.png")
        if r["latents"]:
            lat = torch.load(r["latents"][-1], map_location="cpu")
            latents[r["key"]] = lat
            r["latents_file"] = str(r["latents"][-1])
            r["latents_files_found"] = len(r["latents"])
            for name in ("initial", "final"):
                t = lat.get(name)
                if t is not None:
                    r[f"{name}_sha256"] = _sha256_tensor(t)
                    r[f"{name}_finite"] = bool(torch.isfinite(t).all())
                    r[f"{name}_shape"] = list(t.shape)
                    r[f"{name}_dtype"] = str(t.dtype)
        print(f"[quality] {r['key']}: {frames.shape} decoded", flush=True)

    by_key = {r["key"]: r for r in runs}

    def pick(backend: str, g: float, run_name: str = "warm1") -> str | None:
        key = f"{backend}/{run_name}-g{g:g}"
        return key if key in by_key else None

    guidances = sorted({r["guidance"] for r in runs}, reverse=True)

    # 1. initial noise
    with_init = [r for r in runs if r.get("initial_sha256")]
    initial = _group_equal(with_init, "initial_sha256")
    initial["runs_without_latents"] = [r["key"] for r in runs if not r.get("initial_sha256")]

    # 2. final latents across backends + reference scale (cross-guidance within a backend)
    def lat_pair(ka: str, kb: str) -> dict:
        la, lb = latents[ka], latents[kb]
        return {"a": ka, "b_reference": kb,
                **_latent_compare(la["final"], lb["final"], la.get("initial"), lb.get("initial"))}

    latent_cmp = {}
    for g in guidances:
        kn, kt = pick("neuron", g), pick("trainium", g)
        if kn in latents and kt in latents:
            latent_cmp[f"neuron_vs_trainium_g{g:g}"] = lat_pair(kn, kt)
    for b in BACKENDS:
        k5, k1 = pick(b, 5.0), pick(b, 1.0)
        if k5 in latents and k1 in latents:
            latent_cmp[f"reference_{b}_g5_vs_g1"] = lat_pair(k5, k1)

    # determinism within backend, per guidance
    determinism = {}
    for b in BACKENDS:
        for g in guidances:
            group = [r for r in runs if r["backend"] == b and r["guidance"] == g]
            warm = [r for r in group if r["run"].startswith("warm")]
            determinism[f"{b}_g{g:g}"] = {
                "warm_final_latents": _group_equal(warm, "final_sha256"),
                "warm_mp4": _group_equal(warm, "mp4_sha256"),
                "warm_decoded_frames": _group_equal(warm, "frames_sha256"),
                "all_runs_final_latents": _group_equal(group, "final_sha256"),
                "all_runs_mp4": _group_equal(group, "mp4_sha256"),
                "all_runs_decoded_frames": _group_equal(group, "frames_sha256"),
            }

    # 3. decoded video comparison
    video_cmp = {}
    for g in guidances:
        kn, kt = pick("neuron", g), pick("trainium", g)
        if kn and kt:
            video_cmp[f"neuron_vs_trainium_g{g:g}"] = {
                "a": kn, "b": kt, **_video_compare(videos[kn], videos[kt])}
    for b in BACKENDS:
        k5, k1 = pick(b, 5.0), pick(b, 1.0)
        if k5 and k1:
            video_cmp[f"reference_{b}_g5_vs_g1"] = {
                "a": k5, "b": k1, **_video_compare(videos[k5], videos[k1])}

    # 6. CPU VAE reference decode of the warm1 latents
    cpu_vae: dict = {}
    ref_videos: dict[str, np.ndarray] = {}
    if not skip_cpu_vae:
        (qdir / "ref").mkdir(exist_ok=True)
        vae = _CpuVae()
        for g in guidances:
            for b in BACKENDS:
                key = pick(b, g)
                if key not in latents:
                    continue
                t0 = time.time()
                stem = key.replace("/", "_")
                mp4 = qdir / "ref" / f"{stem}_cpuvae.mp4"
                raw, enc = vae.decode(latents[key]["final"], mp4)
                ref_videos[key] = enc
                _contact_sheet(enc, f"CPU fp32 VAE decode of {key} final latents"
                               ).save(qdir / "sheets" / f"cpuvae_{stem}.png")
                cpu_vae[key] = {
                    "mp4": str(mp4), "mp4_sha256": _sha256_file(mp4),
                    "decode_s": round(time.time() - t0, 1),
                    "backend_mp4_vs_cpu_vae_mp4": _video_compare(videos[key], enc),
                    "backend_mp4_vs_cpu_vae_raw": _video_compare(videos[key], raw),
                    "h264_loss_cpu_vae_mp4_vs_raw": _video_compare(enc, raw),
                    "stats": _video_stats(enc),
                }
                psnr = cpu_vae[key]["backend_mp4_vs_cpu_vae_mp4"]["psnr_db_mean"]
                print(f"[quality] {key}: CPU VAE decode {cpu_vae[key]['decode_s']} s, "
                      f"out.mp4 vs CPU VAE PSNR {psnr:.2f} dB", flush=True)
            kn, kt = pick("neuron", g), pick("trainium", g)
            if kn in ref_videos and kt in ref_videos:
                cpu_vae[f"neuron_vs_trainium_g{g:g}"] = {
                    "a": kn, "b": kt, **_video_compare(ref_videos[kn], ref_videos[kt])}
        del vae

    # 5. CLIP
    clip_meta = None
    if not skip_clip:
        clip = _Clip()
        clip_meta = {"model": CLIP_ID, "revision": clip.revision, "texts": clip.texts}
        for r in runs:
            r["clip"] = clip.score(videos[r["key"]])
            if r["key"] in ref_videos:
                cpu_vae[r["key"]]["clip"] = clip.score(ref_videos[r["key"]])
            print(f"[quality] {r['key']}: clip prompt={r['clip']['prompt_cosine_mean']:.4f} "
                  f"margin={r['clip']['margin_vs_best_control']:.4f}", flush=True)

    # 7. blind sheets: key kept out of metrics.json and stdout
    blind_key = {}
    for g in guidances:
        for label, backend in BLIND.items():
            key = pick(backend, g)
            if key is None:
                continue
            name = f"{label}_g{g:g}.png"
            _blind_strip(videos[key]).save(qdir / "blind" / name)
            blind_key[name] = key
    (qdir / "blind_key.json").write_text(json.dumps(blind_key, indent=2) + "\n")

    per_run = {}
    for r in runs:
        per_run[r["key"]] = {k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()
                             if k not in ("latents", "key")}
    metrics = {
        "meta": {
            "script": "benchmark/wan21_quality.py",
            "out": str(out),
            "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "elapsed_s": round(time.time() - started, 1),
            "torch_threads": threads,
            "prompt": PROMPT,
            "clip": clip_meta,
            "definitions": {
                "latent_rel_l2": "||a - b|| / ||b||, b = reference (trainium, or the g1 run)",
                "latent_psnr_db": "10 log10(peak^2 / MSE), peak = max(b) - min(b)",
                "update_cosine": "cosine of (final - initial) between the two runs",
                "video_psnr": "per frame, RGB uint8 decoded from out.mp4, peak 255",
                "ssim": "BT.601 luma, gaussian 11x11 sigma 1.5, K1=.01 K2=.03 L=255, valid",
                "temporal_mean_abs_diff": "mean |frame[i+1] - frame[i]| over RGB (0-255)",
                "laplacian_var": "variance of 4-neighbour Laplacian of luma, valid region",
                "saturated_frac": f"fraction of RGB values <= {SAT_LOW} or >= {SAT_HIGH}",
                "clip": "cosine(text_emb, image_emb), CLIPProcessor resize+center-crop 224",
                "margin_vs_best_control": "prompt cosine - max control cosine (frame means)",
                "comparisons": "cross-backend pairs use the warm1 run of each backend",
                "cpu_vae_reference": "diffusers AutoencoderKLWan fp32 CPU decode of the "
                                     "probe's final latents, encoded like the CLI (crf 18)",
            },
        },
        "initial_noise": initial,
        "final_latents": latent_cmp,
        "determinism": determinism,
        "video_compare": video_cmp,
        "cpu_vae_reference": cpu_vae or None,
        "runs": per_run,
        "sheets": sorted(str(p) for p in (qdir / "sheets").glob("*.png")),
        "blind_sheets": sorted(str(p) for p in (qdir / "blind").glob("*.png")),
    }
    (qdir / "metrics.json").write_text(json.dumps(_finite(metrics), indent=2) + "\n")
    print(f"[quality] wrote {qdir / 'metrics.json'} in {metrics['meta']['elapsed_s']} s")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="the benchmark output dir (has runs.jsonl)")
    parser.add_argument("--threads", type=int, default=int(_THREADS),
                        help="torch CPU threads (default 3; env WAN_QUALITY_THREADS)")
    parser.add_argument("--skip-clip", action="store_true", help="skip the CLIP score")
    parser.add_argument("--skip-cpu-vae", action="store_true",
                        help="skip the CPU fp32 VAE reference decode (~100 s per video)")
    args = parser.parse_args(argv)
    run(Path(args.out), args.threads, skip_clip=args.skip_clip, skip_cpu_vae=args.skip_cpu_vae)


if __name__ == "__main__":
    main()
