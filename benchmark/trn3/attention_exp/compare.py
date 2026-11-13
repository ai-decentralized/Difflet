"""Summarize ab_e2e.sh: per model, the denoising time with each exp engine, the PSNR between the
two outputs, and the output files' hashes.

    python benchmark/trn3/attention_exp/compare.py OUT_DIR

Step times come from each model's own log line: the in-run step median where the model logs one
(FLUX, Qwen-Image, Wan, HunyuanVideo 1.5); otherwise (HunyuanVideo, LTX-2) the DiT stage time
minus its load and warm-up, divided by the step count, which also holds the prompt encode.
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

MODELS = [("flux", "png", 50), ("qwen", "png", 50), ("wan", "mp4", 40), ("hv15", "mp4", 30),
          ("hv", "mp4", 30), ("ltx", "mp4", 30)]


def step_ms(log: Path, steps: int):
    text = log.read_text(errors="replace").replace("\r", "\n")
    m = re.findall(r"step median ([\d.]+) ms", text) or re.findall(r"median ([\d.]+) ms", text)
    if m:
        return float(m[-1]), "step median"
    m = re.findall(r"load\+warmup ([\d.]+)s, stage ([\d.]+)s", text)
    if m:
        load, stage = map(float, m[-1])
        return 1000 * (stage - load) / steps, "(stage - load) / steps"
    return None, "not found"


def frames(path: Path) -> np.ndarray:
    if path.suffix == ".png":
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)[None]
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]).astype(np.float64)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a - b) ** 2).mean(axis=tuple(range(1, a.ndim)))
    return float(np.mean(10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-12))))


def main(out: Path):
    print("| model | Vector exp | Scalar exp | change | PSNR Vector vs Scalar | md5 Vector / Scalar |")
    print("|---|---|---|---|---|---|")
    for model, ext, steps in MODELS:
        logs = [out / f"{model}_{e}.log" for e in ("vector", "scalar")]
        files = [out / f"{model}_{e}.{ext}" for e in ("vector", "scalar")]
        if not all(p.exists() for p in logs + files):
            print(f"| {model} | missing | | | | |")
            continue
        (tv, how), (ts, _) = step_ms(logs[0], steps), step_ms(logs[1], steps)
        change = f"{100 * (ts - tv) / tv:+.1f}%" if tv and ts else "-"
        db = psnr(frames(files[0]), frames(files[1]))
        md5 = " / ".join(hashlib.md5(p.read_bytes()).hexdigest()[:12] for p in files)
        print(f"| {model} | {tv:.1f} ms | {ts:.1f} ms | {change} | {db:.1f} dB | {md5} |  <!-- {how} -->")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
