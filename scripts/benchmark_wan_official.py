"""Compile and run the pinned Wan 2.1 14B 480P/81-frame Diffusers profile.

Use --compile-only to prepare artifacts, then --generate-only in the same
output directory. Every invocation writes a separate receipt and stage logs.
Timing includes process startup, weight loading, both CFG branches, VAE, and
video export; it is not a denoiser-only or resident-serving measurement.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


MODEL = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
REVISION = "38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
PROMPT = "A cat walks on the grass, realistic"
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt_path = out / f"receipt-{stamp}.json"
    video = out / f"video-{stamp}.mp4"
    work = out / f"work-{stamp}"
    common = [
        "--model-id", MODEL, "--revision", REVISION, "--tp-degree", "4",
        "--height", "480", "--width", "832", "--num-frames", "81", "--wan-vae-chunked",
    ]
    source_files = set(Path("difflet/models/wan").rglob("*.py"))
    source_files.update(Path("difflet/backends/trainium/wan").rglob("*.py"))
    source_files.update(Path(p) for p in (
        "difflet/cli/main.py", "difflet/cli/stage.py", "difflet/cli/orchestrators/wan.py",
        "difflet/backends/trainium/utils/compile_serial.py", __file__,
    ))
    record = {
        "status": "running", "model_id": MODEL, "revision": REVISION,
        "shape": [480, 832, 81], "steps": 50, "guidance_scale": 5.0,
        "prompt": PROMPT, "negative_prompt": NEGATIVE, "seed": 42,
        "dit_dtype": "bfloat16", "vae_dtype": "float32", "tp_degree": 4,
        "step_cache": False, "video_fps": 16,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)},
        "stages": [],
    }

    def save():
        receipt_path.write_text(json.dumps(record, indent=2) + "\n")

    def run(command, extra):
        argv = [sys.executable, "-m", "difflet.cli.main", command, *common, *extra]
        log = out / f"{command}-{stamp}.log"
        stage = {"command": argv, "log": str(log), "status": "running"}
        record["stages"].append(stage)
        save()
        print(f"Starting {command}; log: {log}", flush=True)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("BASE_COMPILE_WORK_DIR", str(out / "compiler"))
        start = time.perf_counter()
        with log.open("w") as handle:
            result = subprocess.run(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
        stage.update(seconds=time.perf_counter() - start, returncode=result.returncode,
                     status="passed" if result.returncode == 0 else "failed")
        save()
        if result.returncode:
            raise RuntimeError(f"{command} failed; see {log}")

    save()
    try:
        if not args.generate_only:
            run("compile", [])
        if not args.compile_only:
            run("generate", [
                "--prompt", PROMPT, "--negative-prompt", NEGATIVE,
                "--steps", "50", "--guidance-scale", "5.0", "--seed", "42",
                "--output", str(video), "--work-dir", str(work), "--keep-work-dir",
            ])
            import torch

            latents = torch.load(work / "latents.pt", map_location="cpu", weights_only=True)
            assert list(latents.shape) == [1, 16, 21, 60, 104]
            assert torch.isfinite(latents).all()
            assert float(latents.float().std()) > 0
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate,codec_name",
                "-of", "json", str(video),
            ], text=True))
            stream = probe["streams"][0]
            assert (stream["height"], stream["width"], int(stream["nb_read_frames"])) == (480, 832, 81)
            record.update(video=str(video), video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                          video_probe=probe, latent_shape=list(latents.shape),
                          latent_dtype=str(latents.dtype), latent_finite=True,
                          latent_std=float(latents.float().std()))
        record["status"] = "passed"
        record["validation_scope"] = "compile_only" if args.compile_only else "text_to_video_e2e"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
        print(f"Receipt: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
