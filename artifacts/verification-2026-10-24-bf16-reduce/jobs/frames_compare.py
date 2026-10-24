"""frames_compare.py <ref out.png|out.mp4> <cand ...>: PSNR/SSIM via scripts/quality_vs_reference."""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "qvr", "/home/ubuntu/Difflet/.claude/worktrees/bf16-tp-reduce-qwen-ltx2/scripts/quality_vs_reference.py"
)
qvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qvr)
ref, cand = Path(sys.argv[1]), Path(sys.argv[2])
metrics = qvr.frame_metrics(qvr._read_frames(ref.parent), qvr._read_frames(cand.parent))
metrics.pop("psnr_per_frame")
print(json.dumps({"ref": str(ref), "cand": str(cand), **metrics}))
