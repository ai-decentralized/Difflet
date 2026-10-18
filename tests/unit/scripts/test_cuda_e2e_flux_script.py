"""scripts/cuda_e2e_flux_vs_diffusers.py imports and parses on a CPU host (the run is GPU-only)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

import cuda_e2e_flux_vs_diffusers as flux_e2e  # noqa: E402
import cuda_e2e_vs_diffusers as qwen_e2e  # noqa: E402


def test_defaults_are_the_gated_flux_dev_run():
    args = flux_e2e.build_parser().parse_args(["--model-path", "/snap"])
    assert args.model_path == "/snap" and args.model_id == "black-forest-labs/FLUX.1-dev"
    assert args.steps == 28 and args.guidance_scale == 3.5 and args.seed == 0
    assert (args.height, args.width) == (1024, 1024)
    assert args.out == "/tmp/logs/cuda_e2e_flux.json" and args.images_dir == "/tmp/logs"


def test_metrics_are_the_qwen_oracle_s():
    # One metric definition for both e2e gates.
    assert flux_e2e.image_metrics is qwen_e2e.image_metrics
    assert flux_e2e._save is qwen_e2e._save


def test_timings_record_carries_load_generate_and_peak(monkeypatch):
    # F4b: the Flux oracle records per-side timings as the Wan oracle does.
    rec = flux_e2e.timings_record(load_s=12.345, generate_s=16.78, peak_bytes=68_440_000_000)
    assert rec == {"load_s": 12.3, "generate_s": 16.8, "peak_alloc_gb": 68.44}
    rec = flux_e2e.timings_record(load_s=1.0, generate_s=2.0, peak_bytes=None)
    assert rec["peak_alloc_gb"] is None
