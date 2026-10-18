"""scripts/cuda_bench_wan_dit.py on a CPU host: the parser, the identical-input construction and
the loop-overhead probe with a fake DiT (the forwards themselves are GPU-only). F4b task 3."""

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

import cuda_bench_wan_dit as bench  # noqa: E402

from difflet.models.wan.dit_config import WanDiTConfig  # noqa: E402

# Wan 2.2 T2V-A14B's transformer/config.json, the fields WanDiTConfig keeps.
_A14B = {
    "_class_name": "WanTransformer3DModel",
    "in_channels": 16,
    "out_channels": 16,
    "num_attention_heads": 40,
    "attention_head_dim": 128,
    "num_layers": 40,
    "text_dim": 4096,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "patch_size": [1, 2, 2],
    "eps": 1e-6,
}


def _snapshot(tmp_path, *, two_experts: bool = True) -> pathlib.Path:
    snap = tmp_path / "wan"
    for name in ("transformer", "transformer_2") if two_experts else ("transformer",):
        (snap / name).mkdir(parents=True)
        (snap / name / "config.json").write_text(json.dumps(_A14B))
    (snap / "scheduler").mkdir()
    (snap / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "UniPCMultistepScheduler",
                "num_train_timesteps": 1000,
                "prediction_type": "flow_prediction",
                "use_flow_sigmas": True,
                "flow_shift": 3.0,
            }
        )
    )
    (snap / "model_index.json").write_text(json.dumps({"boundary_ratio": 0.875}))
    return snap


def test_defaults_are_the_f1_wan_shape_and_the_high_noise_expert():
    args = bench.build_parser().parse_args(["--model-path", "/snap"])
    assert args.expert == "transformer" and args.cfg_batch == 1
    assert (args.height, args.width, args.num_frames, args.text_seq_len) == (480, 832, 9, 512)
    assert args.timestep == 999.0 and args.warmup == 3 and args.iters == 10
    assert args.cases == "difflet,diffusers" and not args.loop_probe and args.probe_steps == 40
    args = bench.build_parser().parse_args(
        ["--model-path", "/snap", "--expert", "transformer_2", "--cfg-batch", "2", "--loop-probe"]
    )
    assert args.expert == "transformer_2" and args.cfg_batch == 2 and args.loop_probe


def test_bench_inputs_are_the_expert_s_latent_geometry_in_the_model_dtype(tmp_path):
    cfg = WanDiTConfig.from_pretrained(
        _snapshot(tmp_path) / "transformer", height=480, width=832, num_frames=9, text_seq_len=512
    )
    hidden, timestep, text = bench.bench_inputs(
        cfg, batch=2, timestep=999.0, device=torch.device("cpu"), dtype=torch.bfloat16, seed=0
    )
    # (B, C, F', H/8, W/8): 9 frames -> 3 latent frames, 480x832 -> 60x104.
    assert tuple(hidden.shape) == (2, 16, 3, 60, 104) and hidden.dtype == torch.bfloat16
    # The timestep is what Difflet's _CudaDiT hands the expert: fp32, one per sample.
    assert tuple(timestep.shape) == (2,) and timestep.dtype == torch.float32
    assert timestep.tolist() == [999.0, 999.0]
    assert tuple(text.shape) == (2, 512, 4096) and text.dtype == torch.bfloat16
    again, _, _ = bench.bench_inputs(
        cfg, batch=2, timestep=999.0, device=torch.device("cpu"), dtype=torch.bfloat16, seed=0
    )
    assert torch.equal(hidden, again)  # seeded: both sides see the same noise


def test_loop_probe_counts_the_dit_calls_and_separates_host_time_from_dit_time(tmp_path):
    snap = _snapshot(tmp_path)
    probe = bench.loop_probe(
        str(snap),
        steps=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        height=480,
        width=832,
        num_frames=9,
        text_seq_len=512,
    )
    # 4 steps x (cond + uncond) with true CFG (guidance 4.0 / 3.0): 8 DiT calls.
    assert probe["steps"] == 4 and probe["dit_calls"] == 8
    assert (
        probe["experts"] == {"transformer": 2, "transformer_2": 6}
        or sum(probe["experts"].values()) == 8
    )
    assert probe["total_ms"] >= probe["dit_ms"] >= 0.0
    assert probe["host_ms"] == probe["total_ms"] - probe["dit_ms"]
    assert probe["per_step_ms"] == probe["total_ms"] / 4
    assert probe["host_per_step_ms"] == probe["host_ms"] / 4
    assert probe["scheduler"] == "UniPCMultistepScheduler"
    assert tuple(probe["latent_shape"]) == (1, 16, 3, 60, 104)
