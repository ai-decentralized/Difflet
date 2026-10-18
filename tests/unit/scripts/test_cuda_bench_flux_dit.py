"""scripts/cuda_bench_flux_dit.py on a CPU host: the parser and the identical-input construction
(the forwards are GPU-only). F4c task 1."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

import cuda_bench_flux_dit as bench  # noqa: E402

from difflet.models.flux.dit_config import FluxDiTConfig  # noqa: E402


def _flux_dev_config(**over) -> FluxDiTConfig:
    # FLUX.1-dev's transformer/config.json, the fields FluxDiTConfig keeps.
    d = dict(
        patch_size=1,
        in_channels=64,
        out_channels=None,
        num_layers=19,
        num_single_layers=38,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=4096,
        pooled_projection_dim=768,
        guidance_embeds=True,
        height=1024,
        width=1024,
    )
    d.update(over)
    return FluxDiTConfig(**d)


def test_defaults_are_the_f5_flux_dev_run():
    args = bench.build_parser().parse_args(["--model-path", "/snap"])
    assert (args.height, args.width) == (1024, 1024)
    assert args.timestep == 0.5 and args.guidance == 3.5
    assert args.warmup == 3 and args.iters == 10 and args.cases == "difflet,diffusers"
    assert args.out == "/tmp/logs/bench_flux_dit.json"


def test_bench_inputs_match_the_dit_contract():
    cfg = _flux_dev_config()
    inputs = bench.bench_inputs(
        cfg, timestep=0.5, guidance=3.5, device=torch.device("cpu"), dtype=torch.bfloat16, seed=0
    )
    # Packed latents: 1024x1024 -> 64x64 patches of 2x2 VAE latents = 4096 tokens x 64 channels.
    assert tuple(inputs["hidden_states"].shape) == (1, 4096, 64)
    assert tuple(inputs["encoder_hidden_states"].shape) == (1, 512, 4096)
    assert tuple(inputs["pooled_projections"].shape) == (1, 768)
    for name in ("hidden_states", "encoder_hidden_states", "pooled_projections"):
        assert inputs[name].dtype == torch.bfloat16, name
    # The pipeline hands both models t / 1000 and guidance as (B,) tensors in the model dtype.
    assert inputs["timestep"].tolist() == [0.5] and inputs["timestep"].dtype == torch.bfloat16
    assert inputs["guidance"].tolist() == [3.5] and inputs["guidance"].dtype == torch.bfloat16
    assert tuple(inputs["txt_ids"].shape) == (512, 3) and not inputs["txt_ids"].any()
    assert tuple(inputs["img_ids"].shape) == (4096, 3)
    again = bench.bench_inputs(
        cfg, timestep=0.5, guidance=3.5, device=torch.device("cpu"), dtype=torch.bfloat16, seed=0
    )
    assert torch.equal(inputs["hidden_states"], again["hidden_states"])  # seeded


def test_image_ids_are_the_pipeline_s():
    from diffusers import FluxPipeline

    ids = bench.latent_image_ids(1024, 1024, device=torch.device("cpu"), dtype=torch.float32)
    expected = FluxPipeline._prepare_latent_image_ids(1, 64, 64, torch.device("cpu"), torch.float32)
    assert torch.equal(ids, expected)
    # Schnell-shaped config: no guidance -> an empty guidance tensor, as the adapter passes.
    inputs = bench.bench_inputs(
        _flux_dev_config(guidance_embeds=False),
        timestep=0.5,
        guidance=3.5,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        seed=0,
    )
    assert inputs["guidance"].numel() == 0
