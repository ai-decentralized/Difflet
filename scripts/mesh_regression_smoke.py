#!/usr/bin/env python
"""On-device parallel-mesh bit-identity regression: single-run runner.

Runs ONE model in ONE parallel mode on a fixed-seed input and saves the output
tensor(s) to ``--out``. Covered (model, mode) pairs — every parallel axis the
mesh refactor touched, per supported model:

  * wan     cfg (tp=2 x cfg-parallel, batch=2) and cp (tp=2 x cp=2)
  * flux    cp  (tp=2 x cp=2)   — CP paths changed; CFG stays CLI-blocked
  * hunyuan cp  (tp=2 x cp=2)
  * qwen    cp  (tp=2 x cp=2)
  * ltx2    cfg (tp=2 x cfg-parallel, batch=2, video+audio outputs)

The driver (``scripts/mesh_regression_smoke.sh``) runs each pair twice — once
with PYTHONPATH at the pre-refactor baseline commit, once on the refactor
branch — and byte-compares the outputs (PSNR must be inf). Each run happens in
its own process because NxD parallel_state / the process-group layer
initialize once per process. HunyuanVideo-1.5 has no CFG/CP wiring (both
rejected at entry), so it gets a plain tp-only compile/forward smoke via
``scripts/hunyuan15_tiny_compile_smoke.py`` instead of a baseline compare.

Real weights are not required: ``--make-checkpoint`` synthesizes a tiny seeded
diffusers checkpoint (save_pretrained format) per model that both revisions
load identically — bit-identity needs identical weights, not real ones. The
runner imports difflet lazily so the SAME script file drives both the baseline
and refactor checkouts via PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os

import torch

TP_DEGREE = 2
WORLD_SIZE = 4
WEIGHT_SEED = 0
INPUT_SEED = 1234


# --------------------------------------------------------------------------- #
# tiny seeded checkpoints (diffusers save_pretrained format, one per model)
# --------------------------------------------------------------------------- #

def _save(model, ckpt_dir: str) -> None:
    subdir = os.path.join(ckpt_dir, "transformer")
    model.save_pretrained(subdir, safe_serialization=True)
    print(f"[mesh-regression] tiny checkpoint (seed {WEIGHT_SEED}) -> {subdir}")


def make_wan_checkpoint(ckpt_dir: str) -> None:
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    torch.manual_seed(WEIGHT_SEED)
    _save(
        WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=4,   # divisible by tp=2
            attention_head_dim=64,
            in_channels=16,
            out_channels=16,
            text_dim=64,
            freq_dim=256,
            ffn_dim=256,
            num_layers=2,
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=1024,
        ),
        ckpt_dir,
    )


def make_flux_checkpoint(ckpt_dir: str) -> None:
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    torch.manual_seed(WEIGHT_SEED)
    _save(
        FluxTransformer2DModel(
            patch_size=1,
            in_channels=64,
            num_layers=2,
            num_single_layers=2,
            # The Difflet Flux wrapper hardcodes FluxPosEmbed(axes_dim=(16,56,56)),
            # i.e. a 128-dim rope — so head_dim must stay at the real model's 128.
            attention_head_dim=128,
            num_attention_heads=4,
            joint_attention_dim=64,
            pooled_projection_dim=32,
            guidance_embeds=True,
            axes_dims_rope=(16, 56, 56),
        ),
        ckpt_dir,
    )


def make_hunyuan_checkpoint(ckpt_dir: str) -> None:
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel,
    )

    torch.manual_seed(WEIGHT_SEED)
    _save(
        HunyuanVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            num_attention_heads=4,
            # attention_cte dual-stream kernels are tuned for the real model's
            # head_dim=128 geometry; smaller head dims trip DMA bounds checks.
            attention_head_dim=128,
            num_layers=1,
            num_single_layers=2,
            num_refiner_layers=1,
            mlp_ratio=2.0,
            patch_size=2,
            patch_size_t=1,
            qk_norm="rms_norm",
            guidance_embeds=True,
            text_embed_dim=64,
            pooled_projection_dim=32,
            rope_theta=256.0,
            rope_axes_dim=(16, 56, 56),  # sums to attention_head_dim
        ),
        ckpt_dir,
    )


def make_qwen_checkpoint(ckpt_dir: str) -> None:
    from diffusers.models.transformers.transformer_qwenimage import (
        QwenImageTransformer2DModel,
    )

    torch.manual_seed(WEIGHT_SEED)
    _save(
        QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=64,
            out_channels=16,
            num_layers=2,
            attention_head_dim=64,
            num_attention_heads=4,
            joint_attention_dim=64,
            guidance_embeds=False,
            axes_dims_rope=(8, 28, 28),
        ),
        ckpt_dir,
    )


def make_ltx2_checkpoint(ckpt_dir: str) -> None:
    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    torch.manual_seed(WEIGHT_SEED)
    _save(
        LTX2VideoTransformer3DModel(
            in_channels=8,
            out_channels=8,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=2,
            attention_head_dim=4,
            cross_attention_dim=8,
            vae_scale_factors=(8, 32, 32),
            pos_embed_max_pos=20,
            base_height=2048,
            base_width=2048,
            audio_in_channels=8,
            audio_out_channels=8,
            audio_patch_size=16,
            audio_patch_size_t=1,
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_cross_attention_dim=8,
            audio_scale_factor=4,
            audio_pos_embed_max_pos=20,
            audio_sampling_rate=16000,
            audio_hop_length=160,
            num_layers=1,
            caption_channels=8,
            rope_double_precision=False,
            use_prompt_embeddings=True,
        ),
        ckpt_dir,
    )


MAKERS = {
    "wan": make_wan_checkpoint,
    "flux": make_flux_checkpoint,
    "hunyuan": make_hunyuan_checkpoint,
    "qwen": make_qwen_checkpoint,
    "ltx2": make_ltx2_checkpoint,
}


# --------------------------------------------------------------------------- #
# per-model device runs (compile -> load -> fixed-seed forward -> save)
# --------------------------------------------------------------------------- #

def run_wan(mode: str, ckpt_dir: str, out_path: str, work_dir: str) -> None:
    from difflet.models.wan.application import create_wan_backbone_config
    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication

    cfg_parallel = mode == "cfg"
    batch = 2 if cfg_parallel else 1
    height, width, latent_frames, text_seq_len, text_dim = 128, 128, 1, 512, 64

    config = create_wan_backbone_config(
        model_path=ckpt_dir,
        world_size=WORLD_SIZE,
        tp_degree=TP_DEGREE,
        dtype=torch.bfloat16,
        height=height,
        width=width,
        num_frames=latent_frames,
        batch_size=batch,
        context_parallel_enabled=(mode == "cp"),
        cp_mode="gather_kv",
        cfg_parallel_enabled=cfg_parallel,
    )

    app = NeuronWanBackboneApplication(
        model_path=os.path.join(ckpt_dir, "transformer"), config=config
    )
    _compile_load(app, work_dir, mode)

    torch.manual_seed(INPUT_SEED)
    hidden = torch.randn(
        [batch, 16, latent_frames, height // 8, width // 8], dtype=torch.bfloat16
    )
    timestep = torch.randn([batch], dtype=torch.bfloat16)
    encoder = torch.randn([batch, text_seq_len, text_dim], dtype=torch.bfloat16)
    with torch.no_grad():
        out = app.forward(hidden, timestep, encoder)
    _save_out(out, out_path)


def run_flux(mode: str, ckpt_dir: str, out_path: str, work_dir: str) -> None:
    assert mode == "cp"
    from difflet.models.flux.modeling_flux import (
        FluxBackboneInferenceConfig,
        NeuronFluxBackboneApplication,
    )
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.utils.diffusers_adapter import load_diffusers_config

    height, width, text_seq_len = 256, 256, 512
    component_dir = os.path.join(ckpt_dir, "transformer")
    neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        world_size=WORLD_SIZE,
        torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    config = FluxBackboneInferenceConfig(
        context_parallel_enabled=True,
        cp_mode="gather_kv",
        neuron_config=neuron_config,
        load_config=load_diffusers_config(component_dir),
        height=height,
        width=width,
    )
    config.vae_scale_factor = 8

    num_patches = height * width // ((2 * 8) ** 2)  # 256

    app = NeuronFluxBackboneApplication(model_path=component_dir, config=config)
    _compile_load(app, work_dir, mode)

    torch.manual_seed(INPUT_SEED)
    hidden_states = torch.randn([1, num_patches, config.in_channels], dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(
        [1, text_seq_len, config.joint_attention_dim], dtype=torch.bfloat16
    )
    pooled_projections = torch.randn([1, config.pooled_projection_dim], dtype=torch.bfloat16)
    timestep = torch.randn([1], dtype=torch.bfloat16)
    guidance = torch.randn([1], dtype=torch.bfloat16)
    img_ids = torch.zeros([num_patches, 3], dtype=torch.bfloat16)
    txt_ids = torch.zeros([text_seq_len, 3], dtype=torch.bfloat16)
    with torch.no_grad():
        out = app.forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            guidance=guidance,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )
    _save_out(out, out_path)


def run_hunyuan(mode: str, ckpt_dir: str, out_path: str, work_dir: str) -> None:
    assert mode == "cp"
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
        NeuronHunyuanVideoBackboneApplication,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    # 256x256 @ 9 frames -> latent video seq 3*16*16 = 768, per-rank 384 (3x128):
    # keeps the sequence shard a multiple of the attention kernel's 128 tile.
    height, width, num_frames, text_seq_len = 256, 256, 9, 256
    component_dir = os.path.join(ckpt_dir, "transformer")
    neuron_config = NeuronConfig(
        batch_size=1,
        tp_degree=TP_DEGREE,
        world_size=WORLD_SIZE,
        torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    config = HunyuanVideoBackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(component_dir),
        height=height,
        width=width,
        num_frames=num_frames,
        text_seq_len=text_seq_len,
        context_parallel_enabled=True,
        cp_mode="gather_kv",
    )

    app = NeuronHunyuanVideoBackboneApplication(model_path=component_dir, config=config)
    _compile_load(app, work_dir, mode)

    torch.manual_seed(INPUT_SEED)
    hidden = torch.randn(
        [1, 16, config.latent_frames, config.latent_height, config.latent_width],
        dtype=torch.bfloat16,
    )
    timestep = torch.ones([1], dtype=torch.bfloat16)
    encoder = torch.randn([1, text_seq_len, config.text_embed_dim], dtype=torch.bfloat16)
    mask = torch.ones([1, text_seq_len], dtype=torch.int64)
    pooled = torch.randn([1, config.pooled_projection_dim], dtype=torch.bfloat16)
    guidance = torch.ones([1], dtype=torch.bfloat16)
    with torch.no_grad():
        out = app.forward(hidden, timestep, encoder, mask, pooled, guidance)
    _save_out(out, out_path)


def run_qwen(mode: str, ckpt_dir: str, out_path: str, work_dir: str) -> None:
    assert mode == "cp"
    from difflet.models.qwen_image.application import create_qwen_image_transformer_config
    from difflet.backends.trainium.qwen_image.transformer import (
        NeuronQwenImageTransformerApplication,
    )

    height, width, text_seq_len = 256, 256, 256
    image_seq_len = (height // 16) * (width // 16)  # 256

    config = create_qwen_image_transformer_config(
        model_path=ckpt_dir,
        world_size=WORLD_SIZE,
        tp_degree=TP_DEGREE,
        dtype=torch.bfloat16,
        height=height,
        width=width,
        text_seq_len=text_seq_len,
        batch_size=1,
        context_parallel_enabled=True,
        cp_mode="gather_kv",
    )

    app = NeuronQwenImageTransformerApplication(
        model_path=os.path.join(ckpt_dir, "transformer"), config=config
    )
    _compile_load(app, work_dir, mode)

    torch.manual_seed(INPUT_SEED)
    hidden = torch.randn([1, image_seq_len, int(config.in_channels)], dtype=torch.bfloat16)
    timestep = torch.ones([1], dtype=torch.bfloat16)
    encoder = torch.randn(
        [1, text_seq_len, int(config.joint_attention_dim)], dtype=torch.bfloat16
    )
    mask = torch.ones([1, text_seq_len], dtype=torch.bool)
    guidance = torch.full([1], 4.0, dtype=torch.bfloat16)
    with torch.no_grad():
        out = app.models[0](hidden, timestep, encoder, mask, guidance)
    _save_out(out, out_path)


def run_ltx2(mode: str, ckpt_dir: str, out_path: str, work_dir: str) -> None:
    assert mode == "cfg"
    from difflet import DiffletParallelConfig, DiffletPipeline

    pipe = DiffletPipeline.from_pretrained(
        ckpt_dir,
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=TP_DEGREE, cfg_parallel_enabled=True),
        dtype=torch.bfloat16,
        height=64,
        width=64,
        num_frames=9,
        compile_cache_dir=os.path.join(work_dir, "compile_cfg"),
        load=True,
        skip_warmup=True,
        application_kwargs={"text_seq_len": 4, "audio_num_frames": 2},
    )

    torch.manual_seed(INPUT_SEED)
    bundle = _ltx2_bundle(pipe)
    with torch.no_grad():
        video_out, audio_out = pipe(bundle)
    out = {
        "video": video_out.detach().to(torch.float32).cpu(),
        "audio": audio_out.detach().to(torch.float32).cpu(),
    }
    torch.save(out, out_path)
    print(
        f"[mesh-regression] saved output video={tuple(out['video'].shape)} "
        f"audio={tuple(out['audio'].shape)} -> {out_path}"
    )


def _ltx2_bundle(pipe):
    from difflet.models.ltx_2.application import LTX2DiTInputBundle
    from difflet.models.ltx_2.pipeline import (
        make_ltx_2_audio_coords,
        make_ltx_2_video_coords,
    )

    contract = pipe.app.dit_input_contract()
    cfg = pipe.app.transformer.config

    def rand(name):
        return torch.randn(contract[name]["shape"], dtype=torch.bfloat16)

    return LTX2DiTInputBundle(
        hidden_states=rand("hidden_states"),
        audio_hidden_states=rand("audio_hidden_states"),
        encoder_hidden_states=rand("encoder_hidden_states"),
        audio_encoder_hidden_states=rand("audio_encoder_hidden_states"),
        timestep=torch.ones(contract["timestep"]["shape"], dtype=torch.bfloat16),
        sigma=torch.ones(contract["sigma"]["shape"], dtype=torch.bfloat16),
        encoder_attention_mask=torch.ones(
            contract["encoder_attention_mask"]["shape"], dtype=torch.bool
        ),
        audio_encoder_attention_mask=torch.ones(
            contract["audio_encoder_attention_mask"]["shape"], dtype=torch.bool
        ),
        video_coords=make_ltx_2_video_coords(
            batch_size=contract["video_coords"]["shape"][0],
            num_frames=int(cfg.latent_num_frames),
            height=int(cfg.latent_height),
            width=int(cfg.latent_width),
            device="cpu",
            patch_size=int(cfg.patch_size),
            patch_size_t=int(cfg.patch_size_t),
            scale_factors=tuple(cfg.vae_scale_factors),
            causal_offset=int(getattr(cfg, "causal_offset", 1)),
            fps=float(getattr(cfg, "frame_rate", 24.0)),
        ),
        audio_coords=make_ltx_2_audio_coords(
            batch_size=contract["audio_coords"]["shape"][0],
            audio_num_frames=int(cfg.audio_num_frames),
            device="cpu",
            patch_size_t=int(cfg.audio_patch_size_t),
            scale_factor=int(cfg.audio_scale_factor),
            causal_offset=int(getattr(cfg, "causal_offset", 1)),
            sampling_rate=int(cfg.audio_sampling_rate),
            hop_length=int(cfg.audio_hop_length),
        ),
    )


def _compile_load(app, work_dir: str, mode: str) -> None:
    compile_dir = os.path.join(work_dir, f"compile_{mode}")
    os.makedirs(compile_dir, exist_ok=True)
    print(f"[mesh-regression] compiling -> {compile_dir}")
    app.compile(compile_dir, debug=False)
    print("[mesh-regression] loading weights to device")
    app.load(compile_dir)


def _save_out(out, out_path: str) -> None:
    def to_cpu(t):
        return t.detach().to(torch.float32).cpu()

    if isinstance(out, dict):
        out = {str(k): to_cpu(v) for k, v in out.items()}
        shapes = {k: tuple(v.shape) for k, v in out.items()}
    elif isinstance(out, (tuple, list)):
        out = {str(i): to_cpu(v) for i, v in enumerate(out)}
        shapes = {k: tuple(v.shape) for k, v in out.items()}
    else:
        out = to_cpu(out)
        shapes = tuple(out.shape)
    torch.save(out, out_path)
    print(f"[mesh-regression] saved output {shapes} -> {out_path}")


RUNNERS = {
    "wan": run_wan,
    "flux": run_flux,
    "hunyuan": run_hunyuan,
    "qwen": run_qwen,
    "ltx2": run_ltx2,
}


def main() -> None:
    p = argparse.ArgumentParser(description="parallel-mesh bit-identity single-run runner")
    p.add_argument("--model", choices=sorted(RUNNERS), default="wan")
    p.add_argument("--make-checkpoint", default=None, metavar="DIR",
                   help="synthesize the model's tiny seeded checkpoint into DIR and exit")
    p.add_argument("--mode", choices=["cfg", "cp"], default=None)
    p.add_argument("--ckpt", default=None, help="checkpoint dir from --make-checkpoint")
    p.add_argument("--out", default=None, help="path to save the output tensor(s) (.pt)")
    p.add_argument("--work-dir", default="/tmp/mesh_regression")
    args = p.parse_args()

    if args.make_checkpoint:
        MAKERS[args.model](args.make_checkpoint)
        return
    if not (args.mode and args.ckpt and args.out):
        p.error("--mode, --ckpt and --out are required unless --make-checkpoint is given")
    print(f"[mesh-regression] model={args.model} mode={args.mode} tp={TP_DEGREE} world={WORLD_SIZE}")
    RUNNERS[args.model](args.mode, args.ckpt, args.out, args.work_dir)


if __name__ == "__main__":
    main()
