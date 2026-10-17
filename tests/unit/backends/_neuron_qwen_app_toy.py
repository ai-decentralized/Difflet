"""The tiny Qwen-Image snapshot for the Q2 tests (the composed neuron application), and their gloo
workers (not collected by pytest).

``write_tiny_qwen_snapshot(root)`` writes a diffusers-layout snapshot around Q1's tiny DiT
(``_neuron_qwen_toy.write_tiny_qwen_model``: 2 blocks, joint_attention_dim 24, 64x96, 8 text
rows): ``scheduler/`` (Qwen-Image's own FlowMatch config, dynamic shifting on), ``vae/`` (a seeded
``AutoencoderKLQwenImage`` with z_dim 4 = the DiT's out_channels), and, when the real snapshot's
tokenizer is on this machine, ``tokenizer/`` (copied) and ``text_encoder/`` (a seeded 2-layer
``Qwen2_5_VLForConditionalGeneration`` with hidden size 24 and Qwen's full vocabulary, so the
real token ids embed). Without the tokenizer the prompt tests skip and the rest still run.

The stub encoder (``StubQwenTokenizer``/``StubQwenVL``) stands in for Qwen2.5-VL on rank 0 in the
broadcast and post-processing tests: its tokenizer pads ``34 + n`` valid ids to ``max_length``
on the right, and its "hidden state" is ``stub_hidden(ids)``, a ramp plus the token id, so a row
in the wrong place, a missed drop or a missed zero-pad changes the result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.unit.backends._neuron_qwen_toy import (
    QWEN_SNAPSHOT,
    TINY_QWEN_TRANSFORMER,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    write_tiny_qwen_model,
)

TEXT_DIM = TINY_QWEN_TRANSFORMER["joint_attention_dim"]
#: z_dim of the tiny VAE: the DiT's out_channels (the packed latent has 4 * z_dim channels).
TINY_Z_DIM = TINY_QWEN_TRANSFORMER["out_channels"]
TINY_PACKED_LATENT_SHAPE = (1, 24, 4 * TINY_Z_DIM)
TINY_IMAGE_SHAPE = (1, 3, TINY_SHAPE["height"], TINY_SHAPE["width"])
#: Qwen-Image's scheduler/scheduler_config.json, verbatim.
QWEN_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.34.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
TINY_VAE_CONFIG = {
    "base_dim": 8,
    "z_dim": TINY_Z_DIM,
    "dim_mult": [1, 2, 2, 2],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "latents_mean": [0.1, -0.2, 0.3, -0.4],
    "latents_std": [1.5, 0.8, 1.2, 2.0],
}
#: Fixed text embeddings for the runs that do not encode: 5 valid rows, 3 zero rows.
TINY_VALID_ROWS = 5
TINY_PROMPT = "a cat"  # 2 tokens + the template's 5 closing ones: 7 of the 8 text rows


def _perturbed(model, seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.05 * torch.randn_like(param))
    return model.eval().requires_grad_(False)


def tokenizer_available() -> bool:
    return (QWEN_SNAPSHOT / "tokenizer" / "vocab.json").is_file()


def tiny_text_encoder_config():
    from transformers import Qwen2_5_VLConfig

    return Qwen2_5_VLConfig(
        text_config=dict(
            hidden_size=TEXT_DIM, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, vocab_size=151936,
            max_position_embeddings=512, rope_theta=1000000.0, rms_norm_eps=1e-6,
            rope_scaling={"type": "default", "rope_type": "default", "mrope_section": [2, 2, 2]},
            bos_token_id=151643, eos_token_id=151645,
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=16, num_heads=2, out_hidden_size=TEXT_DIM,
            fullatt_block_indexes=[0], window_size=28, patch_size=14, spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
    )


def write_tiny_qwen_snapshot(root, *, text_encoder: bool = True) -> torch.Tensor:
    """Q1's tiny transformer plus scheduler, VAE and (if available) tokenizer + text encoder;
    returns Q1's diffusers DiT reference."""
    root = Path(root)
    ref = write_tiny_qwen_model(root)
    (root / "scheduler").mkdir(parents=True, exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(QWEN_SCHEDULER_CONFIG, indent=2) + "\n"
    )
    from diffusers import AutoencoderKLQwenImage

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2)
        vae = AutoencoderKLQwenImage(**TINY_VAE_CONFIG)
    _perturbed(vae, 3).save_pretrained(str(root / "vae"), safe_serialization=True)
    if text_encoder and tokenizer_available():
        from transformers import Qwen2_5_VLForConditionalGeneration

        shutil.copytree(QWEN_SNAPSHOT / "tokenizer", root / "tokenizer", dirs_exist_ok=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            encoder = Qwen2_5_VLForConditionalGeneration(tiny_text_encoder_config())
        _perturbed(encoder, 5).save_pretrained(str(root / "text_encoder"),
                                               safe_serialization=True)
    return ref


def tiny_text_inputs():
    """Zero-padded embeddings ``[1, 8, 24]`` (5 valid rows) and their bool mask."""
    gen = torch.Generator().manual_seed(6)
    embeds = torch.zeros(1, TINY_TEXT_SEQ_LEN, TEXT_DIM)
    embeds[:, :TINY_VALID_ROWS] = torch.randn(1, TINY_VALID_ROWS, TEXT_DIM, generator=gen)
    mask = torch.zeros(1, TINY_TEXT_SEQ_LEN, dtype=torch.bool)
    mask[:, :TINY_VALID_ROWS] = True
    return embeds.to(torch.bfloat16), mask


# --------------------------------------------------------------------------
# the stub Qwen2.5-VL
# --------------------------------------------------------------------------


def stub_hidden(input_ids: torch.Tensor, text_dim: int = TEXT_DIM) -> torch.Tensor:
    """The stub encoder's last hidden state for ``input_ids``: a ramp plus the token id."""
    batch, length = input_ids.shape
    ramp = torch.arange(batch * length * text_dim, dtype=torch.float32)
    return ramp.reshape(batch, length, text_dim) / 7 + input_ids[..., None].float()


class StubQwenTokenizer:
    """``34 + n_prompt`` valid ids (the ids are 100 + position), right-padded with 0 to
    ``max_length`` and truncated to it; records each call's text and keyword arguments."""

    def __init__(self, n_prompt: int = 3):
        self.n_prompt = n_prompt
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, dict(kwargs)))
        length = 34 + self.n_prompt
        ids = torch.arange(100, 100 + length)[None]
        mask = torch.ones_like(ids)
        if kwargs.get("padding") == "max_length":
            max_length = int(kwargs["max_length"])
            if kwargs.get("truncation"):
                ids, mask = ids[:, :max_length], mask[:, :max_length]
            pad = max_length - ids.shape[1]
            ids = torch.cat([ids, torch.zeros(1, pad, dtype=ids.dtype)], 1)
            mask = torch.cat([mask, torch.zeros(1, pad, dtype=mask.dtype)], 1)
        if kwargs.get("return_tensors") != "pt":
            return SimpleNamespace(input_ids=ids[0].tolist(), attention_mask=mask[0].tolist())
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class StubQwenVL:
    """``Qwen2_5_VLForConditionalGeneration`` as the encoder uses it: ``.model(...)`` returns
    ``last_hidden_state = stub_hidden(input_ids)``; records the shape and thread count, or fails
    on demand."""

    def __init__(self, text_dim: int = TEXT_DIM):
        self.text_dim = text_dim
        self.fail = False
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.model = self._forward

    def _forward(self, *, input_ids, attention_mask, use_cache):
        assert use_cache is False and attention_mask.shape == input_ids.shape
        self.calls.append((tuple(input_ids.shape), torch.get_num_threads()))
        if self.fail:
            raise ValueError("fake Qwen2.5-VL failure")
        return SimpleNamespace(last_hidden_state=stub_hidden(input_ids, self.text_dim))


# --------------------------------------------------------------------------
# gloo workers: fn(rank, world_size, *args) -> plain Python data
# --------------------------------------------------------------------------


def _attempt(fn):
    from difflet.backends.neuron.core.distributed import RankFailureError

    try:
        out = fn()
    except RankFailureError as exc:
        return {"outcome": f"RankFailureError: {exc}"}
    except Exception as exc:  # noqa: BLE001 - reported to the parent by type and message
        return {"outcome": f"{type(exc).__name__}: {exc}"}
    if out is None:
        return {"outcome": "ok"}
    embeds, mask = out
    return {"outcome": "ok", "dtype": str(embeds.dtype), "embeds": embeds.float().tolist(),
            "mask": mask.tolist(), "mask_dtype": str(mask.dtype)}


def q2_host_text_encoder_worker(rank: int, world_size: int) -> dict:
    """The rank-0 host encoder on gloo with the stub: a broadcast result, a failing encode, a
    recovery, and a failing load (no such model directory)."""
    from difflet.models.qwen_image.neuron_application import _HostQwenTextEncoder

    def encoder():
        return _HostQwenTextEncoder(
            "/nonexistent-q2-model", text_dim=TEXT_DIM, seq_len=TINY_TEXT_SEQ_LEN,
            dtype=torch.bfloat16, device="cpu", threads=2,
        )

    stub_tok, stub_vl = StubQwenTokenizer(n_prompt=3), StubQwenVL()
    host = encoder()
    if rank == 0:  # what load() leaves on rank 0
        host.tokenizer, host.model = stub_tok, stub_vl
    threads_before = torch.get_num_threads()
    result = {"ok": _attempt(lambda: host("a red fox"))}
    result["threads_restored"] = torch.get_num_threads() == threads_before
    stub_vl.fail = True
    result["failed"] = _attempt(lambda: host("a red fox"))
    stub_vl.fail = False
    result["recovered"] = _attempt(lambda: host("a red fox"))
    result["encoder_calls"] = stub_vl.calls
    result["tokenizer_calls"] = [
        (text, {k: v for k, v in kwargs.items()}) for text, kwargs in stub_tok.calls
    ]
    missing = encoder()
    result["load"] = _attempt(missing.load)
    result["loaded"] = missing.model is not None
    return result


def q2_pipeline_worker(
    rank: int,
    world_size: int,
    model_dir: str,
    tp: int,
    cp_mode: str,
    exec_mode: str,
    use_prompt: bool,
    output_type: str = "latent",
    sp: bool = False,
) -> dict:
    """The composed application at TP=tp x CP=world/tp on CPU fp32 (Megatron SP over the TP
    group when ``sp``): two steps from seed 7, on a prompt (the tiny Qwen2.5-VL, encoded on
    rank 0) or on the fixed embeddings."""
    import hashlib

    from difflet.models.qwen_image.neuron_application import TorchNeuronQwenImageApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    cp = world_size // tp
    app = TorchNeuronQwenImageApplication(
        model_path=model_dir,
        parallel=DiffletParallelConfig(tp_degree=tp, cp_degree=cp,
                                       cp_mode=cp_mode if cp > 1 else "gather_kv",
                                       sp_enabled=sp),
        dtype=torch.float32,
        shape=dict(TINY_SHAPE),
        text_seq_len=TINY_TEXT_SEQ_LEN,
        exec_mode=exec_mode,
        device="cpu",
        encoder_threads=2,
        enable_text_encoder=use_prompt,
        enable_vae_decoder=output_type == "pt",
    )
    app.load()
    steps: list[int] = []
    forward_seconds: list[float] = []
    app.step_hook = lambda: steps.append(len(forward_seconds))
    app.transformer_adapter.forward_hook = forward_seconds.append
    kwargs = dict(num_inference_steps=2, guidance_scale=4.0, seed=7, channels=TINY_Z_DIM,
                  output_type=output_type)
    if use_prompt:
        out = app(prompt=TINY_PROMPT, **kwargs)
        embeds, mask = app.encode_prompt(TINY_PROMPT)
    else:
        embeds, mask = tiny_text_inputs()
        out = app(encoder_hidden_states=embeds, encoder_hidden_states_mask=mask, **kwargs)
    final = out.latents
    result = {
        "dtype": str(final.dtype),
        "latents": final.tolist(),
        "sha256": hashlib.sha256(final.contiguous().numpy().tobytes()).hexdigest(),
        "steps": steps,
        "forwards": len(forward_seconds),
        "timesteps": [float(t) for t in app.pipeline.scheduler.timesteps],
        "embeds": embeds.float().tolist(),
        "mask": mask.tolist(),
        "phase_seconds": sorted(app.phase_seconds),
        "unwarmed_shapes": [list(shape) for shape in app.unwarmed_shapes],
        "compiled_blocks": app.compiled_blocks,
    }
    if output_type == "pt":
        result["images"] = out.images.tolist()
    return result


def q2_difflet_pipeline_worker(rank: int, world_size: int, model_dir: str, cache_dir: str) -> dict:
    """``DiffletPipeline.from_pretrained(..., backend="neuron")`` on the tiny snapshot (no AoT
    compile, one rank), then a two-step call through the pipeline."""
    from difflet.pipeline.difflet_pipeline import DiffletPipeline
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    pipe = DiffletPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        parallel=DiffletParallelConfig(tp_degree=world_size),
        dtype=torch.float32,
        height=TINY_SHAPE["height"],
        width=TINY_SHAPE["width"],
        backend="neuron",
        application_kwargs={"device": "cpu", "exec_mode": "eager",
                            "text_seq_len": TINY_TEXT_SEQ_LEN, "enable_text_encoder": False},
        model_path_override=model_dir,
        resolved_source_id="tiny",
        compiled_path_override=cache_dir,
    )
    embeds, mask = tiny_text_inputs()
    out = pipe(encoder_hidden_states=embeds, encoder_hidden_states_mask=mask,
               num_inference_steps=2, seed=7, channels=TINY_Z_DIM)
    return {
        "app": type(pipe.app).__name__,
        "backend": pipe.backend.name,
        "compiled_dir_exists": Path(cache_dir).exists(),
        "latents": out.latents.tolist(),
    }
