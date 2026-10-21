"""CPU-only coverage for the Qwen-Image GPU application and its registry routing.

No weights, no GPU: an empty model dir builds an application with no transformer,
which is enough to prove the routing, the reflective contract and the prompt
padding helper. The GPU round trip is scripts/cuda_smoke.py.
"""

import inspect
import os

import pytest
import torch

from difflet.backends.gpu.application import create_gpu_application
from difflet.models.qwen_image.entry import create_qwen_image_application
from difflet.models.qwen_image.prompt import pad_or_truncate_prompt_embeds
from difflet.pipeline.parallel_config import DiffletParallelConfig


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")


def _build(tmp_path, **kwargs):
    return create_gpu_application(
        "qwen_image",
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": 64, "width": 64},
        **kwargs,
    )


def test_registry_routes_cuda_to_the_gpu_application(tmp_path):
    from difflet.registry import resolve_model

    app = resolve_model("Qwen/Qwen-Image").create_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape={"height": 64, "width": 64},
        backend="cuda",
    )
    assert type(app).__name__ == "CudaQwenImageApplication"
    assert app.transformer is None  # empty model dir -> no transformer/config.json
    assert app.has_compiled_artifacts(str(tmp_path)) is False
    assert app.dtype is torch.bfloat16


def test_entry_error_lists_its_backends(tmp_path):
    with pytest.raises(NotImplementedError, match="trainium and tpu"):
        create_qwen_image_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="bf16",
            shape={"height": 64, "width": 64},
            backend="rocm",
        )


def test_gpu_application_rejects_cfg_parallel(tmp_path):
    with pytest.raises(NotImplementedError, match="--cfg-parallel does not apply to Qwen-Image"):
        create_gpu_application(
            "qwen_image",
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True),
            dtype="bf16",
            shape={"height": 64, "width": 64},
        )


def test_registry_lists_cuda_for_qwen_image():
    from difflet.registry import resolve_model

    entry = resolve_model("Qwen/Qwen-Image")
    assert "cuda" in entry.backends
    entry.require_backend("cuda")  # must not raise


def test_cuda_application_signatures_match_the_pipeline_contract(tmp_path):
    app = _build(tmp_path)
    assert "debug" in inspect.signature(app.compile).parameters
    load_params = inspect.signature(app.load).parameters
    for name in ("start_rank_id", "local_ranks_size", "skip_warmup"):
        assert name in load_params


def test_lifecycle_without_a_transformer_is_an_explicit_error(tmp_path):
    app = _build(tmp_path)
    with pytest.raises(NotImplementedError, match="transformer/config.json"):
        app.compile(str(tmp_path))
    with pytest.raises(NotImplementedError, match="transformer/config.json"):
        app.load(str(tmp_path))


def test_pad_or_truncate_prompt_embeds_pads_and_masks():
    embeds = torch.ones(1, 3, 4)
    padded, mask = pad_or_truncate_prompt_embeds(embeds, None, max_sequence_length=5)
    assert tuple(padded.shape) == (1, 5, 4) and tuple(mask.shape) == (1, 5)
    assert mask.dtype == torch.bool
    assert mask[0].tolist() == [True, True, True, False, False]
    assert torch.equal(padded[0, :3], embeds[0]) and torch.equal(padded[0, 3:], torch.zeros(2, 4))
    truncated, tmask = pad_or_truncate_prompt_embeds(embeds, None, max_sequence_length=2)
    assert tuple(truncated.shape) == (1, 2, 4) and tmask[0].tolist() == [True, True]


def test_tiny_qwen_dit_runs_through_the_cuda_ops_on_cpu(tmp_path):
    """Bind difflet.ops to the CUDA backend in a fresh process and run a tiny DiT on CPU
    tensors: config → meta build → safetensors load → compile → load → forward. This is
    the pre-GPU proof that the modeling code resolves and runs on the CUDA ops."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(f"""
        import json, os, torch
        from safetensors.torch import save_file
        root = {str(tmp_path)!r}
        tdir = os.path.join(root, "transformer"); os.makedirs(tdir)
        cfg = dict(patch_size=2, in_channels=16, out_channels=4, num_layers=1,
                   attention_head_dim=32, num_attention_heads=2, joint_attention_dim=64,
                   guidance_embeds=False, axes_dims_rope=[8, 12, 12])
        json.dump(cfg, open(os.path.join(tdir, "config.json"), "w"))
        from difflet.models.qwen_image.dit_config import QwenImageDiTConfig
        from difflet.models.qwen_image.modeling_qwen_image import _QwenImageTransformerTraceModule
        config = QwenImageDiTConfig.from_pretrained(tdir, height=64, width=64, text_seq_len=8,
                                                    torch_dtype=torch.float32)
        torch.manual_seed(0)
        full = _QwenImageTransformerTraceModule(config)
        # HF layout: the diffusers transformer's own keys; the loader strips "transformer."
        state = {{k[len("transformer."):]: v.contiguous() for k, v in full.state_dict().items()}}
        save_file(state, os.path.join(tdir, "model.safetensors"))
        from difflet.backends.cuda.qwen_image.application import CudaQwenImageApplication
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        app = CudaQwenImageApplication(model_path=root, parallel=DiffletParallelConfig(tp_degree=1),
                                       dtype="fp32", shape={{"height": 64, "width": 64}},
                                       text_seq_len=8, device="cpu", load_text_encoder=False, load_vae=False)
        cache = os.path.join(root, "cache")
        app.compile(cache)
        assert app.has_compiled_artifacts(cache)
        app.load(cache)
        c = app.dit_input_contract()
        from difflet.models.qwen_image.contract import QwenImageDiTInputBundle
        bundle = QwenImageDiTInputBundle(
            hidden_states=torch.randn(*c["hidden_states"]["shape"]),
            timestep=torch.full(c["timestep"]["shape"], 0.5),
            encoder_hidden_states=torch.randn(*c["encoder_hidden_states"]["shape"]),
            encoder_hidden_states_mask=torch.ones(*c["encoder_hidden_states_mask"]["shape"], dtype=torch.bool),
            guidance=torch.zeros(c["guidance"]["shape"]))
        out = app(bundle)
        out = out[0] if isinstance(out, (tuple, list)) else out
        assert tuple(out.shape) == (1, config.image_seq_len, config.out_channels * 4) or tuple(out.shape)[:2] == (1, config.image_seq_len), out.shape
        assert torch.isfinite(out).all()
        import difflet.ops.attention as A
        print("OK", tuple(out.shape), A.attention.__module__ if hasattr(A.attention, "__module__") else "")
        """)
    env = dict(os.environ, DIFFLET_BACKEND="cuda")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "OK" in proc.stdout, proc.stdout


def test_dynamic_shift_mu_matches_the_cli_formula():
    # difflet/cli/orchestrators/qwen_image.py computes mu from the scheduler config; the
    # GPU path must use the same value or its timesteps differ from the Trainium path.
    from types import SimpleNamespace

    from difflet.backends.cuda.qwen_image.application import dynamic_shift_mu

    sc = SimpleNamespace(
        base_shift=0.5, max_shift=0.9, base_image_seq_len=256, max_image_seq_len=8192
    )
    slope = (0.9 - 0.5) / (8192 - 256)
    expected = 4096 * slope + (0.5 - slope * 256)
    assert dynamic_shift_mu(sc, 4096) == pytest.approx(expected)


def test_qwen_timesteps_pass_mu_to_a_dynamic_shift_scheduler():
    from difflet.backends.cuda.qwen_image.application import qwen_image_timesteps

    class _Sched:
        config = type(
            "C",
            (),
            {
                "base_shift": 0.5,
                "max_shift": 0.9,
                "base_image_seq_len": 256,
                "max_image_seq_len": 8192,
            },
        )()

        def set_timesteps(self, sigmas=None, mu=None, device=None, **kw):
            assert mu is not None and len(sigmas) == 4
            self.timesteps = torch.tensor(sigmas) * 1000

    sched = _Sched()
    ts = qwen_image_timesteps(sched, num_inference_steps=4, image_seq_len=4096)
    assert tuple(ts.shape) == (4,) and ts[0] == 1000.0


@pytest.mark.parametrize(
    "kwargs,owner",
    [
        ({"tp_degree": 2}, "not supported"),
        ({"tp_degree": 1, "cp_degree": 2}, "not supported"),
    ],
)
def test_gpu_application_rejects_multi_rank_parallel_up_front(tmp_path, kwargs, owner):
    # tp/cp/dp > 1 would otherwise fail only after loading the 41 GB DiT (at the first
    # collective). One application is one process, one device; refuse before any load.
    with pytest.raises(NotImplementedError, match=owner):
        create_gpu_application(
            "qwen_image",
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(**kwargs),
            dtype="bf16",
            shape={"height": 64, "width": 64},
        )


def test_generate_rejects_guidance_scale_when_the_model_has_no_guidance_embeds(tmp_path):
    # Qwen-Image's transformer has guidance_embeds=False: the guidance tensor is dropped by the
    # modeling code, so a user-supplied guidance_scale would be a silent no-op.
    import json

    tdir = tmp_path / "transformer"
    tdir.mkdir()
    (tdir / "config.json").write_text(
        json.dumps(
            {
                "patch_size": 2,
                "in_channels": 64,
                "num_layers": 1,
                "attention_head_dim": 128,
                "num_attention_heads": 24,
                "joint_attention_dim": 3584,
                "guidance_embeds": False,
                "axes_dims_rope": [16, 56, 56],
            }
        )
    )
    app = _build(tmp_path, device="cpu")
    with pytest.raises(ValueError, match="guidance_embeds"):
        app.generate("a cat", guidance_scale=1.0)


def test_generate_uses_the_given_initial_latents(tmp_path, monkeypatch):
    # The end-to-end oracle (scripts/cuda_e2e_vs_diffusers.py) feeds the same packed noise to
    # difflet and diffusers; generate() must use it verbatim.
    app = _build(tmp_path, device="cpu")
    seen = {}

    class _P:
        scheduler = None
        vae_scale_factor = 8

        def prepare_latents(self, **kw):
            seen.update(kw)
            return kw["latents"]

        def __call__(self, **kw):
            seen["call"] = kw
            return type("O", (), {"images": None})()

    app.pipeline = _P()
    app.vae = None
    monkeypatch.setattr(
        app, "encode_prompt", lambda p: (torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.bool))
    )
    noise = torch.randn(1, 16, 64)
    app.generate("a cat", num_inference_steps=2, seed=0, latents=noise, output_type="latent")
    assert seen["latents"] is noise and seen["call"]["latents"] is noise


def test_torch_compile_is_the_kwarg_and_off_by_default(tmp_path):
    assert _build(tmp_path).torch_compile is False
    assert _build(tmp_path, torch_compile=True).torch_compile is True
    assert _build(tmp_path, torch_compile=False).torch_compile is False


def test_cuda_pipeline_builds_a_teacache_controller_for_cadence(tmp_path):
    app = _build(tmp_path, load_text_encoder=False, load_vae=False, teacache_cadence=2)
    app.load_companions()
    controller = app.pipeline.teacache_controller
    assert controller is not None and int(controller.calibration.cadence) == 2


def test_cuda_pipeline_has_no_teacache_controller_by_default(tmp_path):
    app = _build(tmp_path, load_text_encoder=False, load_vae=False)
    app.load_companions()
    assert app.pipeline.teacache_controller is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"teacache_speedup": 1.5},
        {"teacache_calibration_path": "/x.json"},
        {"teacache_online_delta_alpha": 0.1},
    ],
)
def test_cuda_application_refuses_calibrated_and_online_delta_teacache(tmp_path, kwargs):
    with pytest.raises(NotImplementedError, match="not supported"):
        _build(tmp_path, **kwargs)
