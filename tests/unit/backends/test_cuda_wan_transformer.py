"""One Wan DiT expert as a CUDA component (CUDA plan F1, task 2).

Runs on the CPU (``device="cpu"``): the application's load path — meta build,
materialize, diffusers safetensors → ``load_checkpoint_into`` with the FFN rename,
strict — is device-independent. The diffusers ``WanTransformer3DModel`` is the
oracle; the Difflet config is read from the twin's saved ``config.json`` so the two
cannot describe different models.
"""

import datetime
import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import WanTransformer3DModel

from difflet.backends.cuda.ops_impl import parallel_mesh as pm
from difflet.backends.cuda.wan.transformer import CudaWanTransformerApplication
from difflet.models.wan.dit_config import WanDiTConfig
from difflet.pipeline.parallel_mesh import MeshSpec

TINY = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=8,
    in_channels=4,
    out_channels=4,
    text_dim=16,
    freq_dim=32,
    ffn_dim=32,
    num_layers=1,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
)
# 16x16 pixels, 5 frames -> a 2x2x2 latent (vae 8x spatial, 4x temporal).
GEOMETRY = dict(height=16, width=16, num_frames=5, text_seq_len=6)


@pytest.fixture(autouse=True)
def _clean_mesh():
    # The mesh is process-global and idempotent only for an equal spec: a cfg=2 test
    # after a tp=1 one (or before) must start from nothing.
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


def _reference(transformer_dir) -> WanTransformer3DModel:
    torch.manual_seed(0)
    reference = WanTransformer3DModel(**TINY).eval()
    reference.save_pretrained(str(transformer_dir))
    return reference


def _config(transformer_dir, **overrides) -> WanDiTConfig:
    return WanDiTConfig.from_pretrained(transformer_dir, **{**GEOMETRY, **overrides})


def _inputs(cfg: WanDiTConfig):
    torch.manual_seed(1)
    return (
        torch.randn(1, 4, cfg.latent_frames, cfg.latent_height, cfg.latent_width),
        torch.tensor([500.0]),
        torch.randn(1, 6, 16),
    )


def test_cuda_wan_expert_loads_diffusers_weights_and_matches(tmp_path):
    transformer = tmp_path / "transformer"
    reference = _reference(transformer)
    cfg = _config(transformer, torch_dtype=torch.float32)
    app = CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")
    cache = tmp_path / "cache"
    app.compile(str(cache))
    app.load(str(cache))  # warm-up runs the module on get_example_inputs()

    hidden_states, timestep, text = _inputs(cfg)
    with torch.no_grad():
        expected = reference(hidden_states, timestep, text, return_dict=False)[0]
        actual = app(hidden_states, timestep, text)
    assert actual.shape == expected.shape == (1, 4, 2, 2, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert all(not p.requires_grad for p in app.module.parameters())


def test_example_inputs_match_the_contract(tmp_path):
    transformer = tmp_path / "transformer"
    _reference(transformer)
    cfg = _config(transformer, batch_size=2)
    app = CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")
    hidden_states, timestep, text = app.get_example_inputs()
    assert hidden_states.shape == (2, 4, 2, 2, 2)
    assert timestep.shape == (2,)
    assert text.shape == (2, 6, 16)
    # bf16 by default: the example inputs are in the model dtype.
    assert {t.dtype for t in (hidden_states, timestep, text)} == {torch.bfloat16}


def test_forward_before_load_is_an_explicit_error(tmp_path):
    transformer = tmp_path / "transformer"
    _reference(transformer)
    cfg = _config(transformer, torch_dtype=torch.float32)
    app = CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        app(*_inputs(cfg))


@pytest.mark.parametrize(
    "overrides,follow_up",
    [
        ({"tp_degree": 2}, "follow-up F6"),
        ({"context_parallel_enabled": True}, "follow-up F2b"),
        ({"sp_enabled": True}, "follow-up F6"),
    ],
)
def test_parallel_modes_are_refused_up_front(tmp_path, overrides, follow_up):
    transformer = tmp_path / "transformer"
    _reference(transformer)
    cfg = _config(transformer, **overrides)
    with pytest.raises(NotImplementedError, match=follow_up):
        CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")


def test_cfg_parallel_builds_on_a_cfg_two_mesh(tmp_path, monkeypatch):
    """F2: CFG-parallel is accepted; the expert's mesh is cfg=2, tp=1.

    The CLI parent compiles before torchrun starts the ranks (WORLD_SIZE unset, no
    process group): the manifest it writes records the cfg=2 mesh the ranks load.
    """
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    transformer = tmp_path / "transformer"
    _reference(transformer)
    cfg = _config(transformer, cfg_parallel_enabled=True, batch_size=2)
    app = CudaWanTransformerApplication(model_path=str(transformer), config=cfg, device="cpu")
    assert app.config == MeshSpec(cfg=2, tp=1)
    hidden_states, timestep, text = app.get_example_inputs()
    assert hidden_states.shape[0] == timestep.shape[0] == text.shape[0] == 2
    cache = tmp_path / "cache"
    app.compile(str(cache))
    manifest = json.loads((cache / "cuda_manifest.json").read_text())
    assert manifest["mesh"] == {"dp": 1, "cfg": 2, "cp": 1, "tp": 1}
    assert manifest["world_size"] == 2


# ------------------------------------------------ CFG-parallel on two gloo ranks

CFG_WORLD = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _batch_two_inputs(cfg: WanDiTConfig):
    """[uncond, cond]: two different branches at one timestep, as the orchestrator stacks."""
    torch.manual_seed(2)
    return (
        torch.randn(2, 4, cfg.latent_frames, cfg.latent_height, cfg.latent_width),
        torch.tensor([500.0, 500.0]),
        torch.randn(2, 6, 16),
    )


def _cfg_worker(rank: int, port: int, transformer_dir: str, work_dir: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(CFG_WORLD),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group(
        "gloo", rank=rank, world_size=CFG_WORLD, timeout=datetime.timedelta(seconds=30)
    )
    try:
        from difflet.models.wan import modeling_wan

        cfg = _config(
            transformer_dir, torch_dtype=torch.float32, cfg_parallel_enabled=True, batch_size=2
        )
        app = CudaWanTransformerApplication(model_path=transformer_dir, config=cfg, device="cpu")
        cache = Path(work_dir) / f"cache-{rank}"
        app.compile(str(cache))
        app.load(str(cache), skip_warmup=False)  # the batch-2 warm-up runs the collective too

        # The rank must run its own branch only: record the batch the first block sees.
        block_batches = []
        app.module.blocks[0].register_forward_pre_hook(
            lambda _m, args: block_batches.append(int(args[0].shape[0]))
        )
        inputs = torch.load(Path(work_dir) / "inputs.pt")
        with torch.no_grad():
            out = app(*inputs)
        torch.save(
            {
                "out": out,
                "block_batches": block_batches,
                "gather_impl": modeling_wan.gather_from_tensor_model_parallel_region_with_dim.__module__,
                "cfg_group_is_process_group": isinstance(app.module.cfg_group, dist.ProcessGroup),
            },
            Path(work_dir) / f"rank-{rank}.pt",
        )
    finally:
        pm.destroy_parallel_mesh()
        dist.destroy_process_group()


@pytest.mark.multirank
def test_cfg_parallel_forward_matches_the_serial_batch(tmp_path, monkeypatch):
    """F2 task 3: two ranks, one CFG branch each, gathered back to the full batch.

    Each rank denoises its slice of [uncond, cond] at batch 1 and the all-gather hands
    both ranks the whole batch-2 output, equal to one process running batch 2.
    """
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with gloo is unavailable in this torch build")
    transformer = tmp_path / "transformer"
    _reference(transformer)  # saved before spawning: both ranks load the same weights

    serial_cfg = _config(transformer, torch_dtype=torch.float32, batch_size=2)
    serial = CudaWanTransformerApplication(
        model_path=str(transformer), config=serial_cfg, device="cpu"
    )
    serial.compile(str(tmp_path / "serial-cache"))
    serial.load(str(tmp_path / "serial-cache"))
    inputs = _batch_two_inputs(serial_cfg)
    with torch.no_grad():
        expected = serial(*inputs)
    assert expected.shape == (2, 4, 2, 2, 2)
    # The two branches differ, so a gather in the wrong order (or one branch twice) fails.
    assert not torch.allclose(expected[0], expected[1])
    torch.save(inputs, tmp_path / "inputs.pt")

    # The ranks bind difflet.ops to the CUDA backend (its gloo-capable collectives) on
    # their fresh import of the modeling; the CPU backend's ops are single-process.
    monkeypatch.setenv("DIFFLET_BACKEND", "cuda")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    # Spawned children unpickle _cfg_worker by this module's importlib name
    # (``unit.backends.test_cuda_wan_transformer``): put tests/ on their sys.path.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    mp.spawn(
        _cfg_worker,
        args=(_free_port(), str(transformer), str(tmp_path)),
        nprocs=CFG_WORLD,
        join=True,
    )
    assert not dist.is_initialized()  # the parent never joined a group

    for rank in range(CFG_WORLD):
        result = torch.load(tmp_path / f"rank-{rank}.pt")
        assert result["gather_impl"] == "difflet.backends.cuda.ops_impl.collectives"
        assert result["cfg_group_is_process_group"]
        assert result["block_batches"] == [1], result["block_batches"]
        assert result["out"].shape == (2, 4, 2, 2, 2)
        torch.testing.assert_close(result["out"], expected, rtol=1e-5, atol=1e-5)
