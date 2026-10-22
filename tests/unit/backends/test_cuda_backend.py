"""Unit tests for the CUDA platform runtime.

Runs on a host without CUDA: the device calls are monkeypatched.
"""

import pytest
import torch

from difflet.platforms.cuda import CudaPlatform


def test_prepare_runtime_without_cuda_raises_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available"):
        CudaPlatform().prepare_runtime(parallel=object())


def _fake_distributed(monkeypatch, *, initialised: bool):
    """Stand in for torch.distributed so the torchrun path runs on a CPU host."""
    import atexit

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(("set_device", d)))
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **k: calls.append(("atexit", fn)))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialised)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda backend=None, **kw: calls.append(("init_process_group", backend)),
    )
    return calls


def test_prepare_runtime_multi_process_initialises_nccl_and_binds_local_rank(monkeypatch):
    calls = _fake_distributed(monkeypatch, initialised=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    CudaPlatform().prepare_runtime(parallel=object())
    # The device is bound before the NCCL communicator is created.
    assert calls[:2] == [("set_device", 1), ("init_process_group", "nccl")]


def test_prepare_runtime_destroys_the_process_group_it_created_at_exit(monkeypatch):
    calls = _fake_distributed(monkeypatch, initialised=False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    CudaPlatform().prepare_runtime(parallel=object())
    hooks = [fn for name, fn in calls if name == "atexit"]
    assert len(hooks) == 1
    (hook,) = hooks

    destroyed = []
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: destroyed.append(1))
    # Still initialised at exit: destroyed once, so the ranks exit cleanly.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    hook()
    assert destroyed == [1]
    # Someone already tore it down: the hook is a no-op rather than an error.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    hook()
    assert destroyed == [1]


def test_prepare_runtime_multi_process_reuses_an_existing_process_group(monkeypatch):
    calls = _fake_distributed(monkeypatch, initialised=True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    CudaPlatform().prepare_runtime(parallel=object())
    assert calls == [("set_device", 0)]


def test_prepare_runtime_single_process_sets_local_rank_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(d))

    def _no_init(*a, **kw):
        raise AssertionError("world size 1 must not initialise torch.distributed")

    monkeypatch.setattr(torch.distributed, "init_process_group", _no_init)
    CudaPlatform().prepare_runtime(parallel=object())
    assert calls == [0]
