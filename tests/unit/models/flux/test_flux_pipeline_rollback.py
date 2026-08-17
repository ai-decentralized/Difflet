from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

import difflet.models.flux.pipeline as flux_pipeline
from difflet.pipeline.cache import (
    CacheRunner,
    CacheSession,
    LegacyResidualPredictor,
    PhasedStaticPolicy,
    TeaCacheControllerAdapter,
)


class _FakeTransformer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(in_channels=4, guidance_embeds=False)
        self.calls = 0

    def __call__(self, *, hidden_states, timestep, **kwargs):
        del kwargs
        self.calls += 1
        timestep = timestep.reshape(-1, 1, 1).to(hidden_states.dtype)
        return (hidden_states * 0.25 + timestep * 0.5,)


class _FakeScheduler:
    order = 1

    def __init__(self, num_steps: int) -> None:
        self.config = SimpleNamespace(
            base_image_seq_len=1,
            max_image_seq_len=4,
            base_shift=0.0,
            max_shift=1.0,
        )
        self.sigmas = tuple(float(index) for index in range(num_steps + 1))
        self._step_index = None
        self._begin_index = None
        self.physical_steps = 0

    def step(self, noise_pred, timestep, latents, *, return_dict):
        del timestep, return_dict
        if self._step_index is None:
            self._step_index = 0
            self._begin_index = 0
        else:
            self._step_index += 1
        self.physical_steps += 1
        return (latents - noise_pred * 0.1,)


class _FakeProgress:
    def __init__(self) -> None:
        self.updates = 0

    def update(self) -> None:
        self.updates += 1


class _FakeFluxHost:
    default_sample_size = 1
    vae_scale_factor = 1
    _execution_device = torch.device("cpu")
    interrupt = False

    def __init__(self, num_steps: int) -> None:
        self.transformer = _FakeTransformer()
        self.scheduler = _FakeScheduler(num_steps)
        self.teacache_probe = SimpleNamespace(teacache_probe_fused=False)
        self._progress = _FakeProgress()

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    def check_inputs(self, *args, **kwargs) -> None:
        del args, kwargs

    def encode_prompt(self, **kwargs):
        del kwargs
        return (
            torch.ones((1, 1, 1), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
        )

    def prepare_latents(self, *args):
        supplied = args[-1]
        latent = (
            torch.tensor([[[2.0]]], dtype=torch.float32)
            if supplied is None
            else supplied.detach().clone()
        )
        return latent, torch.zeros((1, 1), dtype=torch.float32)

    @contextmanager
    def progress_bar(self, *, total):
        del total
        yield self._progress

    def maybe_free_model_hooks(self) -> None:
        return None


def _controller(mask: tuple[bool, ...]) -> TeaCacheControllerAdapter:
    session = CacheSession(
        CacheRunner(PhasedStaticPolicy(mask), LegacyResidualPredictor()),
        num_steps=len(mask),
        configuration_source="flux-rollback-test",
    )
    return TeaCacheControllerAdapter(session)


def _run(host: _FakeFluxHost, controller: TeaCacheControllerAdapter) -> torch.Tensor:
    return flux_pipeline.NeuronFluxPipeline._call_with_teacache(
        host,
        prompt="test",
        height=1,
        width=1,
        num_inference_steps=5,
        output_type="latent",
        return_dict=False,
        cache_controller=controller,
    )[0]


def test_forced_rollback_dense_replay_is_bitwise_equal_to_dense_path(monkeypatch):
    timesteps = tuple(torch.tensor(value) for value in (1000, 800, 600, 400, 200))

    def fake_retrieve(scheduler, num_inference_steps, device, **kwargs):
        del device, kwargs
        scheduler._step_index = None
        scheduler._begin_index = None
        return timesteps, num_inference_steps

    monkeypatch.setattr(flux_pipeline, "retrieve_timesteps", fake_retrieve)
    monkeypatch.setattr(flux_pipeline, "calculate_shift", lambda *args: 0.0)
    # This is the zero-machine P1 acceptance test.  Import availability must
    # not turn its CPU tensors into a request for physical NeuronCore leases.
    monkeypatch.setattr(flux_pipeline, "XLA_AVAILABLE", False)

    dense_host = _FakeFluxHost(5)
    dense_controller = _controller((True, True, True, True, True))
    dense_output = _run(dense_host, dense_controller)

    rollback_host = _FakeFluxHost(5)
    rollback_host._cache_forced_rollback_plan = {
        "checkpoint_step": 2,
        "rollback_after_step": 3,
    }
    rollback_controller = _controller((True, True, False, True, True))
    rollback_output = _run(rollback_host, rollback_controller)

    assert torch.equal(rollback_output, dense_output)
    assert all(
        torch.equal(rollback, dense)
        for rollback, dense in zip(
            rollback_host._tc_last_trajectory,
            dense_host._tc_last_trajectory,
        )
    )
    assert len(rollback_host._tc_last_trajectory) == 5
    assert rollback_host.scheduler._step_index == dense_host.scheduler._step_index == 4
    assert rollback_host.scheduler.physical_steps == 7
    assert dense_host.scheduler.physical_steps == 5
    assert rollback_host.transformer.calls == 6
    assert dense_host.transformer.calls == 5
    assert rollback_controller.stats()["full_steps"] == 5
    assert rollback_controller.stats()["skipped_steps"] == 0
    assert rollback_host._progress.updates == 5
