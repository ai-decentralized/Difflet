"""Host-runnable H1c models for a resident FLUX cache-step data plane."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidentCacheStepState(nn.Module):
    def __init__(self, *, seq_len: int, channels: int, dtype: torch.dtype) -> None:
        super().__init__()
        shape = (1, int(seq_len), int(channels))
        self.anchor0 = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        self.anchor1 = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
        self.latent = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)

    def _outputs(
        self,
        selected_latent: torch.Tensor,
        new_anchor0: torch.Tensor,
        new_anchor1: torch.Tensor,
        new_latent: torch.Tensor,
        checksum: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if checksum is None:
            checksum = selected_latent.float().square().mean()
        return selected_latent, checksum, new_anchor0, new_anchor1, new_latent

    def _euler(self, noise: torch.Tensor, delta_sigma: torch.Tensor) -> torch.Tensor:
        # Exact non-stochastic FlowMatchEulerDiscreteScheduler update:
        # sample.float() + (sigma_next - sigma) * model_output, then cast to
        # the model-output dtype.
        delta = delta_sigma.float().reshape(-1)[0]
        return (self.latent.float() + delta * noise.float()).to(dtype=noise.dtype)


class ResidentCacheInitializeModel(_ResidentCacheStepState):
    def forward(
        self, initial_latent: torch.Tensor, request_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero0 = self.anchor0 * 0.0
        zero1 = self.anchor1 * 0.0
        new_latent = initial_latent + self.latent * 0.0 + zero0 + zero1
        checksum = initial_latent.float().square().mean() + request_token.float().sum() * 0.0
        return self._outputs(initial_latent, zero0, zero1, new_latent, checksum)


class ResidentCacheAnchorStepModel(_ResidentCacheStepState):
    def forward(
        self, actual_noise: torch.Tensor, delta_sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        updated = self._euler(actual_noise, delta_sigma)
        selected = updated + self.anchor0 * 0.0 + self.anchor1 * 0.0
        new_anchor0 = self.anchor1 + self.anchor0 * 0.0
        new_anchor1 = actual_noise + self.anchor0 * 0.0 + self.anchor1 * 0.0
        return self._outputs(selected, new_anchor0, new_anchor1, updated)


class ResidentCacheSkipStepModel(_ResidentCacheStepState):
    def forward(
        self, coefficients: torch.Tensor, delta_sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = coefficients.float().reshape(-1)
        predicted = (
            self.anchor0.float() * weights[0] + self.anchor1.float() * weights[1]
        ).to(dtype=self.anchor0.dtype)
        updated = self._euler(predicted, delta_sigma)
        selected = updated + self.anchor0 * 0.0 + self.anchor1 * 0.0
        return self._outputs(selected, self.anchor0, self.anchor1, updated)


class ResidentCacheSkipStepBarrierModel(_ResidentCacheStepState):
    """Fuse predictor and Euler while preserving the BF16 predictor contract.

    The existing serving contract materializes the predictor result as BF16
    before the scheduler consumes it.  In an ordinary fused XLA graph the
    compiler can fold the BF16 cast into the following FP32 arithmetic.  The
    XLA optimization barrier makes that precision boundary observable to the
    compiler without exposing the tensor to the host.
    """

    def forward(
        self, coefficients: torch.Tensor, delta_sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = coefficients.float().reshape(-1)
        predicted = (
            self.anchor0.float() * weights[0] + self.anchor1.float() * weights[1]
        ).to(dtype=self.anchor0.dtype)
        if predicted.device.type == "xla":
            import torch_xla.core.xla_model as xm

            xm.optimization_barrier_([predicted])
        updated = self._euler(predicted, delta_sigma)
        selected = updated + self.anchor0 * 0.0 + self.anchor1 * 0.0
        return self._outputs(selected, self.anchor0, self.anchor1, updated)


class ResidentCacheSkipSegmentScanModel(_ResidentCacheStepState):
    """Run a fixed-size skip segment as an XLA While/scan.

    A12 contains six contiguous skip segments with at most nine steps.  The
    latent is a BF16 loop carry, making the dtype part of the While body
    signature instead of a removable intermediate cast in a straight-line
    fused graph.  Shorter segments pad coefficients and delta sigma with zero.
    """

    max_segment_steps = 9

    def _step(
        self,
        latent: torch.Tensor,
        values: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients, delta_sigma = values
        weights = coefficients.float().reshape(-1)
        predicted = (
            self.anchor0.float() * weights[0] + self.anchor1.float() * weights[1]
        ).to(dtype=self.anchor0.dtype)
        updated = (
            latent.float() + delta_sigma.float().reshape(-1)[0] * predicted.float()
        ).to(dtype=latent.dtype)
        return updated, updated.float().square().mean()

    def _while_condition(
        self,
        sentinel: torch.Tensor,
        counter: torch.Tensor,
        limit: torch.Tensor,
        one: torch.Tensor,
        latent: torch.Tensor,
        anchor0: torch.Tensor,
        anchor1: torch.Tensor,
        *control_scalars: torch.Tensor,
    ) -> torch.Tensor:
        del sentinel, one, latent, anchor0, anchor1
        del control_scalars
        return counter < limit

    def _while_body(
        self,
        sentinel: torch.Tensor,
        counter: torch.Tensor,
        limit: torch.Tensor,
        one: torch.Tensor,
        latent: torch.Tensor,
        anchor0: torch.Tensor,
        anchor1: torch.Tensor,
        *control_scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        del sentinel
        if len(control_scalars) != 3 * self.max_segment_steps:
            raise ValueError("while body requires 27 scalar controls")
        coefficient0 = control_scalars[: self.max_segment_steps]
        coefficient1 = control_scalars[
            self.max_segment_steps : 2 * self.max_segment_steps
        ]
        delta_sigmas = control_scalars[2 * self.max_segment_steps :]
        predicted = (
            anchor0.float() * coefficient0[0]
            + anchor1.float() * coefficient1[0]
        ).to(dtype=anchor0.dtype)
        updated = (
            latent.float() + delta_sigmas[0] * predicted.float()
        ).to(dtype=latent.dtype)
        shifted_controls = (
            *coefficient0[1:],
            coefficient0[-1],
            *coefficient1[1:],
            coefficient1[-1],
            *delta_sigmas[1:],
            delta_sigmas[-1],
        )
        return (
            counter.clone(),
            counter + one,
            limit.clone(),
            one.clone(),
            updated,
            anchor0.clone(),
            anchor1.clone(),
            *(value.clone() for value in shifted_controls),
        )

    def forward(
        self, coefficients: torch.Tensor, delta_sigmas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if coefficients.shape != (self.max_segment_steps, 2):
            raise ValueError(
                f"coefficients must have shape {(self.max_segment_steps, 2)}"
            )
        if delta_sigmas.shape != (self.max_segment_steps, 1):
            raise ValueError(
                f"delta_sigmas must have shape {(self.max_segment_steps, 1)}"
            )
        if self.latent.device.type == "xla":
            # The public scan frontend uses AOTAutograd FakeTensor tracing, and
            # the public HigherOrderOperator is intercepted by Neuron's HLO
            # naming mode.  The registered XLA implementation exposes the same
            # While builder with fake carried inputs, which also prevents input
            # views from being hoisted as unmatched body parameters.
            from torch_xla.experimental.fori_loop import _xla_while_loop_wrapper

            counter = torch.zeros((), dtype=torch.int64, device=self.latent.device)
            sentinel = counter.clone()
            one = torch.ones((), dtype=torch.int64, device=self.latent.device)
            limit = torch.full(
                (),
                self.max_segment_steps,
                dtype=torch.int64,
                device=self.latent.device,
            )
            coefficient0 = coefficients[:, 0].unbind(0)
            coefficient1 = coefficients[:, 1].unbind(0)
            delta_values = delta_sigmas.reshape(self.max_segment_steps).unbind(0)
            loop_outputs = _xla_while_loop_wrapper(
                self._while_condition,
                self._while_body,
                (
                    sentinel,
                    counter,
                    limit,
                    one,
                    self.latent,
                    self.anchor0,
                    self.anchor1,
                    *coefficient0,
                    *coefficient1,
                    *delta_values,
                ),
                (),
                fake_tensor=True,
            )
            updated = loop_outputs[4]
            checksums = updated.float().square().mean().reshape(1)
        else:
            updated = self.latent
            checksum_values = []
            for index in range(self.max_segment_steps):
                updated, checksum = self._step(
                    updated, (coefficients[index], delta_sigmas[index])
                )
                checksum_values.append(checksum)
            checksums = torch.stack(checksum_values)
        selected = updated + self.anchor0 * 0.0 + self.anchor1 * 0.0
        return self._outputs(
            selected, self.anchor0, self.anchor1, updated, checksums
        )


class ResidentCachePredictOnlyModel(_ResidentCacheStepState):
    """Materialize the predictor result as a BF16 ranked NEFF output."""

    def forward(
        self, coefficients: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = coefficients.float().reshape(-1)
        predicted = (
            self.anchor0.float() * weights[0] + self.anchor1.float() * weights[1]
        ).to(dtype=self.anchor0.dtype)
        selected = predicted + self.latent * 0.0
        return self._outputs(selected, self.anchor0, self.anchor1, self.latent)


class ResidentCacheSchedulerStepModel(_ResidentCacheStepState):
    """Consume a BF16 ranked prediction and update the resident latent."""

    def forward(
        self, predicted: torch.Tensor, delta_sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        updated = self._euler(predicted, delta_sigma)
        selected = updated + self.anchor0 * 0.0 + self.anchor1 * 0.0
        return self._outputs(selected, self.anchor0, self.anchor1, updated)


class ResidentCacheFinalizeModel(_ResidentCacheStepState):
    def forward(
        self, finalize_token: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = self.latent + self.anchor0 * 0.0 + self.anchor1 * 0.0
        checksum = selected.float().square().mean() + finalize_token.float().sum() * 0.0
        return self._outputs(selected, self.anchor0, self.anchor1, self.latent, checksum)


__all__ = [
    "ResidentCacheAnchorStepModel",
    "ResidentCacheFinalizeModel",
    "ResidentCacheInitializeModel",
    "ResidentCachePredictOnlyModel",
    "ResidentCacheSchedulerStepModel",
    "ResidentCacheSkipSegmentScanModel",
    "ResidentCacheSkipStepBarrierModel",
    "ResidentCacheSkipStepModel",
]
