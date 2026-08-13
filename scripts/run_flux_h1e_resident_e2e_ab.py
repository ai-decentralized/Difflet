#!/usr/bin/env python3
"""Run the H1e materializing-vs-resident real-prompt FLUX A12 A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--black-forest-labs--FLUX.1-dev/snapshots/"
    "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
)
DEFAULT_COMPILED_ROOT = Path("/home/ubuntu/.cache/difflet/flux/ebbfe33fed51b364")
DEFAULT_CACHE_ARTIFACT = Path(
    "/home/ubuntu/difflet-artifacts/flux-h1c-resident-cache-step-20260812/"
    "compiled-split-v2"
)
DEFAULT_CONTEXT_ARTIFACT = Path(
    "/home/ubuntu/difflet-artifacts/flux-h1d-request-context-20260812/compiled-v2"
)
ANCHORS = frozenset((0, 1, 2, 3, 4, 5, 9, 15, 21, 31, 41, 49))
PROMPTS = (
    "a red fox sitting in a snowy forest at dawn, sharp detail",
    "a bustling night market with neon signs and steam, cinematic",
    "a precise architectural cutaway of a compact coastal research station, "
    "labeled rooms, realistic materials",
    "a macro photograph of a translucent blue dragonfly wing covered in morning "
    "dew, shallow depth of field",
)
SEEDS = (101, 202, 303, 404)


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
        import torch_neuronx  # noqa: F401
    except (ModuleNotFoundError, OSError):
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = (
                f"{NEURON_VENV / 'bin'}:/opt/aws/neuron/bin:{env.get('PATH', '')}"
            )
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.pipelines.flux.pipeline_flux import (  # noqa: E402
    calculate_shift,
    retrieve_timesteps,
)

from difflet.backends.trainium.flux.request_context_stage import (  # noqa: E402
    NeuronFluxRequestContextStageApplication,
)
from difflet.backends.trainium.flux.resident_cache_step_split import (  # noqa: E402
    NeuronFluxResidentCacheStepSplitApplication,
)
from difflet.models.flux.application import (  # noqa: E402
    NeuronFluxApplication,
    create_flux_config,
)
from difflet.models.flux.modeling_flux import FluxBackboneInferenceConfig  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual.float() - expected.float()
    return (
        float(difference.abs().max().item()),
        float(
            torch.linalg.vector_norm(difference).item()
            / max(torch.linalg.vector_norm(expected.float()).item(), 1e-30)
        ),
    )


def _checksum(value: torch.Tensor) -> dict[str, float]:
    data = value.float()
    return {
        "mean": float(data.mean().item()),
        "mean_square": float(data.square().mean().item()),
        "maximum_absolute": float(data.abs().max().item()),
    }


def _schedule(model_path: Path, seq_len: int):
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_path), subfolder="scheduler"
    )
    sigmas = np.linspace(1.0, 1.0 / 50, 50)
    mu = calculate_shift(
        seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        scheduler, 50, "cpu", sigmas=sigmas, mu=mu
    )
    return timesteps, scheduler.sigmas[1:] - scheduler.sigmas[:-1]


def _read_rank0(ranked_output) -> torch.Tensor:
    return ranked_output[0][0].cpu()


def _coefficients(history_steps: list[int], step: int) -> torch.Tensor:
    a, b = history_steps[-2:]
    denominator = float(b - a)
    return torch.tensor(
        [(b - step) / denominator, (step - a) / denominator],
        dtype=torch.float32,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--compiled-root", type=Path, default=DEFAULT_COMPILED_ROOT)
    parser.add_argument("--cache-artifact", type=Path, default=DEFAULT_CACHE_ARTIFACT)
    parser.add_argument(
        "--context-artifact", type=Path, default=DEFAULT_CONTEXT_ARTIFACT
    )
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument(
        "--profile-arm", choices=("materializing", "resident"), default=None
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


class H1eRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        (
            clip_config,
            t5_config,
            backbone_config,
            decoder_config,
        ) = create_flux_config(
            model_path=str(args.model_path),
            world_size=4,
            backbone_tp_degree=4,
            dtype=torch.bfloat16,
            height=1024,
            width=1024,
        )
        self.app = NeuronFluxApplication(
            model_path=str(args.model_path),
            text_encoder_config=clip_config,
            text_encoder2_config=t5_config,
            backbone_config=backbone_config,
            decoder_config=decoder_config,
            height=1024,
            width=1024,
        )
        cache_config = FluxBackboneInferenceConfig.load(str(args.cache_artifact))
        self.cache = NeuronFluxResidentCacheStepSplitApplication(
            model_path=str(args.model_path / "transformer"), config=cache_config
        )
        context_config = FluxBackboneInferenceConfig.load(str(args.context_artifact))
        self.context = NeuronFluxRequestContextStageApplication(
            model_path=str(args.model_path / "transformer"), config=context_config
        )
        self.backbone = self.app.pipe.transformer
        self.pipe = self.app.pipe
        self.load_seconds: dict[str, float] = {}

    def load(self) -> None:
        started = time.perf_counter()
        self.app.load(str(self.args.compiled_root), skip_warmup=True)
        self.load_seconds["flux_pipeline_components"] = time.perf_counter() - started
        for name, application, artifact in (
            ("cache_step", self.cache, self.args.cache_artifact),
            ("request_context", self.context, self.args.context_artifact),
        ):
            started = time.perf_counter()
            application.load(str(artifact), skip_warmup=True)
            self.load_seconds[name] = time.perf_counter() - started

    def _prepare_request(self, prompt: str, seed: int):
        timings: dict[str, float] = {}
        full_started = time.perf_counter()
        started = time.perf_counter()
        prompt_embeds, pooled, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=torch.device("cpu"),
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )
        prompt_embeds = prompt_embeds.to(torch.bfloat16)
        pooled = pooled.to(torch.bfloat16)
        timings["text_encode_s"] = time.perf_counter() - started

        started = time.perf_counter()
        generator = torch.Generator().manual_seed(seed)
        latents, image_ids = self.pipe.prepare_latents(
            1,
            int(self.backbone.config.in_channels) // 4,
            1024,
            1024,
            prompt_embeds.dtype,
            torch.device("cpu"),
            generator,
            None,
        )
        ids = torch.cat((text_ids, image_ids), dim=0)
        rotary = torch.stack(self.backbone.model.pos_embed(ids), dim=2).to(
            torch.bfloat16
        )
        guidance = torch.tensor([3.5], dtype=torch.bfloat16)
        timesteps, deltas = _schedule(self.args.model_path, int(latents.shape[1]))
        timings["latent_rotary_schedule_setup_s"] = time.perf_counter() - started
        return (
            full_started,
            timings,
            latents.to(torch.bfloat16),
            (prompt_embeds, pooled, guidance, rotary),
            timesteps,
            deltas,
        )

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        unpacked = self.pipe._unpack_latents(latent, 1024, 1024, self.pipe.vae_scale_factor)
        unpacked = (
            unpacked / self.pipe.vae.config.scaling_factor
        ) + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(unpacked, return_dict=False)[0]
        return image.detach().cpu()

    def _backbone_materializing(
        self,
        latent: torch.Tensor,
        context_values: tuple[torch.Tensor, ...],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        encoder, pooled, guidance, rotary = context_values
        ranked_inputs = [
            [latent, encoder, pooled, timestep, guidance, rotary] for _ in range(4)
        ]
        ranked = self.backbone.traced_model.nxd_model.forward_ranked(ranked_inputs)
        return _read_rank0(ranked)

    def _backbone_resident(
        self,
        ranked_latent,
        ranked_context,
        timestep: torch.Tensor,
    ):
        ranked_inputs = [
            [
                latent_outputs[0],
                context_outputs[0],
                context_outputs[1],
                timestep,
                context_outputs[2],
                context_outputs[3],
            ]
            for latent_outputs, context_outputs in zip(
                ranked_latent, ranked_context, strict=True
            )
        ]
        return self.backbone.traced_model.nxd_model.forward_ranked(ranked_inputs)

    def run_request(self, arm: str, prompt: str, seed: int) -> dict[str, Any]:
        (
            full_started,
            timings,
            initial,
            context_values,
            timesteps,
            deltas,
        ) = self._prepare_request(prompt, seed)
        denoise_started = time.perf_counter()
        context_stage_seconds = 0.0
        final_read_seconds = 0.0
        history_steps: list[int] = []

        if arm == "materializing":
            latent = initial
            host_anchors: list[torch.Tensor] = []
            for step in range(50):
                if step in ANCHORS:
                    timestep = torch.tensor(
                        [float(timesteps[step].item()) / 1000.0],
                        dtype=torch.bfloat16,
                    )
                    noise = self._backbone_materializing(
                        latent, context_values, timestep
                    )
                    host_anchors.append(noise)
                    history_steps.append(step)
                    if len(host_anchors) > 2:
                        del host_anchors[0]
                        del history_steps[0]
                else:
                    coefficients = _coefficients(history_steps, step)
                    noise = (
                        host_anchors[0].float() * float(coefficients[0].item())
                        + host_anchors[1].float() * float(coefficients[1].item())
                    ).to(torch.bfloat16)
                latent = (
                    latent.float() + float(deltas[step].item()) * noise.float()
                ).to(torch.bfloat16)
            final_latent = latent
        elif arm == "resident":
            started = time.perf_counter()
            ranked_context = self.context.ranked_forward(0, *context_values)
            context_stage_seconds = time.perf_counter() - started
            resident = self.cache.ranked_forward(
                initial, torch.tensor([seed, 0, 0, 0], dtype=torch.int32)
            )
            for step in range(50):
                if step in ANCHORS:
                    timestep = torch.tensor(
                        [float(timesteps[step].item()) / 1000.0],
                        dtype=torch.bfloat16,
                    )
                    ranked_noise = self._backbone_resident(
                        resident, ranked_context, timestep
                    )
                    resident = self.cache.ranked_anchor(
                        ranked_noise,
                        torch.tensor(
                            [float(deltas[step].item()), 0.0], dtype=torch.float32
                        ),
                    )
                    history_steps.append(step)
                    if len(history_steps) > 2:
                        del history_steps[0]
                else:
                    predicted = self.cache.ranked_forward(
                        _coefficients(history_steps, step)
                    )
                    resident = self.cache.ranked_scheduler(
                        predicted,
                        torch.tensor(
                            [float(deltas[step].item())], dtype=torch.float32
                        ),
                    )
            started = time.perf_counter()
            final_latent = _read_rank0(
                self.cache.ranked_forward(torch.tensor([0, 0, 1], dtype=torch.int32))
            )
            final_read_seconds = time.perf_counter() - started
        else:
            raise ValueError(f"unknown arm: {arm}")

        timings["context_stage_s"] = context_stage_seconds
        timings["final_read_s"] = final_read_seconds
        timings["denoise_total_s"] = time.perf_counter() - denoise_started
        started = time.perf_counter()
        decoded = self._decode(final_latent)
        timings["vae_decode_s"] = time.perf_counter() - started
        timings["full_request_s"] = time.perf_counter() - full_started
        return {
            "arm": arm,
            "prompt": prompt,
            "seed": seed,
            "timings": timings,
            "final_latent": final_latent,
            "decoded": decoded,
            "final_latent_checksum": _checksum(final_latent),
            "decoded_checksum": _checksum(decoded),
        }


def _public_result(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in ("final_latent", "decoded")}


def main() -> int:
    args = _parse_args()
    for name in (
        "model_path",
        "compiled_root",
        "cache_artifact",
        "context_artifact",
        "result",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for artifact in (
        args.compiled_root / "transformer",
        args.compiled_root / "text_encoder",
        args.compiled_root / "text_encoder_2",
        args.compiled_root / "decoder",
        args.cache_artifact,
        args.context_artifact,
    ):
        if not (artifact / "model.pt").is_file():
            raise FileNotFoundError(artifact / "model.pt")

    runner = H1eRunner(args)
    runner.load()
    common = {
        "schema": "difflet-flux-h1e-resident-e2e-ab-raw-result",
        "schema_revision": 1,
        "study_id": "flux-h1e-resident-e2e-ab-20260812",
        "mode": (
            f"profile_{args.profile_arm}"
            if args.profile_arm is not None
            else "quick"
            if args.quick
            else "paired_benchmark"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "load_seconds": runner.load_seconds,
        "artifacts": {
            name: {
                "path": str(path),
                "model_pt_sha256": _sha256(path / "model.pt"),
            }
            for name, path in {
                "text_encoder": args.compiled_root / "text_encoder",
                "text_encoder_2": args.compiled_root / "text_encoder_2",
                "backbone": args.compiled_root / "transformer",
                "decoder": args.compiled_root / "decoder",
                "cache_step": args.cache_artifact,
                "request_context": args.context_artifact,
            }.items()
        },
        "generation": {
            "height": 1024,
            "width": 1024,
            "num_steps": 50,
            "guidance_scale": 3.5,
            "anchor_steps": sorted(ANCHORS),
            "predictor": "order1_index_taylorseer",
        },
    }

    if args.profile_arm is not None:
        run = runner.run_request(args.profile_arm, PROMPTS[0], SEEDS[0])
        payload = {
            **common,
            "status": "profile_request_completed",
            "request": _public_result(run),
        }
        _write_json(args.result, payload)
        print(
            f"[h1e] profile {args.profile_arm} "
            f"full={run['timings']['full_request_s']:.3f}s",
            flush=True,
        )
        return 0

    warmups = []
    for arm in ("materializing", "resident"):
        run = runner.run_request(
            arm,
            "a matte gray cube centered on a neutral studio background",
            909,
        )
        warmups.append(_public_result(run))
        print(
            f"[h1e] warmup {arm} full={run['timings']['full_request_s']:.3f}s",
            flush=True,
        )

    repetitions = 1 if args.quick else 6
    prompt_count = 1 if args.quick else len(PROMPTS)
    records = []
    maximum_latent_absolute_error = 0.0
    maximum_latent_relative_l2_error = 0.0
    maximum_decoded_absolute_error = 0.0
    maximum_decoded_relative_l2_error = 0.0
    for prompt_index in range(prompt_count):
        prompt = PROMPTS[prompt_index]
        seed = SEEDS[prompt_index]
        for repetition in range(repetitions):
            order = (
                ("materializing", "resident")
                if repetition % 2 == 0
                else ("resident", "materializing")
            )
            by_arm = {}
            for arm in order:
                run = runner.run_request(arm, prompt, seed)
                by_arm[arm] = run
                print(
                    f"[h1e] p={prompt_index} rep={repetition} {arm} "
                    f"denoise={run['timings']['denoise_total_s']:.3f}s "
                    f"full={run['timings']['full_request_s']:.3f}s",
                    flush=True,
                )
            baseline = by_arm["materializing"]
            resident = by_arm["resident"]
            latent_absolute, latent_relative = _errors(
                resident["final_latent"], baseline["final_latent"]
            )
            decoded_absolute, decoded_relative = _errors(
                resident["decoded"], baseline["decoded"]
            )
            maximum_latent_absolute_error = max(
                maximum_latent_absolute_error, latent_absolute
            )
            maximum_latent_relative_l2_error = max(
                maximum_latent_relative_l2_error, latent_relative
            )
            maximum_decoded_absolute_error = max(
                maximum_decoded_absolute_error, decoded_absolute
            )
            maximum_decoded_relative_l2_error = max(
                maximum_decoded_relative_l2_error, decoded_relative
            )
            records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "seed": seed,
                    "repetition": repetition,
                    "order": list(order),
                    "materializing": _public_result(baseline),
                    "resident": _public_result(resident),
                    "parity": {
                        "final_latent_maximum_absolute_error": latent_absolute,
                        "final_latent_relative_l2_error": latent_relative,
                        "decoded_maximum_absolute_error": decoded_absolute,
                        "decoded_relative_l2_error": decoded_relative,
                    },
                }
            )

    exact = (
        maximum_latent_absolute_error == 0.0
        and maximum_latent_relative_l2_error == 0.0
        and maximum_decoded_absolute_error == 0.0
        and maximum_decoded_relative_l2_error == 0.0
    )
    payload = {
        **common,
        "status": "paired_benchmark_completed" if exact else "parity_failed",
        "warmups": warmups,
        "paired_request_count": len(records),
        "records": records,
        "parity_summary": {
            "final_latent_maximum_absolute_error": maximum_latent_absolute_error,
            "final_latent_maximum_relative_l2_error": maximum_latent_relative_l2_error,
            "decoded_maximum_absolute_error": maximum_decoded_absolute_error,
            "decoded_maximum_relative_l2_error": maximum_decoded_relative_l2_error,
            "passed": exact,
        },
    }
    _write_json(args.result, payload)
    print(
        f"[h1e] {payload['status']} pairs={len(records)} "
        f"latent_abs={maximum_latent_absolute_error:.6g}",
        flush=True,
    )
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
