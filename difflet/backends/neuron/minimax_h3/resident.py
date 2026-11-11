"""MiniMax-H3 resident on the neuron backend: one loaded model serving request after request.

The ``difflet generate`` launch loads, runs a batch and exits; a resident worker (``difflet
serve``) keeps the model across requests. What stays on the device, per rank of the mesh (TP,
or TP x Ulysses/gather_kv CP, e.g. TP4 x U2 on two chips):

* the DiT at the profile's one static shape and text bucket (9.4 GiB at tp4);
* the video decoder, head-sharded over the TP group (2.3 GiB at tp4 in fp32).

The Qwen3-VL conditioner (25B params, 47 GiB in bf16; 11.7 GiB per rank at tp4) runs once at the
start of a request. ``DIFFLET_H3_TEXT_ENCODER`` picks where:

* ``device``: loaded, used and released on the device inside every request (~11 s load +
  ~3.4 s encode); it needs that HBM free beside the resident DiT;
* ``host`` (default): resident in host memory on rank 0 and run on its CPU (~7 s at a 512-token bucket on
  trn3, 96 threads), the embeddings broadcast to every rank; no HBM at all;
* ``auto``: the device first; when any rank runs out of HBM loading it, every rank
  drops it and the request (and every later one) falls back to the host. At a 7 s 768p clip
  the device attempt does not fit beside the DiT (measured on trn3).

The audio decoder runs on the host of the primary rank. Every rank must call ``generate`` (the collectives need them all); only rank
0 gets the media back.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ResidentRequestResult:
    """Rank 0: ``video`` ``(1, 3, frames, H, W)`` in ``[0, 1]`` and ``audio`` ``(1, channels,
    samples)`` in ``[-1, 1]`` at ``sample_rate``; other ranks: both None."""

    video: torch.Tensor | None
    audio: torch.Tensor | None
    sample_rate: int
    seconds: dict[str, float] = field(default_factory=dict)


class NeuronH3DiTCall:
    """The denoise loop's DiT callable on the neuron backend: the Trainium graph's positional
    call on host tensors in, the host ``(video, audio)`` velocity pair out. The two bool masks
    go to the device as int32 (the application's call); every rank must call it."""

    def __init__(self, app) -> None:
        self.app = app
        self.seconds: list[float] = []

    def __call__(self, *inputs):
        import time

        import torch

        device = self.app.device
        moved = []
        for tensor in inputs:
            if tensor.dtype == torch.bool:
                tensor = tensor.to(torch.int32)
            moved.append(tensor.contiguous().to(device))
        start = time.perf_counter()
        video, audio = self.app(*moved)
        video, audio = video.cpu(), audio.cpu()
        self.seconds.append(time.perf_counter() - start)
        return video, audio


#: Where the conditioner runs: "host" (default), "auto" (device, falling back to the host on HBM
#: exhaustion) or "device".
TEXT_ENCODER_ENV = "DIFFLET_H3_TEXT_ENCODER"
TEXT_ENCODER_MODES = ("auto", "device", "host")


def text_encoder_mode() -> str:
    mode = os.environ.get(TEXT_ENCODER_ENV, "").strip().lower() or "host"
    if mode not in TEXT_ENCODER_MODES:
        raise ValueError(f"{TEXT_ENCODER_ENV} must be one of {TEXT_ENCODER_MODES}, got {mode!r}")
    return mode


#: How many denoise steps run between two cancellation checks (every rank agrees on each).
CANCEL_CHECK_STEPS_ENV = "DIFFLET_H3_CANCEL_CHECK_STEPS"
_DEFAULT_CANCEL_CHECK_STEPS = 5


def cancel_check_steps() -> int:
    raw = os.environ.get(CANCEL_CHECK_STEPS_ENV, "").strip()
    steps = int(raw) if raw else _DEFAULT_CANCEL_CHECK_STEPS
    if steps < 1:
        raise ValueError(f"{CANCEL_CHECK_STEPS_ENV} must be a positive integer, got {raw!r}")
    return steps


class ResidentRequestCancelled(Exception):
    """Every rank agreed to stop the request (raised on all of them at the same step)."""


def _is_hbm_exhaustion(exc: BaseException) -> bool:
    """The neuron runtime's out-of-device-memory error (its message names the allocation)."""
    text = str(exc)
    return isinstance(exc, RuntimeError) and ("alloc" in text or "NRT_RESOURCE" in text
                                              or "out of memory" in text.lower())


class NeuronMiniMaxH3Resident:
    def __init__(self, *, model_dir: str, parallel, height: int, width: int, num_frames: int,
                 text_bucket: int, cache_dir: str | None, vae_dtype=torch.float32,
                 log=print) -> None:
        self.model_dir = str(model_dir)
        self.parallel = parallel
        self.height, self.width, self.num_frames = int(height), int(width), int(num_frames)
        self.text_bucket = int(text_bucket)
        self.cache_dir = cache_dir
        self.vae_dtype = vae_dtype
        self._log = log
        self.dit_app = None
        self.vae = None
        self.audio_decoder = None
        self._tables: dict[int, torch.Tensor] = {}
        self.rank, self.world = 0, 1
        self.text_encoder_mode = text_encoder_mode()
        self._host_encoder = None  # rank 0 only, loaded at the first host encode

    def log(self, message: str) -> None:
        if self.rank == 0:
            self._log(message)

    # ------------------------------------------------------------------ startup

    def load(self) -> None:
        from difflet.backends.neuron.core.distributed import world_info
        from difflet.backends.neuron.minimax_h3.transformer import (
            NeuronMiniMaxH3Config,
            NeuronMiniMaxH3TransformerApplication,
        )
        from difflet.backends.neuron.minimax_h3.vae import (
            DeviceTileDecoder,
            NeuronMiniMaxH3VideoVaeApplication,
        )

        self.rank, self.world = world_info()
        start = time.perf_counter()
        config = NeuronMiniMaxH3Config.from_pretrained(
            str(Path(self.model_dir) / "transformer"), height=self.height, width=self.width,
            num_frames=self.num_frames, text_seq_len=self.text_bucket,
            tp_degree=int(self.parallel.tp_degree), precomputed_adaln=True,
        )
        self.config = config
        self.dit_app = NeuronMiniMaxH3TransformerApplication(
            model_path=str(Path(self.model_dir) / "transformer"), config=config,
            parallel=self.parallel, dtype=torch.bfloat16)
        self.dit_app.load()
        self.log(f"[minimax-h3 resident] DiT loaded on {self.world} ranks in "
                 f"{time.perf_counter() - start:.1f}s, "
                 f"{self.dit_app.weight_bytes / 2**30:.2f} GiB/rank")
        start = time.perf_counter()
        vae_app = NeuronMiniMaxH3VideoVaeApplication(
            model_path=str(Path(self.model_dir) / "vae"), parallel=self.parallel,
            dtype=self.vae_dtype, tp_shard=True)
        vae_app.load()
        self.vae = DeviceTileDecoder(vae_app)
        self.log(f"[minimax-h3 resident] video decoder ({self.vae_dtype}, TP-sharded) loaded in "
                 f"{time.perf_counter() - start:.1f}s, {vae_app.weight_bytes / 2**30:.2f} GiB/rank")
        if self.rank == 0:
            from difflet.models.minimax_h3.host_components import (
                default_threads,
                load_host_audio_decoder,
            )

            self.audio_decoder = load_host_audio_decoder(self.model_dir,
                                                         threads=default_threads())

    # ------------------------------------------------------------------ per request

    def encode(self, prompt: str) -> dict[str, Any]:
        """The conditioner's record for ``prompt`` on every rank (see the module docstring for
        where it runs). Every rank must call it."""
        if self.text_encoder_mode != "host":
            record = self._encode_on_device(prompt)
            if record is not None:
                return record
            # auto, and some rank ran out of HBM: this request and every later one go to the host.
            self.text_encoder_mode = "host"
            self.log("[minimax-h3 resident] the text encoder does not fit in HBM beside the DiT; "
                     "falling back to the host for this and every later request")
        return self._encode_on_host(prompt)

    def _encode_on_device(self, prompt: str) -> dict[str, Any] | None:
        """Load, use and release the device conditioner; None when ``auto`` and any rank ran out
        of HBM (every rank agrees before anyone falls back, so the collectives stay matched)."""
        from difflet.backends.neuron.core.distributed import RankFailureError, sync_status
        from difflet.backends.neuron.minimax_h3.text_encoder import (
            NeuronMiniMaxH3TextEncoderApplication,
        )

        encoder = NeuronMiniMaxH3TextEncoderApplication(
            model_path=str(Path(self.model_dir) / "text_encoder"), parallel=self.parallel,
            dtype=torch.bfloat16)
        device = self.dit_app.device
        failure: BaseException | None = None
        try:
            encoder.load(skip_warmup=True)
        except (RuntimeError, RankFailureError) as exc:
            failure = exc
        try:
            # Every rank reports; the healthy ones raise RankFailureError when any rank failed.
            sync_status(failure is None, device=device, what="device text encoder load")
        except RankFailureError as exc:
            failure = failure or exc
        if failure is not None:
            encoder.release()
            if self.text_encoder_mode == "device" or not (
                    _is_hbm_exhaustion(failure) or isinstance(failure, RankFailureError)):
                raise failure
            return None
        try:
            return encoder.encode(prompt)
        finally:
            encoder.release()

    def _encode_on_host(self, prompt: str) -> dict[str, Any]:
        """Rank 0 encodes on its CPU (the conditioner stays resident in host memory); the
        embeddings, mask and live length are broadcast to every rank."""
        from difflet.backends.neuron.core.distributed import broadcast_tensor, rank0_call

        device = self.dit_app.device

        def encode() -> dict[str, Any]:
            if self._host_encoder is None:
                from difflet.models.minimax_h3.host_components import (
                    HostQwen3VLTextEncoder,
                    default_threads,
                )

                start = time.perf_counter()
                self._host_encoder = HostQwen3VLTextEncoder(
                    self.model_dir, seq_len=self.text_bucket, threads=default_threads()).load()
                self.log(f"[minimax-h3 resident] host text encoder loaded in "
                         f"{time.perf_counter() - start:.1f}s")
            return self._host_encoder.encode(prompt)

        record = rank0_call(encode, device=device, what="host text encode")
        hidden = broadcast_tensor(None if record is None else record["encoder_hidden_states"],
                                  device=device).cpu()
        mask = broadcast_tensor(
            None if record is None else record["attention_mask"].to(torch.int32),
            device=device).cpu().to(torch.bool)
        count = broadcast_tensor(
            None if record is None else torch.tensor([int(record["num_text_tokens"])],
                                                     dtype=torch.int32),
            device=device).cpu()
        layer = record["text_encoder_layer"] if record is not None else None
        return {"encoder_hidden_states": hidden, "attention_mask": mask,
                "num_text_tokens": int(count.item()), "text_encoder_layer": layer}

    def modulation_table(self, steps: int) -> torch.Tensor:
        """The precomputed AdaLN table of a ``steps``-step schedule: built once on rank 0 into
        ``<cache_dir>/minimax_h3_neuron/`` (the file the CLI writes too), read by every rank."""
        steps = int(steps)
        if steps not in self._tables:
            from difflet.backends.neuron.core.distributed import rank0_call
            from difflet.models.minimax_h3.pipeline import (
                build_adaln_modulation_table,
                load_minimax_h3_schedulers,
            )

            root = Path(self.cache_dir or Path.home() / ".cache" / "difflet")
            path = root / "minimax_h3_neuron" / f"adaln_table_steps{steps}.pt"
            config = self.config

            def build() -> None:
                if path.exists():
                    return
                video_sched, audio_sched = load_minimax_h3_schedulers(self.model_dir)
                video_sched.set_timesteps(steps, device="cpu")
                audio_sched.set_timesteps(steps, device="cpu")
                # Slot order matches the denoise loop: 0 = audio, 1 = video.
                timesteps = torch.stack([audio_sched.timesteps.to(torch.float32),
                                         video_sched.timesteps.to(torch.float32)], dim=-1)
                table = build_adaln_modulation_table(
                    str(Path(self.model_dir) / "transformer"), timesteps,
                    num_layers=int(config.num_layers), freq_dim=int(config.freq_dim),
                    time_embed_hidden_dim=int(config.time_embed_hidden_dim),
                    time_embed_dim=int(config.time_embed_dim),
                    hidden_size=int(config.hidden_size), dtype=torch.bfloat16)
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(".pt.partial")
                torch.save(table, partial)
                partial.replace(path)

            rank0_call(build, device=self.dit_app.device, what="AdaLN modulation table")
            self._tables[steps] = torch.load(path)
        return self._tables[steps]

    def agree_to_stop(self, stop: bool) -> bool:
        """Whether any rank wants to stop: one all-reduce(MAX) of a flag, so every rank leaves
        the loop at the same step and no collective is left waiting for a peer."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()) or self.world == 1:
            return bool(stop)
        flag = torch.tensor([1 if stop else 0], dtype=torch.int32, device=self.dit_app.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(int(flag.cpu().item()))

    def generate(self, prompt: str, *, steps: int, seed: int,
                 scratch_dir: str | Path, should_stop=None) -> ResidentRequestResult:
        """One request on every rank. ``should_stop()`` (this rank's cancellation/deadline) is
        polled after the text encode and every ``DIFFLET_H3_CANCEL_CHECK_STEPS`` denoise steps;
        once any rank says stop, every rank raises ``ResidentRequestCancelled`` together."""
        from difflet.backends.neuron.minimax_h3.vae import decode_video_on_ranks
        from difflet.models.minimax_h3.pipeline import (
            decode_minimax_h3_audio,
            denoise_minimax_h3_t2va,
            load_minimax_h3_schedulers,
        )

        seconds: dict[str, float] = {}
        start = time.perf_counter()
        text = self.encode(prompt)
        seconds["text"] = time.perf_counter() - start
        live = int(text["num_text_tokens"])
        if live > self.text_bucket:
            raise ValueError(f"the prompt has {live} tokens; this profile's text bucket is "
                             f"{self.text_bucket}")
        table = self.modulation_table(steps)
        video_scheduler, audio_scheduler = load_minimax_h3_schedulers(self.model_dir)
        every = cancel_check_steps()
        total = int(steps)

        def check(where: str) -> None:
            if should_stop is not None and self.agree_to_stop(bool(should_stop())):
                self.log(f"[minimax-h3 resident] request cancelled ({where})")
                raise ResidentRequestCancelled(where)

        def after_step(index: int) -> None:
            if (index + 1) % every == 0 and index + 1 < total:
                check(f"after denoise step {index + 1}/{total}")

        check("after the text encode")
        start = time.perf_counter()
        output = denoise_minimax_h3_t2va(
            NeuronH3DiTCall(self.dit_app),
            encoder_hidden_states=text["encoder_hidden_states"][:, :self.text_bucket],
            encoder_attention_mask=text["attention_mask"][:, :self.text_bucket],
            num_text_tokens=live, height=self.height, width=self.width,
            num_frames=self.num_frames, num_inference_steps=int(steps), seed=int(seed),
            video_scheduler=video_scheduler, audio_scheduler=audio_scheduler,
            model_dtype=torch.bfloat16, adaln_modulation_table=table,
            step_callback=after_step if should_stop is not None else None,
        )
        seconds["denoise"] = time.perf_counter() - start
        check("after the denoise")
        start = time.perf_counter()
        group, groups, writer = self.vae.app.tile_groups(self.rank, self.world)
        video = decode_video_on_ranks(
            self.vae, output.video_latents, self.vae.config, rank=group, world=groups,
            scratch_dir=scratch_dir, device=self.dit_app.device, writer=writer)
        seconds["video_vae"] = time.perf_counter() - start
        sample_rate = 0
        audio = None
        if self.rank == 0:
            start = time.perf_counter()
            cfg = self.audio_decoder.config
            sample_rate = int(cfg["sampling_rate"])
            audio = decode_minimax_h3_audio(
                self.audio_decoder, output.audio_latents, latents_mean=cfg["latents_mean"],
                latents_std=cfg["latents_std"], hop_length=800,
            ).float().clamp(-1, 1)
            seconds["audio_vae"] = time.perf_counter() - start
            video = video.float().clamp(0, 1)
        self.log("[minimax-h3 resident] request: "
                 + ", ".join(f"{k} {v:.1f}s" for k, v in seconds.items()))
        return ResidentRequestResult(video=video, audio=audio, sample_rate=sample_rate,
                                     seconds=seconds)

    def shutdown(self) -> None:
        self._host_encoder = None
        self.dit_app = None
        self.vae = None
        self.audio_decoder = None
        self._tables.clear()


__all__ = [
    "CANCEL_CHECK_STEPS_ENV",
    "NeuronH3DiTCall",
    "NeuronMiniMaxH3Resident",
    "ResidentRequestCancelled",
    "ResidentRequestResult",
    "TEXT_ENCODER_ENV",
    "cancel_check_steps",
    "text_encoder_mode",
]
