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

Modes: a resident serves one or more of text-only (0 keyframes), one keyframe (first or last)
and both keyframes, each at its own text bucket (``modes={keyframes: bucket}``). The DiT's
weights load once; every mode is a graph of its own on them (FL2VA graphs carry the keyframe
condition frames ahead of the generated rows and a third timestep slot), warmed at load, so a
mode costs a warm-up at startup and its graph's device memory, not another copy of the weights.
A request runs the graph of its keyframe count, which every rank must agree on. A request's keyframes are prepared and
encoded on rank 0's host only (Qwen3-VL with its vision tower, read at ``hidden_states[50]``, and
the video VAE encoder), whatever ``DIFFLET_H3_TEXT_ENCODER`` says, since the device conditioner
has no vision tower; the presentation, its row tags, the condition latents and the anchors are
broadcast to every rank.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def _mode_name(keyframes: int) -> str:
    return {0: "text-only", 1: "one-keyframe", 2: "first+last-keyframe"}[int(keyframes)]


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
                 text_bucket: int | None = None, cache_dir: str | None,
                 vae_dtype=torch.float32, log=None, num_keyframes: int = 0,
                 modes: dict[int, int] | None = None) -> None:
        self.model_dir = str(model_dir)
        self.parallel = parallel
        self.height, self.width, self.num_frames = int(height), int(width), int(num_frames)
        if modes is None:
            if text_bucket is None:
                raise ValueError("pass modes={keyframes: text_bucket} or text_bucket")
            modes = {int(num_keyframes): int(text_bucket)}
        self.modes = {int(k): int(b) for k, b in sorted(modes.items())}
        if not self.modes or any(k not in (0, 1, 2) or b < 1 for k, b in self.modes.items()):
            raise ValueError(f"modes must map keyframe counts 0/1/2 to text buckets, got {modes}")
        self.cache_dir = cache_dir
        self.vae_dtype = vae_dtype
        self._log = log or (lambda message: print(f"[minimax-h3 resident] {message}", flush=True))
        self.dit_app = None
        self.vae = None
        self.audio_decoder = None
        self._tables: dict[int, torch.Tensor] = {}
        self.rank, self.world = 0, 1
        self.text_encoder_mode = text_encoder_mode()
        self._host_encoder = None  # rank 0 only, loaded at the first host encode
        self._host_video_encoder = None  # rank 0 only, FL2VA

    @property
    def fl2va(self) -> bool:
        """Whether any mode takes keyframes (the host conditioner then loads its vision tower)."""
        return any(self.modes)

    @property
    def text_bucket(self) -> int:
        """The largest bucket (the host conditioner's padding length)."""
        return max(self.modes.values())

    @property
    def num_keyframes(self) -> int:
        """The single mode's keyframe count (a one-mode resident)."""
        if len(self.modes) != 1:
            raise AttributeError("a multi-mode resident has no single num_keyframes; see modes")
        return next(iter(self.modes))

    def log(self, message: str) -> None:
        """Rank 0 only. ``log`` (the constructor's) gets the bare message; the default prints it
        as ``[minimax-h3 resident] ...``, the serving adapter as ``[difflet serve] MiniMax-H3:``."""
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
        (first_keyframes, first_bucket), *more = self.modes.items()
        config = NeuronMiniMaxH3Config.from_pretrained(
            str(Path(self.model_dir) / "transformer"), height=self.height, width=self.width,
            num_frames=self.num_frames, text_seq_len=first_bucket,
            tp_degree=int(self.parallel.tp_degree), precomputed_adaln=True,
            num_keyframes=first_keyframes,
        )
        self.config = config
        self.dit_app = NeuronMiniMaxH3TransformerApplication(
            model_path=str(Path(self.model_dir) / "transformer"), config=config,
            parallel=self.parallel, dtype=torch.bfloat16)
        self.dit_app.load()
        self.log(f"DiT loaded on {self.world} ranks in "
                 f"{time.perf_counter() - start:.1f}s ({_mode_name(first_keyframes)} graph, "
                 f"text bucket {first_bucket}), {self.dit_app.weight_bytes / 2**30:.2f} GiB/rank")
        for keyframes, bucket in more:
            # Another graph on the same weights: the shapes come from the call inputs, so the
            # config only steers the warm-up's example inputs.
            start = time.perf_counter()
            config.num_keyframes, config.text_seq_len = keyframes, bucket
            self.dit_app.warmup()
            self.log(f"DiT {_mode_name(keyframes)} graph (text bucket {bucket}) warmed in "
                     f"{time.perf_counter() - start:.1f}s")
        neuron = getattr(torch, "neuron", None)
        if len(self.modes) > 1 and hasattr(neuron, "max_memory_allocated"):
            self.log(f"DiT HBM after the {len(self.modes)} graphs: "
                     f"{neuron.max_memory_allocated() / 2**30:.2f} GiB peak (rank 0)")
        start = time.perf_counter()
        vae_app = NeuronMiniMaxH3VideoVaeApplication(
            model_path=str(Path(self.model_dir) / "vae"), parallel=self.parallel,
            dtype=self.vae_dtype, tp_shard=True)
        vae_app.load()
        self.vae = DeviceTileDecoder(vae_app)
        self.log(f"video decoder ({self.vae_dtype}, TP-sharded) loaded in "
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
            self.log("the text encoder does not fit in HBM beside the DiT; "
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
            return self._conditioner().encode(prompt, seq_len=self.modes[0])

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

    def _conditioner(self):
        """Rank 0's host Qwen3-VL, one copy for every mode: with its vision tower when any mode
        takes keyframes, padded per call to the request's bucket."""
        if self._host_encoder is None:
            from difflet.models.minimax_h3.host_components import (
                HostQwen3VLTextEncoder,
                default_threads,
            )

            start = time.perf_counter()
            self._host_encoder = HostQwen3VLTextEncoder(
                self.model_dir, seq_len=self.text_bucket, threads=default_threads(),
                vision=self.fl2va).load()
            self.log(f"host conditioner{' (with the vision tower)' if self.fl2va else ''} "
                     f"loaded in {time.perf_counter() - start:.1f}s")
        return self._host_encoder

    def encode_fl2va(self, prompt: str, first_image=None, last_image=None) -> dict[str, Any]:
        """The FL2VA presentation and keyframe conditions on every rank. Rank 0 prepares the
        keyframes onto this resident's canvas and encodes them on its host; the other ranks'
        images are ignored. Returns the ``encode`` record plus ``text_token_tags``,
        ``condition_latents`` (one ``(1, 24, 1, h, w)`` per keyframe) and ``keyframe_anchors``."""
        from difflet.backends.neuron.core.distributed import broadcast_tensor, rank0_call

        device = self.dit_app.device

        def encode() -> dict[str, Any]:
            from difflet.models.minimax_h3.host_components import (
                default_threads,
                load_host_video_encoder,
            )
            from difflet.models.minimax_h3.keyframes import keyframe_pixels, prepare_keyframes

            keyframes, anchors, _, _ = prepare_keyframes(
                first_image, last_image, height=self.height, width=self.width)
            if len(keyframes) not in self.modes:
                raise ValueError(f"this resident serves {self.mode_names()}; the request has "
                                 f"{len(keyframes)} keyframe(s)")
            encoder = self._conditioner()
            if self._host_video_encoder is None:
                self._host_video_encoder = load_host_video_encoder(
                    self.model_dir, threads=default_threads())
            start = time.perf_counter()
            record = encoder.encode_presentation(prompt, keyframes,
                                                 seq_len=self.modes[len(keyframes)])
            record["condition_latents"] = torch.cat(
                [self._host_video_encoder(keyframe_pixels(k)) for k in keyframes])
            record["keyframe_anchors"] = anchors
            self.log(f"FL2VA presentation ({record['num_text_tokens']} rows, "
                     f"{', '.join(anchors)} keyframe) and conditions encoded on the host in "
                     f"{time.perf_counter() - start:.1f}s")
            return record

        record = rank0_call(encode, device=device, what="host FL2VA encode")
        have = record is not None

        def bcast(value):
            return broadcast_tensor(value if have else None, device=device).cpu()

        hidden = bcast(record["encoder_hidden_states"] if have else None)
        mask = bcast(record["attention_mask"].to(torch.int32) if have else None).to(torch.bool)
        count = int(bcast(torch.tensor([int(record["num_text_tokens"])], dtype=torch.int32)
                          if have else None).item())
        tags = bcast(record["text_token_tags"].to(torch.int32) if have else None).long()
        conditions = bcast(record["condition_latents"].float() if have else None)
        anchor_codes = bcast(torch.tensor([("first", "last").index(a)
                                           for a in record["keyframe_anchors"]],
                                          dtype=torch.int32) if have else None)
        return {
            "encoder_hidden_states": hidden, "attention_mask": mask, "num_text_tokens": count,
            "text_encoder_layer": record["text_encoder_layer"] if have else None,
            "text_token_tags": tags,
            "condition_latents": list(conditions.split(1)),
            "keyframe_anchors": tuple(("first", "last")[int(c)] for c in anchor_codes.tolist()),
        }

    def mode_names(self) -> str:
        return ", ".join(_mode_name(k) for k in self.modes)

    def modulation_table(self, steps: int, keyframes: bool = False) -> torch.Tensor:
        """The precomputed AdaLN table of a ``steps``-step schedule: built once on rank 0 into
        ``<cache_dir>/minimax_h3_neuron/`` (the file the CLI writes too), read by every rank."""
        steps = int(steps)
        key = (steps, bool(keyframes))
        if key not in self._tables:
            from difflet.backends.neuron.core.distributed import rank0_call
            from difflet.models.minimax_h3.pipeline import (
                build_adaln_modulation_table,
                load_minimax_h3_schedulers,
            )

            from difflet.models.minimax_h3.pipeline import step_timestep_slots

            root = Path(self.cache_dir or Path.home() / ".cache" / "difflet")
            keyframes = bool(keyframes)
            suffix = "_fl2va" if keyframes else ""
            path = root / "minimax_h3_neuron" / f"adaln_table_steps{steps}{suffix}.pt"
            config = self.config

            def build() -> None:
                if path.exists():
                    return
                video_sched, audio_sched = load_minimax_h3_schedulers(self.model_dir)
                video_sched.set_timesteps(steps, device="cpu")
                audio_sched.set_timesteps(steps, device="cpu")
                # Slot order matches the denoise loop: 0 audio, 1 video, (FL2VA) 2 keyframes.
                timesteps = step_timestep_slots(video_sched.timesteps, audio_sched.timesteps,
                                                keyframes=keyframes)
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
            self._tables[key] = torch.load(path)
        return self._tables[key]

    def agree_to_stop(self, stop: bool) -> bool:
        """Whether any rank wants to stop: one all-reduce(MAX) of a flag, so every rank leaves
        the loop at the same step and no collective is left waiting for a peer."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()) or self.world == 1:
            return bool(stop)
        flag = torch.tensor([1 if stop else 0], dtype=torch.int32, device=self.dit_app.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(int(flag.cpu().item()))

    def agree_on_keyframes(self, count: int) -> int:
        """The request's keyframe count, checked equal on every rank (one all-reduce): the ranks
        must run the same graph, or the collectives of two graphs would interleave."""
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()) or self.world == 1:
            return int(count)
        both = torch.tensor([count, -count], dtype=torch.int32, device=self.dit_app.device)
        dist.all_reduce(both, op=dist.ReduceOp.MAX)
        high, low = int(both[0].item()), -int(both[1].item())
        if high != low:
            raise ValueError(f"the ranks disagree on the request's keyframe count ({low}..{high})")
        return high

    def generate(self, prompt: str, *, steps: int, seed: int,
                 scratch_dir: str | Path, should_stop=None, first_image=None,
                 last_image=None, num_keyframes: int | None = None) -> ResidentRequestResult:
        """One request on every rank. ``should_stop()`` (this rank's cancellation/deadline) is
        polled after the text encode and every ``DIFFLET_H3_CANCEL_CHECK_STEPS`` denoise steps;
        once any rank says stop, every rank raises ``ResidentRequestCancelled`` together.

        ``first_image`` / ``last_image`` (PIL images, read on rank 0 only) pick the request's
        mode. Every rank must name the same keyframe count: pass ``num_keyframes`` on ranks that
        do not hold the images (a serving worker hands every rank the request, images included).
        """
        from difflet.backends.neuron.minimax_h3.vae import decode_video_on_ranks
        from difflet.models.minimax_h3.pipeline import (
            decode_minimax_h3_audio,
            denoise_minimax_h3_t2va,
            load_minimax_h3_schedulers,
        )

        seconds: dict[str, float] = {}
        if num_keyframes is None:
            num_keyframes = sum(image is not None for image in (first_image, last_image))
        keyframes = self.agree_on_keyframes(int(num_keyframes))
        if keyframes not in self.modes:
            raise ValueError(f"this resident serves {self.mode_names()}; the request has "
                             f"{keyframes} keyframe(s)")
        bucket = self.modes[keyframes]
        start = time.perf_counter()
        text = (self.encode_fl2va(prompt, first_image, last_image) if keyframes
                else self.encode(prompt))
        seconds["text"] = time.perf_counter() - start
        live = int(text["num_text_tokens"])
        if live > bucket:
            raise ValueError(f"the prompt has {live} tokens; this mode's text bucket is "
                             f"{bucket}")
        table = self.modulation_table(steps, keyframes=keyframes > 0)
        video_scheduler, audio_scheduler = load_minimax_h3_schedulers(self.model_dir)
        every = cancel_check_steps()
        total = int(steps)

        def check(where: str) -> None:
            if should_stop is not None and self.agree_to_stop(bool(should_stop())):
                self.log(f"request cancelled ({where})")
                raise ResidentRequestCancelled(where)

        def after_step(index: int) -> None:
            if (index + 1) % every == 0 and index + 1 < total:
                check(f"after denoise step {index + 1}/{total}")

        check("after the text encode")
        start = time.perf_counter()
        output = denoise_minimax_h3_t2va(
            NeuronH3DiTCall(self.dit_app),
            encoder_hidden_states=text["encoder_hidden_states"][:, :bucket],
            encoder_attention_mask=text["attention_mask"][:, :bucket],
            num_text_tokens=live, height=self.height, width=self.width,
            num_frames=self.num_frames, num_inference_steps=int(steps), seed=int(seed),
            video_scheduler=video_scheduler, audio_scheduler=audio_scheduler,
            model_dtype=torch.bfloat16, adaln_modulation_table=table,
            step_callback=after_step if should_stop is not None else None,
            condition_latents=text.get("condition_latents"),
            keyframe_anchors=tuple(text.get("keyframe_anchors", ())),
            text_token_tags=text.get("text_token_tags"),
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
        self.log("request: "
                 + ", ".join(f"{k} {v:.1f}s" for k, v in seconds.items()))
        return ResidentRequestResult(video=video, audio=audio, sample_rate=sample_rate,
                                     seconds=seconds)

    def shutdown(self) -> None:
        self._host_encoder = None
        self._host_video_encoder = None
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
