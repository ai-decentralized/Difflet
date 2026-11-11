"""MiniMax-H3 T2VA orchestrator — four sequential Trainium stages.

Stage 1 (text):      Qwen3-VL layer-50 conditioner, TP4.
Stage 2 (generate):  33B H3 Omni Transformer,          TP4.
Stage 3 (video_vae): H3 visual VAE decoder,             1 core initially.
Stage 4 (audio_vae): H3 audio VAE decoder and AV mux,   1 core initially.

Only light tokenization, scheduling, tensor layout, normalization and media
post-processing remain on the host.  Staging is required by HBM capacity, not
because the heavy components are intended to execute on the host.

With DIFFLET_BACKEND naming the non-AoT torchrun backend (neuron, TorchNeuron), the same four
stages and work files run as follows (phase 1 of the port):

* text: Qwen3-VL to hidden_states[50] on the host CPU (``HostQwen3VLTextEncoder``), one rank;
* generate: the DiT on ``--tp-degree`` torchrun ranks, one NeuronCore each
  (``NeuronMiniMaxH3TransformerApplication``, per-block compiled, AdaLN precomputed on the
  host); every rank runs the same host denoise loop, rank 0 writes the latents;
* video_vae: the ViT decoder on one NeuronCore (``NeuronMiniMaxH3VideoVaeApplication``, one
  256 px tile per call, ``--vae-dtype`` fp32 by default or bf16), or on the host in fp32 with
  ``--host-vae``;
* audio_vae: the audio decoder on the host CPU in fp32, one rank.

``compile`` then warms up the device stages only (the DiT and, unless ``--host-vae``, the video
decoder), which fills the NEFF cache; the host stages have nothing to compile.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from difflet.cli import runner
from difflet.cli.orchestrators.base import ModelOrchestrator
from difflet.cli.orchestrators.torchrun import (
    decodes_on_device,
    refuse_torchrun_conflicts,
    root_compile_cache_at_cache_dir,
    torchrun_backend,
    vae_dtype_conflict,
)
from difflet.common.orchestrators import minimax_h3 as h3_common

if TYPE_CHECKING:
    import torch

_HF_MODEL_ID = h3_common.HF_MODEL_ID
_MODEL_TYPE = h3_common.MODEL_TYPE
_CLI_NAME = h3_common.CLI_NAME
_VIRTUAL_CORE_SIZE = h3_common.VIRTUAL_CORE_SIZE
_STAGES = ("text", "generate", "video_vae", "audio_vae")


def _save_av(
    video: "torch.Tensor",
    audio: "torch.Tensor",
    output_path: str,
    *,
    fps: int = 24,
    sample_rate: int = 32000,
) -> bool:
    """Encode BCTHW RGB plus BCS audio to MP4 through temporary lossless handoff."""

    if Path(output_path).suffix.lower() != ".mp4":
        return False
    try:
        import numpy as np
        from diffusers.utils import export_to_video
    except ImportError:
        return False

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="difflet-h3-av-") as temp_dir:
        silent_path = Path(temp_dir) / "video.mp4"
        audio_path = Path(temp_dir) / "audio.wav"
        frames = video.detach().float().clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy()
        samples = audio.detach().float().clamp(-1, 1)[0].transpose(0, 1).cpu().numpy()
        try:
            export_to_video(list(frames.astype("float32")), str(silent_path), fps=fps)
            pcm = np.rint(samples * 32767.0).astype("<i2")
            with wave.open(str(audio_path), "wb") as wav:
                wav.setnchannels(int(pcm.shape[1]))
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm.tobytes())
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(silent_path),
                    "-i",
                    str(audio_path),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(destination),
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            print(f"[minimax-h3] AV export failed ({exc}); saving tensors instead", flush=True)
            return False
    print(f"[minimax-h3] video with audio saved to {destination}", flush=True)
    return True


class _NeuronH3DiT:
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


class MiniMaxH3Orchestrator(ModelOrchestrator):
    def download(self) -> None:
        from difflet.pipeline.path_resolver import resolve_model_path
        from difflet.registry import resolve_model

        entry = resolve_model(_HF_MODEL_ID, model_type=_MODEL_TYPE)
        resolve_model_path(
            _HF_MODEL_ID,
            revision=self.args.revision,
            local_files_only=False,
            allow_patterns=entry.download_patterns,
        )
        print(f"[difflet] weights ready for {_HF_MODEL_ID}")

    def compile(self) -> None:
        self._validate_topology()
        work_dir = Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        work_dir.mkdir(parents=True, exist_ok=True)
        shared = self._shared_cli_args(stage_mode="compile", work_dir=str(work_dir))
        stages = _STAGES
        if torchrun_backend() is not None:
            refuse_torchrun_conflicts([c for c in (vae_dtype_conflict(self.args),) if c])
            root_compile_cache_at_cache_dir(self.args.cache_dir, tag="minimax-h3")
            # The host stages have nothing to compile.
            stages = ("generate", "video_vae") if self._device_vae() else ("generate",)
        for stage in stages:
            runner.run_stage(
                _HF_MODEL_ID,
                stage,
                num_cores=self._stage_cores(stage),
                virtual_core_size=_VIRTUAL_CORE_SIZE,
                cli_args=shared,
            )

    def generate(self) -> None:
        self._validate_topology()
        work_dir = Path(
            self.args.work_dir or Path.home() / ".cache" / "difflet" / "work" / _CLI_NAME
        )
        work_dir.mkdir(parents=True, exist_ok=True)
        shared = self._shared_cli_args(stage_mode="generate", work_dir=str(work_dir))
        if torchrun_backend() is not None:
            refuse_torchrun_conflicts([c for c in (vae_dtype_conflict(self.args),) if c])
            root_compile_cache_at_cache_dir(self.args.cache_dir, tag="minimax-h3")
        try:
            for stage in _STAGES:
                runner.run_stage(
                    _HF_MODEL_ID,
                    stage,
                    num_cores=self._stage_cores(stage),
                    virtual_core_size=_VIRTUAL_CORE_SIZE,
                    cli_args=shared,
                )
        except Exception:
            print(f"[difflet] work dir preserved at {work_dir} for inspection", file=sys.stderr)
            raise
        if not self.args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_stage_internal(self, stage: str, args: argparse.Namespace) -> None:
        if torchrun_backend() is not None:
            handlers = {
                "text": self._stage_text_host,
                "generate": self._stage_generate_neuron,
                "video_vae": (self._stage_video_vae_neuron if getattr(args, "device_vae", False)
                              else self._stage_video_vae_host),
                "audio_vae": self._stage_audio_vae_host,
            }
        else:
            handlers = {
                "text": self._stage_text,
                "generate": self._stage_generate,
                "video_vae": self._stage_video_vae,
                "audio_vae": self._stage_audio_vae,
            }
        try:
            handler = handlers[stage]
        except KeyError as exc:
            raise ValueError(f"Unknown MiniMax-H3 stage: {stage!r}") from exc
        handler(args)

    def _stage_text(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from neuronx_distributed_inference.models.config import NeuronConfig, TensorCaptureConfig
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl_text import (
            NeuronQwen3VLTextForCausalLM,
        )
        from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        encoder_path = str(Path(model_dir) / "text_encoder")
        compiled_dir = self._stage_compiled_dir("text", args)
        tp_degree = args.tp_degree or 4

        hf_config = AutoConfig.from_pretrained(encoder_path)
        text_config = hf_config.text_config
        if getattr(text_config, "pad_token_id", None) is None:
            text_config.pad_token_id = 0
        if int(getattr(text_config, "num_hidden_layers", 0)) <= h3_common.TEXT_ENCODER_LAYER:
            raise ValueError(
                "MiniMax-H3 requires Qwen3-VL hidden_states[50], but the text encoder "
                f"has only {getattr(text_config, 'num_hidden_layers', None)} layers."
            )

        weights_dir = compiled_dir / "weights"
        shard_paths = [
            weights_dir / f"tp{rank}_sharded_checkpoint.safetensors" for rank in range(tp_degree)
        ]
        save_sharded_checkpoint = args.stage_mode == "compile" or all(
            path.exists() for path in shard_paths
        )
        neuron_config = NeuronConfig(
            tp_degree=tp_degree,
            batch_size=1,
            seq_len=h3_common.TEXT_SEQ_LEN,
            torch_dtype=torch.bfloat16,
            on_device_sampling_config={},
            tensor_capture_config=TensorCaptureConfig(modules_to_capture=["layers.49"]),
            save_sharded_checkpoint=save_sharded_checkpoint,
        )
        config = NeuronQwen3VLTextForCausalLM.get_config_cls()(
            neuron_config,
            load_config=load_pretrained_config(hf_config=text_config),
        )
        app = NeuronQwen3VLTextForCausalLM(encoder_path, config)
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        from difflet.cli.dp import stage_loop

        app.load(str(compiled_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                tokenized = tokenizer(
                    req.prompt,
                    add_special_tokens=False,
                    max_length=h3_common.TEXT_SEQ_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_ids = tokenized.input_ids.to(torch.int32)
                attention_mask = tokenized.attention_mask.to(torch.int32)
                token_count = int(attention_mask.sum().item())
                if token_count < 1:
                    raise ValueError("MiniMax-H3 prompt must produce at least one token.")
                output = app(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=torch.arange(h3_common.TEXT_SEQ_LEN, dtype=torch.int32).unsqueeze(
                        0
                    ),
                    sampling_params=torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
                )
                hidden_states = output.captured_tensors[0].to(torch.bfloat16)
                # Padding rows are never part of the official packed document.
                # Preserve a fixed inter-stage tensor and carry the live length
                # explicitly so the DiT stage can build its masked tail layout.
                hidden_states[:, token_count:] = 0
                destination = stage_loop.work_file(args, req, "text.pt")
                torch.save(
                    {
                        "encoder_hidden_states": hidden_states,
                        "attention_mask": attention_mask.to(torch.bool),
                        "num_text_tokens": token_count,
                        "text_encoder_layer": h3_common.TEXT_ENCODER_LAYER,
                    },
                    destination,
                )
                print(
                    f"[text] Qwen3-VL hidden_states[50] {tuple(hidden_states.shape)} "
                    f"({token_count} live tokens) -> {destination}"
                )

    def _stage_generate(self, args: argparse.Namespace) -> None:
        import torch

        from difflet.backends.trainium.minimax_h3.transformer import (
            NeuronMiniMaxH3TransformerApplication,
        )
        from difflet.models.minimax_h3.application import (
            create_minimax_h3_transformer_config,
        )
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        transformer_path = str(Path(model_dir) / "transformer")
        compiled_dir = self._stage_compiled_dir("generate", args)
        config = create_minimax_h3_transformer_config(
            model_path=model_dir,
            tp_degree=args.tp_degree or 4,
            dtype=torch.bfloat16,
            height=args.height or h3_common.DEFAULT_HEIGHT,
            width=args.width or h3_common.DEFAULT_WIDTH,
            num_frames=args.num_frames or h3_common.DEFAULT_NUM_FRAMES,
            precomputed_adaln=bool(getattr(args, "adaln_precompute", False)),
        )
        app = NeuronMiniMaxH3TransformerApplication(
            model_path=transformer_path,
            config=config,
        )
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        from difflet.cli.dp import stage_loop
        from difflet.models.minimax_h3.pipeline import (
            denoise_minimax_h3_t2va,
            load_minimax_h3_schedulers,
        )

        app.load(str(compiled_dir), skip_warmup=True)
        height = args.height or h3_common.DEFAULT_HEIGHT
        width = args.width or h3_common.DEFAULT_WIDTH
        num_frames = args.num_frames or h3_common.DEFAULT_NUM_FRAMES

        adaln_tables: dict[int, torch.Tensor] = {}

        def _adaln_table(num_steps: int) -> torch.Tensor | None:
            if not bool(getattr(config, "precomputed_adaln", False)):
                return None
            if num_steps not in adaln_tables:
                from difflet.models.minimax_h3.pipeline import build_adaln_modulation_table

                cache_path = Path(compiled_dir) / f"adaln_table_steps{num_steps}.pt"
                if cache_path.exists():
                    adaln_tables[num_steps] = torch.load(cache_path)
                else:
                    video_sched, audio_sched = load_minimax_h3_schedulers(model_dir)
                    video_sched.set_timesteps(num_steps, device="cpu")
                    audio_sched.set_timesteps(num_steps, device="cpu")
                    # Slot order matches the denoise loop: 0 = audio, 1 = video.
                    timesteps = torch.stack(
                        [
                            audio_sched.timesteps.to(torch.float32),
                            video_sched.timesteps.to(torch.float32),
                        ],
                        dim=-1,
                    )
                    table = build_adaln_modulation_table(
                        str(Path(model_dir) / "transformer"),
                        timesteps,
                        num_layers=int(config.num_layers),
                        freq_dim=int(config.freq_dim),
                        time_embed_hidden_dim=int(config.time_embed_hidden_dim),
                        time_embed_dim=int(config.time_embed_dim),
                        hidden_size=int(config.hidden_size),
                        dtype=torch.bfloat16,
                    )
                    torch.save(table, cache_path)
                    adaln_tables[num_steps] = table
                    print(
                        f"[generate] precomputed AdaLN table for {num_steps} steps "
                        f"({table.numel() * table.element_size() / 2**20:.0f} MiB) -> {cache_path}"
                    )
            return adaln_tables[num_steps]

        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                text = torch.load(stage_loop.work_file(args, req, "text.pt"))
                video_scheduler, audio_scheduler = load_minimax_h3_schedulers(model_dir)
                request_steps = int(
                    stage_loop.effective(req, args, "steps", h3_common.DEFAULT_STEPS)
                )
                output = denoise_minimax_h3_t2va(
                    app,
                    encoder_hidden_states=text["encoder_hidden_states"],
                    encoder_attention_mask=text["attention_mask"],
                    num_text_tokens=int(text["num_text_tokens"]),
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=request_steps,
                    seed=req.seed,
                    video_scheduler=video_scheduler,
                    audio_scheduler=audio_scheduler,
                    model_dtype=torch.bfloat16,
                    adaln_modulation_table=_adaln_table(request_steps),
                )
                destination = stage_loop.work_file(args, req, "latents.pt")
                torch.save(
                    {
                        "video_latents": output.video_latents.cpu(),
                        "audio_latents": output.audio_latents.cpu(),
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                    },
                    destination,
                )
                print(
                    f"[generate] video latents {tuple(output.video_latents.shape)}, "
                    f"audio latents {tuple(output.audio_latents.shape)} -> {destination}"
                )

    def _stage_video_vae(self, args: argparse.Namespace) -> None:
        import torch

        from difflet.backends.trainium.minimax_h3.vae import (
            NeuronMiniMaxH3VideoVAEDecoderApplication,
        )
        from difflet.models.minimax_h3.application import create_minimax_h3_video_vae_config
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        vae_path = str(Path(model_dir) / "vae")
        compiled_dir = self._stage_compiled_dir("video_vae", args)
        config = create_minimax_h3_video_vae_config(
            model_path=model_dir,
            height=args.height or h3_common.DEFAULT_HEIGHT,
            width=args.width or h3_common.DEFAULT_WIDTH,
            num_frames=args.num_frames or h3_common.DEFAULT_NUM_FRAMES,
        )
        app = NeuronMiniMaxH3VideoVAEDecoderApplication(
            model_path=vae_path,
            config=config,
        )
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        from difflet.cli.dp import stage_loop
        from difflet.models.minimax_h3.pipeline import decode_minimax_h3_video

        app.load(str(compiled_dir), skip_warmup=True)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"))
                video = decode_minimax_h3_video(
                    app,
                    latents["video_latents"],
                    latents_mean=config.latents_mean,
                    latents_std=config.latents_std,
                    clip_length=int(config.clip_length),
                    token_drop=int(config.token_drop),
                    temporal_compression_ratio=int(config.temporal_compression_ratio),
                    spatial_compression_ratio=int(config.spatial_compression_ratio),
                )
                destination = stage_loop.work_file(args, req, "video.pt")
                torch.save(video.to(torch.float16), destination)
                print(f"[video_vae] RGB video {tuple(video.shape)} -> {destination}")

    def _stage_audio_vae(self, args: argparse.Namespace) -> None:
        import torch

        from difflet.backends.trainium.minimax_h3.vae import (
            NeuronMiniMaxH3AudioVAEDecoderApplication,
        )
        from difflet.models.minimax_h3.application import create_minimax_h3_audio_vae_config
        from difflet.pipeline.path_resolver import resolve_model_path

        model_dir = self._model_dir(args, resolve_model_path)
        vae_path = str(Path(model_dir) / "audio_vae")
        compiled_dir = self._stage_compiled_dir("audio_vae", args)
        config = create_minimax_h3_audio_vae_config(
            model_path=model_dir,
            num_frames=args.num_frames or h3_common.DEFAULT_NUM_FRAMES,
        )
        app = NeuronMiniMaxH3AudioVAEDecoderApplication(
            model_path=vae_path,
            config=config,
        )
        if args.stage_mode == "compile":
            app.compile(str(compiled_dir))
            return

        from difflet.cli.dp import stage_loop
        from difflet.models.minimax_h3.pipeline import decode_minimax_h3_audio

        app.load(str(compiled_dir), skip_warmup=True)
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"))
                video = torch.load(stage_loop.work_file(args, req, "video.pt")).float()
                audio = decode_minimax_h3_audio(
                    app,
                    latents["audio_latents"],
                    latents_mean=config.latents_mean,
                    latents_std=config.latents_std,
                    chunk_latent_frames=int(config.audio_chunk_latent_frames),
                    chunk_core_frames=int(config.audio_chunk_core_frames),
                    hop_length=800,
                )
                output_path = Path(req.output)
                if not _save_av(
                    video,
                    audio,
                    str(output_path),
                    sample_rate=int(config.sampling_rate),
                ):
                    tensor_path = output_path.with_suffix(".pt")
                    tensor_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "video": video.to(torch.float16),
                            "audio": audio,
                            "fps": 24,
                            "sample_rate": int(config.sampling_rate),
                        },
                        tensor_path,
                    )
                    print(f"[audio_vae] AV tensors saved to {tensor_path}")

    # ------------------------------------------------- torchrun (neuron) backend stages

    def _stage_text_host(self, args: argparse.Namespace) -> None:
        """Qwen3-VL hidden_states[50] on the host CPU; writes the Trainium stage's text.pt."""
        import torch

        from difflet.cli.dp import stage_loop
        from difflet.models.minimax_h3.host_components import (
            HostQwen3VLTextEncoder,
            default_threads,
        )
        from difflet.pipeline.path_resolver import resolve_model_path

        if args.stage_mode == "compile":
            return
        model_dir = self._model_dir(args, resolve_model_path)
        encoder = HostQwen3VLTextEncoder(model_dir, threads=default_threads()).load()
        for req in stage_loop.claim_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                record = encoder.encode(req.prompt)
                destination = stage_loop.work_file(args, req, "text.pt")
                torch.save(record, destination)
                print(
                    f"[text] Qwen3-VL hidden_states[{record['text_encoder_layer']}] (host) "
                    f"{tuple(record['encoder_hidden_states'].shape)} "
                    f"({record['num_text_tokens']} live tokens) -> {destination}",
                    flush=True,
                )

    def _neuron_adaln_table(self, args, model_dir: str, config, num_steps: int):
        """The precomputed AdaLN table for ``num_steps``, built once on rank 0 into
        ``<cache-dir>/minimax_h3_neuron/`` and read by every rank."""
        import torch

        from difflet.backends.neuron.core.distributed import rank0_call
        from difflet.models.minimax_h3.pipeline import (
            build_adaln_modulation_table,
            load_minimax_h3_schedulers,
        )

        cache_root = Path(args.cache_dir or Path.home() / ".cache" / "difflet")
        path = cache_root / "minimax_h3_neuron" / f"adaln_table_steps{num_steps}.pt"

        def build() -> None:
            if path.exists():
                return
            video_sched, audio_sched = load_minimax_h3_schedulers(model_dir)
            video_sched.set_timesteps(num_steps, device="cpu")
            audio_sched.set_timesteps(num_steps, device="cpu")
            # Slot order matches the denoise loop: 0 = audio, 1 = video.
            timesteps = torch.stack(
                [audio_sched.timesteps.to(torch.float32), video_sched.timesteps.to(torch.float32)],
                dim=-1,
            )
            table = build_adaln_modulation_table(
                str(Path(model_dir) / "transformer"),
                timesteps,
                num_layers=int(config.num_layers),
                freq_dim=int(config.freq_dim),
                time_embed_hidden_dim=int(config.time_embed_hidden_dim),
                time_embed_dim=int(config.time_embed_dim),
                hidden_size=int(config.hidden_size),
                dtype=torch.bfloat16,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save(table, tmp)
            tmp.replace(path)
            print(f"[generate] precomputed AdaLN table for {num_steps} steps "
                  f"({table.numel() * table.element_size() / 2**20:.0f} MiB) -> {path}", flush=True)

        rank0_call(build, device=self._neuron_device, what="AdaLN table")
        return torch.load(path)

    def _stage_generate_neuron(self, args: argparse.Namespace) -> None:
        """The DiT on every torchrun rank; each runs the host denoise loop on identical inputs
        (one CPU generator per request), rank 0 writes the latents."""
        import time

        import torch

        from difflet.backends import get_backend
        from difflet.cli.dp import stage_loop
        from difflet.cli.orchestrators.torchrun import replica_request_scope, torchrun_parallel
        from difflet.pipeline.path_resolver import resolve_model_path

        parallel = torchrun_parallel(args)
        get_backend("neuron").prepare_runtime(parallel)

        from difflet.backends.neuron.core.distributed import world_info
        from difflet.backends.neuron.minimax_h3.transformer import (
            NeuronMiniMaxH3Config,
            NeuronMiniMaxH3TransformerApplication,
        )
        from difflet.models.minimax_h3.pipeline import (
            denoise_minimax_h3_t2va,
            load_minimax_h3_schedulers,
        )

        model_dir = self._model_dir(args, resolve_model_path)
        transformer_path = str(Path(model_dir) / "transformer")
        height = args.height or h3_common.DEFAULT_HEIGHT
        width = args.width or h3_common.DEFAULT_WIDTH
        num_frames = args.num_frames or h3_common.DEFAULT_NUM_FRAMES
        config = NeuronMiniMaxH3Config.from_pretrained(
            transformer_path, height=height, width=width, num_frames=num_frames,
            text_seq_len=h3_common.TEXT_SEQ_LEN, tp_degree=parallel.tp_degree,
            precomputed_adaln=True,
        )
        app = NeuronMiniMaxH3TransformerApplication(
            model_path=transformer_path, config=config, parallel=parallel, dtype=torch.bfloat16,
        )
        self._neuron_device = app.device
        rank = world_info()[0]
        start = time.perf_counter()
        app.load()
        if rank == 0:
            print(f"[generate] DiT loaded on {app.world_size} ranks in "
                  f"{time.perf_counter() - start:.1f}s "
                  f"({ {k: round(v, 1) for k, v in app.phase_seconds.items()} }), "
                  f"{app.weight_bytes / 2**30:.2f} GiB/rank", flush=True)
        if args.stage_mode == "compile":
            return

        dit = _NeuronH3DiT(app)
        for req in stage_loop.claimed_requests(args):
            with replica_request_scope(args, req, final=False):
                text = torch.load(stage_loop.work_file(args, req, "text.pt"))
                steps = int(stage_loop.effective(req, args, "steps", h3_common.DEFAULT_STEPS))
                table = self._neuron_adaln_table(args, model_dir, config, steps)
                video_scheduler, audio_scheduler = load_minimax_h3_schedulers(model_dir)
                start = time.perf_counter()
                output = denoise_minimax_h3_t2va(
                    dit,
                    encoder_hidden_states=text["encoder_hidden_states"],
                    encoder_attention_mask=text["attention_mask"],
                    num_text_tokens=int(text["num_text_tokens"]),
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=steps,
                    seed=req.seed,
                    video_scheduler=video_scheduler,
                    audio_scheduler=audio_scheduler,
                    model_dtype=torch.bfloat16,
                    adaln_modulation_table=table,
                )
                if rank != 0:
                    continue
                elapsed = time.perf_counter() - start
                destination = stage_loop.work_file(args, req, "latents.pt")
                torch.save(
                    {
                        "video_latents": output.video_latents.cpu(),
                        "audio_latents": output.audio_latents.cpu(),
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                    },
                    destination,
                )
                print(
                    f"[generate] {steps} steps in {elapsed:.1f}s "
                    f"({1000 * sum(dit.seconds) / max(len(dit.seconds), 1):.0f} ms/DiT call); "
                    f"video latents {tuple(output.video_latents.shape)}, audio latents "
                    f"{tuple(output.audio_latents.shape)} -> {destination}",
                    flush=True,
                )

    def _stage_video_vae_host(self, args: argparse.Namespace) -> None:
        import time

        import torch

        from difflet.cli.dp import stage_loop
        from difflet.models.minimax_h3.host_components import (
            default_threads,
            load_host_video_decoder,
        )
        from difflet.models.minimax_h3.pipeline import decode_minimax_h3_video
        from difflet.pipeline.path_resolver import resolve_model_path

        if args.stage_mode == "compile":
            return
        model_dir = self._model_dir(args, resolve_model_path)
        decoder = load_host_video_decoder(model_dir, threads=default_threads())
        cfg = decoder.config
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"))
                start = time.perf_counter()
                video = decode_minimax_h3_video(
                    decoder,
                    latents["video_latents"],
                    latents_mean=cfg["latents_mean"],
                    latents_std=cfg["latents_std"],
                    clip_length=int(cfg["clip_length"]),
                    token_drop=int(cfg["token_drop"]),
                )
                destination = stage_loop.work_file(args, req, "video.pt")
                torch.save(video.to(torch.float16), destination)
                print(f"[video_vae] (host) RGB video {tuple(video.shape)} in "
                      f"{time.perf_counter() - start:.1f}s -> {destination}", flush=True)

    def _device_vae(self) -> bool:
        """The neuron backend decodes the video on a NeuronCore unless --host-vae."""
        return decodes_on_device(self.args, default=True)

    def _stage_video_vae_neuron(self, args: argparse.Namespace) -> None:
        """The video decoder on one NeuronCore, one tile per call; ``compile`` loads it with
        the warm-up decode (which compiles the tile), ``generate`` skips the warm-up."""
        import time

        import torch

        from difflet.backends import get_backend
        from difflet.cli.dp import stage_loop
        from difflet.cli.orchestrators.torchrun import component_dtype
        from difflet.pipeline.parallel_config import DiffletParallelConfig
        from difflet.pipeline.path_resolver import resolve_model_path

        get_backend("neuron").prepare_runtime(DiffletParallelConfig(tp_degree=1))
        from difflet.backends.neuron.minimax_h3.vae import (
            DeviceTileDecoder,
            NeuronMiniMaxH3VideoVaeApplication,
        )
        from difflet.models.minimax_h3.pipeline import decode_minimax_h3_video

        model_dir = self._model_dir(args, resolve_model_path)
        dtype = getattr(torch, component_dtype(getattr(args, "vae_dtype", None)))
        app = NeuronMiniMaxH3VideoVaeApplication(model_path=str(Path(model_dir) / "vae"),
                                                 dtype=dtype)
        start = time.perf_counter()
        app.load(skip_warmup=args.stage_mode != "compile")
        print(f"[video_vae] device decoder ({dtype}) loaded in "
              f"{time.perf_counter() - start:.1f}s", flush=True)
        if args.stage_mode == "compile":
            return
        decoder = DeviceTileDecoder(app)
        cfg = app.vae_config
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=False):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"))
                start = time.perf_counter()
                video = decode_minimax_h3_video(
                    decoder,
                    latents["video_latents"],
                    latents_mean=cfg["latents_mean"],
                    latents_std=cfg["latents_std"],
                    clip_length=int(cfg["clip_length"]),
                    token_drop=int(cfg["token_drop"]),
                )
                destination = stage_loop.work_file(args, req, "video.pt")
                torch.save(video.to(torch.float16), destination)
                print(f"[video_vae] (device) RGB video {tuple(video.shape)} in "
                      f"{time.perf_counter() - start:.1f}s -> {destination}", flush=True)

    def _stage_audio_vae_host(self, args: argparse.Namespace) -> None:
        import time

        import torch

        from difflet.cli.dp import stage_loop
        from difflet.models.minimax_h3.host_components import (
            default_threads,
            load_host_audio_decoder,
        )
        from difflet.models.minimax_h3.pipeline import decode_minimax_h3_audio
        from difflet.pipeline.path_resolver import resolve_model_path

        if args.stage_mode == "compile":
            return
        model_dir = self._model_dir(args, resolve_model_path)
        decoder = load_host_audio_decoder(model_dir, threads=default_threads())
        cfg = decoder.config
        for req in stage_loop.claimed_requests(args):
            with stage_loop.request_scope(args, req, final=True):
                latents = torch.load(stage_loop.work_file(args, req, "latents.pt"))
                video = torch.load(stage_loop.work_file(args, req, "video.pt")).float()
                start = time.perf_counter()
                # The whole latent in one decode (the reference); the Trainium graph decodes
                # fixed 48-frame windows only because its shape is static.
                audio = decode_minimax_h3_audio(
                    decoder,
                    latents["audio_latents"],
                    latents_mean=cfg["latents_mean"],
                    latents_std=cfg["latents_std"],
                    hop_length=800,
                )
                print(f"[audio_vae] (host) waveform {tuple(audio.shape)} in "
                      f"{time.perf_counter() - start:.1f}s", flush=True)
                output_path = Path(req.output)
                if not _save_av(video, audio, str(output_path),
                                sample_rate=int(cfg["sampling_rate"])):
                    tensor_path = output_path.with_suffix(".pt")
                    tensor_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"video": video.to(torch.float16), "audio": audio, "fps": 24,
                                "sample_rate": int(cfg["sampling_rate"])}, tensor_path)
                    print(f"[audio_vae] AV tensors saved to {tensor_path}", flush=True)

    def _validate_topology(self) -> None:
        if (self.args.tp_degree or 4) != 4:
            raise NotImplementedError("MiniMax-H3's first qualified graph is fixed at TP4.")
        if (self.args.cp_degree or 1) != 1:
            raise NotImplementedError("MiniMax-H3 CP is deferred until full-attention parity.")
        if getattr(self.args, "cfg_parallel", False):
            raise NotImplementedError("MiniMax-H3 is guidance-distilled and has no CFG branch.")
        if getattr(self.args, "sp_enabled", False):
            raise NotImplementedError(
                "MiniMax-H3 SP is deferred until the dense TP4 graph is qualified."
            )
        if getattr(self.args, "tp_reduce_dtype", None) is not None:
            raise NotImplementedError(
                "--tp-reduce-dtype is not supported for MiniMax-H3; its all-reduces "
                "keep the NxD fp32 default."
            )

    def _stage_cores(self, stage: str) -> int:
        if torchrun_backend() is not None:
            # Only the DiT is on the device; the host stages run as one rank.
            return (self.args.tp_degree or 4) if stage == "generate" else 1
        return (self.args.tp_degree or 4) if stage in {"text", "generate"} else 1

    @staticmethod
    def _model_dir(args: argparse.Namespace, resolver) -> str:
        model_path = getattr(args, "model_path", None)
        if model_path:
            return str(Path(model_path).expanduser().resolve())
        return resolver(_HF_MODEL_ID, revision=args.revision, local_files_only=True)

    def _stage_compiled_dir(self, stage: str, args: argparse.Namespace) -> Path:
        compiled_dir = getattr(args, "compiled_dir", None)
        if compiled_dir:
            return Path(compiled_dir)
        return h3_common.stage_compiled_dir_from_values(
            stage,
            cache_dir=args.cache_dir,
            tp_degree=args.tp_degree or 4,
            height=args.height or h3_common.DEFAULT_HEIGHT,
            width=args.width or h3_common.DEFAULT_WIDTH,
            num_frames=args.num_frames or h3_common.DEFAULT_NUM_FRAMES,
            adaln_precompute=bool(getattr(args, "adaln_precompute", False)),
        )

    def _shared_cli_args(self, stage_mode: str, work_dir: str | None = None) -> list[str]:
        args = self.args
        parts = [
            "--model-id",
            _HF_MODEL_ID,
            "--tp-degree",
            str(args.tp_degree or 4),
            "--cp-degree",
            str(args.cp_degree or 1),
            "--height",
            str(args.height or h3_common.DEFAULT_HEIGHT),
            "--width",
            str(args.width or h3_common.DEFAULT_WIDTH),
            "--num-frames",
            str(args.num_frames or h3_common.DEFAULT_NUM_FRAMES),
            "--steps",
            str(getattr(args, "steps", None) or h3_common.DEFAULT_STEPS),
            "--seed",
            str(getattr(args, "seed", 42)),
            "--stage-mode",
            stage_mode,
        ]
        for flag, value in (
            ("--prompt", getattr(args, "prompt", None)),
            ("--output", getattr(args, "output", None)),
            ("--cache-dir", getattr(args, "cache_dir", None)),
            ("--revision", getattr(args, "revision", None)),
            ("--model-path", getattr(args, "model_path", None)),
            ("--work-dir", work_dir),
        ):
            if value:
                parts.extend([flag, str(value)])
        if torchrun_backend() is not None and self._device_vae():
            parts.append("--device-vae")
            if getattr(args, "vae_dtype", None):
                parts.extend(["--vae-dtype", str(args.vae_dtype)])
        if getattr(args, "requests_dir", None):
            parts.extend(
                [
                    "--requests-dir",
                    str(args.requests_dir),
                    "--worker-index",
                    str(args.worker_index),
                    "--dp-schedule",
                    str(getattr(args, "dp_schedule", "round_robin")),
                ]
            )
        return parts
