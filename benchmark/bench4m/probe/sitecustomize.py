"""Observer probe for the HunyuanVideo TP4 benchmark (not part of the repo).

The campaign probe (scripts/probe/sitecustomize.py) named in benchmark/trn2/neuron_final.md is
not on this host (searched /home/ubuntu/campaign, every git worktree, and `git log --all --
'*sitecustomize*'`), so this one stands in for it. It only records timestamps: every wrapper
calls the original with the same arguments and returns its result unchanged; nothing else is
touched. Active only when HV1_PROBE_OUT names a directory; it is put first on PYTHONPATH, so it
also loads in every torchrun rank (the env is inherited).

Wrapped (class or module level), where the module is imported:
  difflet.models.hunyuan_video.pipeline: HunyuanVideoOrchestrator._scheduler_step (a stamp
      after each step), ._denoise, ._decode_latents (the AoT generate stage's host decode)
  difflet.models.hunyuan_video.neuron_application: TorchNeuronHunyuanVideoApplication
      .encode_prompt, .load (and, after load, a read of its own `phase_seconds` property;
      added at 02:50 UTC, after the tp4_default and tp4_devvae cells had run)
  difflet.models.hunyuan_video.application: NeuronHunyuanVideoApplication.load (AoT DiT)
  difflet.backends.neuron.hunyuan_video.vae: NeuronHunyuanVideoVaeDecoderApplication.load,
      .decode
  difflet.cli.orchestrators.hunyuan_video: _write_video (neuron: frames to mp4),
      _save_video (AoT), _decode_latents_host (neuron host decode, load + decode + write)
Added 2026-10-25 for the Qwen-Image TP4 stage (additive; the HunyuanVideo targets are unchanged;
the file exactly as the HunyuanVideo cells used it is sitecustomize.hv1-as-used-2026-10-25.py):
  difflet.models.qwen_image.pipeline: QwenImageOrchestrator._scheduler_step (a stamp after each
      step, native ranks and the AoT generate stage alike), ._denoise, ._decode_latents
  difflet.models.qwen_image.neuron_application: TorchNeuronQwenImageApplication.encode_prompt,
      .load (+ the read-only `phase_seconds`), _DeviceQwenVaeDecoder.__call__ (vae decode)
  difflet.models.qwen_image.application: NeuronQwenImageApplication.load (AoT DiT)
  difflet.backends.neuron.wan.vae: NeuronWanVaeDecoderApplication.load (the vae stage decoder)
  difflet.cli.orchestrators.qwen_image: _write_image, _decode_latents_host, and the CLI class's
      _stage_text / _stage_generate / _stage_vae (one span per stage process)
Added 2026-10-25 for the Wan2.1 TP4 stage (additive; the earlier targets are unchanged; the file
exactly as the Qwen-Image cells used it is sitecustomize.qwen-as-used-2026-10-25.py):
  difflet.models.wan.pipeline: WanOrchestrator._scheduler_step (a stamp after each step, native
      ranks and the AoT transformer stage alike), ._denoise, ._decode_latents, .encode_prompt
  difflet.models.wan.neuron_application: TorchNeuronWanApplication.load (+ the read-only
      `phase_seconds`), _DeviceVaeDecoder.__call__ (vae decode)
  difflet.models.wan.application: NeuronWanApplication.load (AoT)
  difflet.cli.orchestrators.wan: _write_frames, _decode_latents_host, _save_video, and the CLI
      class's _stage_transformer / _stage_vae (one span per stage process)
Added 2026-10-26 for the four-model native vs AoT stage (bench-2026-10-26-4m; additive, every
earlier target unchanged; the file as copied is sitecustomize.as-copied-from-2026-10-25.py):
  AoT on origin/main (difflet/backends/trainium):
  difflet.backends.trainium.wan.vae: NeuronWanVAEDecoderApplication.load / __call__ (the AoT
      Qwen-Image vae stage's decoder)
  neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text:
      NeuronQwen2VLTextForCausalLM.load (the AoT Qwen-Image text stage's encoder load)
  difflet.models.ltx_2.pipeline: LTX2Orchestrator._scheduler_step (called twice per step, video
      then audio: a "scheduler_step" stamp after every second call of a _denoise, i.e. at the
      end of each step), ._denoise, ._decode_latents, .prepare_conditioning (text encode)
  difflet.models.ltx_2.application: NeuronLTX2Application.load (DiT load) and the module-level
      _load_ltx_2_host_pipeline (Gemma-3 + connectors + VAE/audio VAE/vocoder on the host)
  difflet.cli.orchestrators.ltx_2: _save_video, LTX2Orchestrator._load_pipeline / .generate
  difflet.cli.prewarm: prewarm_neuron_runtime
  A failure inside a patch is recorded as a "probe_patch_error" event instead of breaking the
  import (a target class renamed on a branch must not change the run).
Added 2026-10-26, phase 2 (native Qwen-Image hash check; additive, every earlier target
unchanged; the file as phase 1 used it is sitecustomize.as-used-phase1-2026-10-26.py):
  Only when HV1_DUMP_LATENTS_DIR names a directory (set by the harness for one run only):
  difflet.models.qwen_image.pipeline: QwenImageOrchestrator._denoise gets an outer wrapper that,
      on rank 0 (RANK "0", or unset), saves bundle.hidden_states (the packed latents entering the
      denoise loop) to <dir>/initial.pt and the returned latents to <dir>/final.pt, both as
      detached CPU clones (torch.save), and records "latent_dump" events with dtype and shape.
      The call's arguments and result are unchanged.
  difflet.models.qwen_image.neuron_application: TorchNeuronQwenImageApplication._initial_latents
      gets a wrapper that records the drawn noise's dtype, shape and device ("initial_latents"
      event, every rank); the result is returned unchanged.
Added 2026-10-26, phase 2 (HunyuanVideo AoT on origin/main; additive, every earlier target
unchanged; the file as the native Qwen-Image hash check used it is
sitecustomize.as-used-qwen-native-hashcheck-2026-10-26.py):
  difflet.cli.orchestrators.hunyuan_video: HunyuanVideoOrchestrator._stage_clip / _stage_llama /
      _stage_generate (one span per stage process)
  difflet.backends.trainium.hunyuan_video.vae: NeuronHunyuanVideoVAEDecoderApplication.decode
  difflet.models.flux.clip.modeling_clip: NeuronClipApplication.load (the clip stage's load)
  neuronx_distributed_inference.models.llama.modeling_llama: NeuronLlamaForCausalLM.load (the
      llama stage's load)
  Only when HV1_DUMP_LATENTS_DIR is set: difflet.models.hunyuan_video.pipeline:
      HunyuanVideoOrchestrator._denoise gets the same outer dump wrapper as Qwen-Image (rank 0 or
      a process without RANK: bundle.hidden_states -> initial.pt, the returned latents ->
      final.pt). Main has no latent dump for HunyuanVideo (difflet/utils/latent_dump.py is used
      only by qwen_image and ltx_2), so this is the only record of the AoT noise.
Added 2026-10-26, phase 2 (Wan2.1 AoT on origin/main; additive, every earlier target unchanged;
the file as the HunyuanVideo AoT cell used it is sitecustomize.as-used-hv-aot-2026-10-26.py):
  Only when HV1_DUMP_LATENTS_DIR is set: difflet.models.wan.pipeline: WanOrchestrator._denoise
      gets an outer dump wrapper (rank 0 or a process without RANK): its keyword `latents` (the
      fp32 noise entering the loop) -> initial.pt, the returned latents -> final.pt. Main has no
      latent dump for Wan (difflet/utils/latent_dump.py is used only by qwen_image and ltx_2).
Added 2026-10-26, phase 2 (LTX-2 native full cell at the hunyuan-neuron HEAD; additive, every
earlier target unchanged; the file as the Wan2.1 AoT and HV/Wan native runs used it is
sitecustomize.as-used-wan-aot-hvwan-native-2026-10-26.py):
  difflet.models.ltx_2.neuron_application: TorchNeuronLtx2Application.load ("app_load" + the
      read-only phase_seconds as an "app_phase_seconds" event), .encode_prompt ("encode_prompt"),
      ._initial_latents (an "initial_latents" event: dtype/shape/device of the video and audio
      noise, every rank; the result is returned unchanged); _DeviceLtx2TextEncoder.load
      ("text_enc_load": tokenizer + device Gemma-3 + rank 0's connectors), ._load_connectors
      ("connectors_load", rank 0), ._connect ("connectors", rank 0); the module-level
      _host_ltx2_pipeline ("host_pipeline_load": the driver's host decode components)
  difflet.backends.neuron.ltx_2.transformer: NeuronLtx2TransformerApplication.load ("dit_load"
      + a "component_phase_seconds" event, component "dit")
  difflet.backends.neuron.ltx_2.text_encoder: NeuronLtx2TextEncoderApplication.load
      ("gemma_load" + a "component_phase_seconds" event, component "gemma")
  difflet.cli.orchestrators.ltx_2: _decode_latents_host ("host_decode": the driver's host decode,
      load + decode + write)
  difflet.cli.runner: run_stage ("run_stage": one span per stage launch, in the driver)
  Only when HV1_DUMP_LATENTS_DIR is set: LTX2Orchestrator._denoise gets an outer dump wrapper
      (rank 0, or a process without RANK): bundle.hidden_states -> initial_video.pt,
      bundle.audio_hidden_states -> initial_audio.pt, the returned (video, audio) ->
      final_video.pt / final_audio.pt (main's DIFFLET_DUMP_LATENTS_DIR names and tensors; the
      native branch has no latent dump). The bench harness sets it only on the native LTX-2 q run.
At exit each process writes <out>/<pid>-rank<r>.json: argv, t0 (sitecustomize import, i.e.
interpreter start), t_exit, the events, the imported difflet package's __file__ (added
2026-10-26), and on a neuron rank torch_neuronx.max_memory_allocated() in GiB.
"""
import atexit
import importlib.abc
import importlib.util
import json
import os
import sys
import time

_OUT = os.environ.get("HV1_PROBE_OUT")
_EVENTS = []
_T0 = time.time()


def _ev(name, **kw):
    _EVENTS.append({"t": time.time(), "name": name, **kw})


_DUMP = os.environ.get("HV1_DUMP_LATENTS_DIR")  # phase 2, 2026-10-26


def _save_latent(fname, t):
    """Save a detached CPU clone of tensor ``t`` to <HV1_DUMP_LATENTS_DIR>/<fname>; never raises."""
    try:
        import torch

        c = t.detach().to("cpu").clone()
        os.makedirs(_DUMP, exist_ok=True)
        path = os.path.join(_DUMP, fname)
        torch.save(c, path)
        _ev("latent_dump", file=path, dtype=str(t.dtype), shape=list(t.shape),
            device=str(t.device))
    except Exception as exc:  # noqa: BLE001
        _ev("latent_dump_error", file=fname, error=repr(exc))


def _wrap_load_phases(owner, label, component):
    """Span ``label`` around owner.load, then a read of its own phase_seconds (read-only)."""
    orig = getattr(owner, "load", None)
    if orig is None:
        return

    def load(self, *a, **k):
        _ev(label + ":start")
        try:
            return orig(self, *a, **k)
        finally:
            _ev(label + ":end")
            try:
                _ev("component_phase_seconds", component=component,
                    phases={k2: round(float(v), 2) for k2, v in self.phase_seconds.items()})
            except Exception as exc:  # noqa: BLE001
                _ev("component_phase_seconds_error", component=component, error=repr(exc))

    owner.load = load


def ltx_dump_wrapper(orig):
    """The LTX-2 _denoise dump wrapper (module level so the CPU self-test can exercise it)."""

    def ltx_den_dump(self, *a, **k):
        rank0 = os.environ.get("RANK") in ("0", None)
        bundle = k.get("bundle") if "bundle" in k else (a[0] if a else None)
        if rank0 and bundle is not None:
            _save_latent("initial_video.pt", bundle.hidden_states)
            _save_latent("initial_audio.pt", bundle.audio_hidden_states)
        out = orig(self, *a, **k)
        if rank0:
            try:
                _save_latent("final_video.pt", out[0])
                _save_latent("final_audio.pt", out[1])
            except Exception as exc:  # noqa: BLE001
                _ev("latent_dump_error", file="final_*", error=repr(exc))
        return out

    return ltx_den_dump


def _wrap(owner, meth, label):
    orig = getattr(owner, meth, None)
    if orig is None:
        return

    def wrapped(*a, **k):
        _ev(label + ":start")
        try:
            return orig(*a, **k)
        finally:
            _ev(label + ":end")

    setattr(owner, meth, wrapped)


def _patch(modname, mod):
    if modname == "difflet.models.hunyuan_video.pipeline":
        cls = mod.HunyuanVideoOrchestrator
        orig = cls._scheduler_step

        def step(self, *a, **k):
            out = orig(self, *a, **k)
            _ev("scheduler_step")
            return out

        cls._scheduler_step = step
        _wrap(cls, "_denoise", "denoise")
        _wrap(cls, "_decode_latents", "pipe_decode")
        if _DUMP:  # phase 2, 2026-10-26 (HV AoT q1): initial/final latents, rank 0 / stage process
            orig_hden = cls._denoise

            def hden_dump(self, *a, **k):
                rank0 = os.environ.get("RANK") in ("0", None)
                bundle = k.get("bundle") if "bundle" in k else (a[0] if a else None)
                if rank0 and bundle is not None:
                    _save_latent("initial.pt", bundle.hidden_states)
                out = orig_hden(self, *a, **k)
                if rank0:
                    _save_latent("final.pt", out)
                return out

            cls._denoise = hden_dump
    elif modname == "difflet.models.hunyuan_video.neuron_application":
        cls = mod.TorchNeuronHunyuanVideoApplication
        _wrap(cls, "encode_prompt", "encode_prompt")
        orig_load = cls.load

        def load(self, *a, **k):
            _ev("app_load:start")
            try:
                return orig_load(self, *a, **k)
            finally:
                _ev("app_load:end")
                try:  # read-only: the application's own per-phase load seconds
                    _ev("app_phase_seconds",
                        phases={k2: round(float(v), 2) for k2, v in self.phase_seconds.items()})
                except Exception as exc:  # noqa: BLE001
                    _ev("app_phase_seconds_error", error=repr(exc))

        cls.load = load
    elif modname == "difflet.models.hunyuan_video.application":
        _wrap(mod.NeuronHunyuanVideoApplication, "load", "aot_app_load")
    elif modname == "difflet.backends.neuron.hunyuan_video.vae":
        cls = mod.NeuronHunyuanVideoVaeDecoderApplication
        _wrap(cls, "load", "vae_load")
        _wrap(cls, "decode", "vae_decode")
    elif modname == "difflet.cli.orchestrators.hunyuan_video":
        _wrap(mod, "_write_video", "write_video")
        _wrap(mod, "_save_video", "save_video")
        _wrap(mod, "_decode_latents_host", "host_decode")
        # phase 2, 2026-10-26 (HV AoT on main): one span per stage process
        for st in ("_stage_clip", "_stage_llama", "_stage_generate"):
            if hasattr(mod.HunyuanVideoOrchestrator, st):
                _wrap(mod.HunyuanVideoOrchestrator, st, "stage" + st[6:])
    # ---- HunyuanVideo AoT on origin/main (phase 2, 2026-10-26; additive) ----
    elif modname == "difflet.backends.trainium.hunyuan_video.vae":
        _wrap(mod.NeuronHunyuanVideoVAEDecoderApplication, "decode", "vae_decode")
    elif modname == "difflet.models.flux.clip.modeling_clip":
        _wrap(mod.NeuronClipApplication, "load", "clip_load")
    elif modname == "neuronx_distributed_inference.models.llama.modeling_llama":
        _wrap(mod.NeuronLlamaForCausalLM, "load", "llama_load")
    # ---- Qwen-Image (added for the 2026-10-25 Qwen-Image TP4 stage; additive) ----
    elif modname == "difflet.models.qwen_image.pipeline":
        cls = mod.QwenImageOrchestrator
        orig_q = cls._scheduler_step

        def qstep(self, *a, **k):
            out = orig_q(self, *a, **k)
            _ev("scheduler_step")
            return out

        cls._scheduler_step = qstep
        _wrap(cls, "_denoise", "denoise")
        _wrap(cls, "_decode_latents", "pipe_decode")
        if _DUMP:  # phase 2, 2026-10-26: native Qwen-Image initial/final latents (rank 0)
            orig_qden = cls._denoise

            def qden_dump(self, *a, **k):
                rank0 = os.environ.get("RANK") in ("0", None)
                bundle = k.get("bundle") if "bundle" in k else (a[0] if a else None)
                if rank0 and bundle is not None:
                    _save_latent("initial.pt", bundle.hidden_states)
                out = orig_qden(self, *a, **k)
                if rank0:
                    _save_latent("final.pt", out)
                return out

            cls._denoise = qden_dump
        _inject = os.environ.get("DV_INJECT_INITIAL")
        if _inject:  # 2026-11-01 PR #134: run the AoT DiT on another run's initial noise
            orig_prep = cls.prepare_latents

            def prep_inject(self, *a, **k):
                import torch

                drawn = orig_prep(self, *a, **k)  # draw as usual: same RNG use, then replace
                given = torch.load(_inject, map_location="cpu")
                if tuple(given.shape) != tuple(drawn.shape) or given.dtype != drawn.dtype:
                    raise RuntimeError(
                        f"DV_INJECT_INITIAL {_inject}: {tuple(given.shape)} {given.dtype} != "
                        f"drawn {tuple(drawn.shape)} {drawn.dtype}")
                _ev("initial_injected", path=_inject, shape=list(given.shape),
                    dtype=str(given.dtype), drawn_equal=bool(torch.equal(given, drawn.cpu())))
                return given.to(device=drawn.device)

            cls.prepare_latents = prep_inject
    elif modname == "difflet.models.qwen_image.neuron_application":
        cls = mod.TorchNeuronQwenImageApplication
        if hasattr(cls, "_initial_latents"):  # phase 2, 2026-10-26: record the noise's dtype
            orig_init = cls._initial_latents

            def qinit(self, *a, **k):
                out = orig_init(self, *a, **k)
                try:
                    _ev("initial_latents", dtype=str(out.dtype), shape=list(out.shape),
                        device=str(out.device))
                except Exception as exc:  # noqa: BLE001
                    _ev("initial_latents_error", error=repr(exc))
                return out

            cls._initial_latents = qinit
        _wrap(cls, "encode_prompt", "encode_prompt")
        orig_qload = cls.load

        def qload(self, *a, **k):
            _ev("app_load:start")
            try:
                return orig_qload(self, *a, **k)
            finally:
                _ev("app_load:end")
                try:  # read-only: the application's own per-phase load seconds
                    _ev("app_phase_seconds",
                        phases={k2: round(float(v), 2) for k2, v in self.phase_seconds.items()})
                except Exception as exc:  # noqa: BLE001
                    _ev("app_phase_seconds_error", error=repr(exc))

        cls.load = qload
        _wrap(mod._DeviceQwenVaeDecoder, "__call__", "vae_decode")
    elif modname == "difflet.models.qwen_image.application":
        _wrap(mod.NeuronQwenImageApplication, "load", "aot_app_load")
    elif modname == "difflet.backends.neuron.wan.vae":
        _wrap(mod.NeuronWanVaeDecoderApplication, "load", "vae_load")
    elif modname == "difflet.cli.orchestrators.qwen_image":
        _wrap(mod, "_write_image", "write_image")
        _wrap(mod, "_decode_latents_host", "host_decode")
        for st in ("_stage_text", "_stage_generate", "_stage_vae"):
            _wrap(mod.QwenImageOrchestrator, st, "stage" + st[6:])
    # ---- Wan2.1 (added for the 2026-10-25 Wan2.1 TP4 stage; additive) ----
    elif modname == "difflet.models.wan.pipeline":
        cls = mod.WanOrchestrator
        orig_w = cls._scheduler_step

        def wstep(self, *a, **k):
            out = orig_w(self, *a, **k)
            _ev("scheduler_step")
            return out

        cls._scheduler_step = wstep
        _wrap(cls, "_denoise", "denoise")
        _wrap(cls, "_decode_latents", "pipe_decode")
        _wrap(cls, "encode_prompt", "encode_prompt")
        if _DUMP:  # phase 2, 2026-10-26 (Wan2.1 AoT q1): initial/final latents, rank 0 / stage
            orig_wden = cls._denoise

            def wden_dump(self, *a, **k):
                rank0 = os.environ.get("RANK") in ("0", None)
                lat = k.get("latents")  # keyword-only in WanOrchestrator._denoise
                if rank0 and lat is not None:
                    _save_latent("initial.pt", lat)
                out = orig_wden(self, *a, **k)
                if rank0:
                    _save_latent("final.pt", out)
                return out

            cls._denoise = wden_dump
    elif modname == "difflet.models.wan.neuron_application":
        cls = mod.TorchNeuronWanApplication
        orig_wload = cls.load

        def wload(self, *a, **k):
            _ev("app_load:start")
            try:
                return orig_wload(self, *a, **k)
            finally:
                _ev("app_load:end")
                try:  # read-only: the application's own per-phase load seconds
                    _ev("app_phase_seconds",
                        phases={k2: round(float(v), 2) for k2, v in self.phase_seconds.items()})
                except Exception as exc:  # noqa: BLE001
                    _ev("app_phase_seconds_error", error=repr(exc))

        cls.load = wload
        _wrap(mod._DeviceVaeDecoder, "__call__", "vae_decode")
    elif modname == "difflet.models.wan.application":
        _wrap(mod.NeuronWanApplication, "load", "aot_app_load")
    elif modname == "difflet.cli.orchestrators.wan":
        _wrap(mod, "_write_frames", "write_video")
        _wrap(mod, "_decode_latents_host", "host_decode")
        _wrap(mod, "_save_video", "save_video")
        for st in ("_stage_transformer", "_stage_vae"):
            _wrap(mod.WanOrchestrator, st, "stage" + st[6:])
    # ---- four-model stage, 2026-10-26 (AoT on origin/main; additive) ----
    elif modname == "difflet.backends.trainium.wan.vae":
        _wrap(mod.NeuronWanVAEDecoderApplication, "load", "vae_load")
        _wrap(mod.NeuronWanVAEDecoderApplication, "__call__", "vae_decode")
    elif modname == "neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text":
        _wrap(mod.NeuronQwen2VLTextForCausalLM, "load", "text_enc_load")
    elif modname == "difflet.models.ltx_2.pipeline":
        cls = mod.LTX2Orchestrator
        orig_l = cls._scheduler_step
        orig_den = cls._denoise
        _calls = [0]

        def lstep(self, *a, **k):
            out = orig_l(self, *a, **k)
            _calls[0] += 1
            if _calls[0] % 2 == 0:  # video then audio: the second call ends the step
                _ev("scheduler_step")
            return out

        def lden(self, *a, **k):
            _calls[0] = 0
            _ev("denoise:start")
            try:
                return orig_den(self, *a, **k)
            finally:
                _ev("denoise:end")

        cls._scheduler_step = lstep
        cls._denoise = lden
        if _DUMP:  # phase 2, 2026-10-26 (LTX-2 native q run): rank 0's initial/final latents
            cls._denoise = ltx_dump_wrapper(lden)
        _wrap(cls, "_decode_latents", "pipe_decode")
        _wrap(cls, "prepare_conditioning", "encode_prompt")
    elif modname == "difflet.models.ltx_2.application":
        _wrap(mod.NeuronLTX2Application, "load", "aot_app_load")
        _wrap(mod, "_load_ltx_2_host_pipeline", "host_pipeline_load")
    elif modname == "difflet.cli.orchestrators.ltx_2":
        _wrap(mod, "_save_video", "save_video")
        if hasattr(mod, "_decode_latents_host"):  # native branch only (2026-10-26)
            _wrap(mod, "_decode_latents_host", "host_decode")
        _wrap(mod.LTX2Orchestrator, "_load_pipeline", "load_pipeline")
        _wrap(mod.LTX2Orchestrator, "generate", "cli_generate")
    elif modname == "difflet.cli.prewarm":
        _wrap(mod, "prewarm_neuron_runtime", "prewarm")
    # ---- LTX-2 native (phase 2, 2026-10-26; additive) ----
    elif modname == "difflet.models.ltx_2.neuron_application":
        cls = mod.TorchNeuronLtx2Application
        orig_lload = cls.load

        def lload(self, *a, **k):
            _ev("app_load:start")
            try:
                return orig_lload(self, *a, **k)
            finally:
                _ev("app_load:end")
                try:  # read-only: the application's own per-phase load seconds
                    _ev("app_phase_seconds",
                        phases={k2: round(float(v), 2) for k2, v in self.phase_seconds.items()})
                except Exception as exc:  # noqa: BLE001
                    _ev("app_phase_seconds_error", error=repr(exc))

        cls.load = lload
        _wrap(cls, "encode_prompt", "encode_prompt")
        orig_linit = cls._initial_latents

        def linit(self, *a, **k):
            out = orig_linit(self, *a, **k)
            try:
                _ev("initial_latents", video_dtype=str(out[0].dtype),
                    video_shape=list(out[0].shape), video_device=str(out[0].device),
                    audio_dtype=str(out[1].dtype), audio_shape=list(out[1].shape),
                    audio_device=str(out[1].device))
            except Exception as exc:  # noqa: BLE001
                _ev("initial_latents_error", error=repr(exc))
            return out

        cls._initial_latents = linit
        _wrap(mod._DeviceLtx2TextEncoder, "load", "text_enc_load")
        _wrap(mod._DeviceLtx2TextEncoder, "_load_connectors", "connectors_load")
        _wrap(mod._DeviceLtx2TextEncoder, "_connect", "connectors")
        _wrap(mod, "_host_ltx2_pipeline", "host_pipeline_load")
    elif modname == "difflet.backends.neuron.ltx_2.transformer":
        _wrap_load_phases(mod.NeuronLtx2TransformerApplication, "dit_load", "dit")
    elif modname == "difflet.backends.neuron.ltx_2.text_encoder":
        _wrap_load_phases(mod.NeuronLtx2TextEncoderApplication, "gemma_load", "gemma")
    elif modname == "difflet.cli.runner":
        _wrap(mod, "run_stage", "run_stage")


_TARGETS = {"difflet.models.hunyuan_video.pipeline",
            "difflet.models.hunyuan_video.neuron_application",
            "difflet.models.hunyuan_video.application",
            "difflet.backends.neuron.hunyuan_video.vae",
            "difflet.cli.orchestrators.hunyuan_video",
            # Qwen-Image stage, 2026-10-25
            "difflet.models.qwen_image.pipeline",
            "difflet.models.qwen_image.neuron_application",
            "difflet.models.qwen_image.application",
            "difflet.backends.neuron.wan.vae",
            "difflet.cli.orchestrators.qwen_image",
            # Wan2.1 stage, 2026-10-25
            "difflet.models.wan.pipeline",
            "difflet.models.wan.neuron_application",
            "difflet.models.wan.application",
            "difflet.cli.orchestrators.wan",
            # four-model stage, 2026-10-26
            "difflet.backends.trainium.wan.vae",
            "neuronx_distributed_inference.models.qwen2_vl.modeling_qwen2_vl_text",
            "difflet.models.ltx_2.pipeline",
            "difflet.models.ltx_2.application",
            "difflet.cli.orchestrators.ltx_2",
            "difflet.cli.prewarm",
            # HunyuanVideo AoT on main, phase 2 2026-10-26
            "difflet.backends.trainium.hunyuan_video.vae",
            "difflet.models.flux.clip.modeling_clip",
            "neuronx_distributed_inference.models.llama.modeling_llama",
            # LTX-2 native, phase 2 2026-10-26
            "difflet.models.ltx_2.neuron_application",
            "difflet.backends.neuron.ltx_2.transformer",
            "difflet.backends.neuron.ltx_2.text_encoder",
            "difflet.cli.runner"}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _patch(fullname, module)
            except Exception as exc:  # noqa: BLE001  (2026-10-26: record, never break the import)
                _ev("probe_patch_error", module=fullname, error=repr(exc))

        loader.exec_module = exec_module
        return spec


def _dump():
    rec = {"pid": os.getpid(), "argv": sys.argv, "t0": _T0, "t_exit": time.time(),
           "rank": os.environ.get("RANK"), "events": _EVENTS}
    _dl = sys.modules.get("difflet")  # 2026-10-26: which difflet tree this process ran
    rec["difflet_file"] = getattr(_dl, "__file__", None) if _dl is not None else None
    tn = sys.modules.get("torch_neuronx")
    if tn is not None and os.environ.get("RANK") is not None:  # the driver never starts NRT
        try:
            rec["peak_device_mem_gib"] = tn.max_memory_allocated() / 2**30
        except Exception as exc:  # noqa: BLE001
            rec["peak_device_mem_error"] = repr(exc)
    try:
        os.makedirs(_OUT, exist_ok=True)
        with open(os.path.join(_OUT, f"{os.getpid()}-rank{rec['rank']}.json"), "w") as f:
            json.dump(rec, f)
    except Exception:  # noqa: BLE001
        pass


if _OUT:
    sys.meta_path.insert(0, _Finder())
    atexit.register(_dump)
