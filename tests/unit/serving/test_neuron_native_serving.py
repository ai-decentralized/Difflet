"""Native neuron serving for FLUX, Qwen-Image, Wan 2.1, HunyuanVideo and LTX-2: routing, the
per-model call contract, the runtime plan, and the rank-0-only output. No device."""

from __future__ import annotations

import dataclasses
import io
import subprocess
import sys
from types import SimpleNamespace

import pytest

from difflet.backends.gpu.policy import guidance_scale_2_refusal
from difflet.backends.neuron import serving as native
from difflet.models.spec import gpu_model_spec
from difflet.serving import model_registry
from difflet.serving.errors import DiffletServingError
from difflet.serving.model_registry import load_request_validator_factory, resolve_serving_model
from difflet.serving.options import ServeOptions
from difflet.serving.types import DiffletGenerateRequest, VideoGenerateOptions

FLUX = "black-forest-labs/FLUX.1-dev"
QWEN = "Qwen/Qwen-Image"
WAN21 = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
HV = "hunyuanvideo-community/HunyuanVideo"
LTX2 = "Lightricks/LTX-2"
NATIVE_IDS = (FLUX, QWEN, WAN21, HV, LTX2)
ADAPTER = "difflet.backends.neuron.serving:NeuronServingStageAdapter"


@pytest.fixture
def on_neuron(monkeypatch):
    monkeypatch.setattr(model_registry, "current_backend", lambda: "neuron")
    monkeypatch.setattr("difflet.backends.registry.current_backend", lambda: "neuron")


def _profile(model_id: str):
    return resolve_serving_model(ServeOptions(model_id=model_id, tp_degree=4)).profile


def _request(model_id: str, profile, **video_fields) -> DiffletGenerateRequest:
    video = None
    if profile.num_frames is not None:
        video = VideoGenerateOptions(num_frames=profile.num_frames, fps=profile.output_fps,
                                     **video_fields)
    return DiffletGenerateRequest(
        request_id="r1", model=model_id, prompt="a red fox", height=profile.height,
        width=profile.width, num_inference_steps=7, guidance_scale=3.0, seed=42,
        output_format="mp4" if video else "png", video=video)


# ------------------------------------------------------------------ routing


@pytest.mark.parametrize("model_id", NATIVE_IDS)
def test_native_models_route_to_the_native_adapter_on_neuron(on_neuron, model_id):
    resolved = resolve_serving_model(ServeOptions(model_id=model_id, tp_degree=4))

    assert resolved.metadata.orchestrator_factory == ADAPTER
    assert load_request_validator_factory(resolved.metadata).__name__ == (
        "NeuronServingRequestValidator")
    assert resolved.profile.parallel.tp_degree == 4


def test_minimax_h3_keeps_its_own_adapter_on_neuron(on_neuron):
    resolved = resolve_serving_model(ServeOptions(model_id="MiniMaxAI/MiniMax-H3", tp_degree=4))

    assert resolved.metadata.orchestrator_factory.endswith(":MiniMaxH3ServingStageAdapter")


def test_wan22_is_not_served_natively(on_neuron):
    with pytest.raises(ValueError, match="Wan2.1-T2V-14B"):
        resolve_serving_model(ServeOptions(model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers"))


def test_other_backends_keep_their_adapters(monkeypatch):
    monkeypatch.setattr(model_registry, "current_backend", lambda: "trainium")

    resolved = resolve_serving_model(ServeOptions(model_id=FLUX))

    assert resolved.metadata.orchestrator_factory != ADAPTER


def test_native_metadata_keeps_the_spec_contract():
    metadata = native.neuron_serving_metadata("ltx_2")

    assert metadata.output_modality == "video"
    assert metadata.checkpoint_ids == (LTX2,)
    assert metadata.output_has_audio is False  # the audio track is not served
    assert metadata.default_fps == 24


# ------------------------------------------------------------------ per-model contract


def test_application_kwargs_load_the_decoder_in_process():
    kwargs = {m: native.native_application_kwargs(m, native._spec(m))
              for m in native.NATIVE_SERVED_CHECKPOINTS}

    assert kwargs["flux"] == {"decode_vae": True}
    assert kwargs["qwen_image"]["enable_vae_decoder"] is True
    assert kwargs["qwen_image"]["text_seq_len"] == 1024
    assert kwargs["wan"]["enable_vae_decoder"] is True
    assert kwargs["wan"]["enable_transformer_2"] is False
    assert kwargs["wan"]["text_encoder_device"] == "neuron"
    assert kwargs["wan"]["text_seq_len"] == 512
    assert kwargs["hunyuan_video"]["text_encoder_device"] == "host"
    assert kwargs["hunyuan_video"]["enable_vae_decoder"] is True
    assert kwargs["ltx_2"]["enable_decode_components"] is True
    assert kwargs["ltx_2"]["text_encoder_device"] == "neuron"


def test_request_kwargs_per_model(on_neuron):
    flux = _profile(FLUX)
    assert native.native_request_kwargs("flux", _request(FLUX, flux), flux, primary=True) == {
        "prompt": "a red fox", "num_inference_steps": 7, "height": flux.height,
        "width": flux.width, "guidance_scale": 3.0, "output_type": "pil"}

    qwen = _profile(QWEN)
    assert native.native_request_kwargs("qwen_image", _request(QWEN, qwen), qwen,
                                        primary=True) == {
        "prompt": "a red fox", "num_inference_steps": 7, "output_type": "pt"}

    wan = _profile(WAN21)
    got = native.native_request_kwargs("wan", _request(WAN21, wan, negative_prompt="blur"), wan,
                                       primary=False)
    assert got["num_frames"] == wan.num_frames
    assert got["negative_prompt"] == "blur"
    assert got["output_type"] == "pt"

    hv = _profile(HV)
    got = native.native_request_kwargs("hunyuan_video", _request(HV, hv), hv, primary=True)
    assert set(got) == {"prompt", "num_inference_steps", "guidance_scale", "output_type"}


def test_ltx2_decodes_on_the_primary_replica_only(on_neuron):
    ltx = _profile(LTX2)
    request = _request(LTX2, ltx, negative_prompt="blur")

    assert native.native_request_kwargs("ltx_2", request, ltx, primary=True)["output_type"] == "pt"
    other = native.native_request_kwargs("ltx_2", request, ltx, primary=False)
    assert other["output_type"] == "latent"
    assert other["negative_prompt"] == "blur"


# ------------------------------------------------------------------ plan and profile


def test_runtime_plan_takes_one_core_per_rank(on_neuron, monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-11")
    profile = _profile(FLUX)

    plan = native.native_runtime_plan(profile, native._spec("flux"), "abc123")

    assert plan.environment.available_core_ids == (8, 9, 10, 11)
    assert plan.allocations[0].world_size == 4
    assert plan.stages[0].placement == "neuron_native"
    assert plan.stages[0].topology.tp_degree == 4
    assert plan.profile_identity


@pytest.mark.parametrize("field,value", [("teacache_speedup", 1.5), ("torch_compile", True),
                                         ("host_vae", True)])
def test_profile_options_native_serving_does_not_run(on_neuron, field, value):
    profile = dataclasses.replace(_profile(WAN21), **{field: value})

    with pytest.raises(DiffletServingError):
        native.refuse_unsupported_profile(profile, native._spec("wan"))


# ------------------------------------------------------------------ probe-free TeaCache


@pytest.mark.parametrize("field,value,expected", [
    ("teacache_cadence", 2, {"teacache_cadence": 2}),
    ("teacache_online_delta", 0.6, {"teacache_online_delta_alpha": 0.6}),
])
def test_probe_free_teacache_is_served_natively(on_neuron, field, value, expected):
    profile = dataclasses.replace(_profile(WAN21), **{field: value})

    native.refuse_unsupported_profile(profile, native._spec("wan"))  # not refused
    assert native.native_teacache_kwargs(profile) == expected


def test_teacache_off_adds_no_application_kwargs(on_neuron):
    assert native.native_teacache_kwargs(_profile(FLUX)) == {}


@pytest.mark.parametrize("model_id", NATIVE_IDS)
def test_serve_resolves_probe_free_teacache_on_neuron(on_neuron, model_id):
    options = ServeOptions(model_id=model_id, tp_degree=4, teacache_cadence=2)

    assert resolve_serving_model(options).profile.teacache_cadence == 2


def test_minimax_h3_refuses_probe_free_teacache(on_neuron):
    options = ServeOptions(model_id="MiniMaxAI/MiniMax-H3", tp_degree=4, teacache_cadence=2)

    with pytest.raises(DiffletServingError, match="teacache"):
        resolve_serving_model(options)


def test_the_loaded_application_gets_the_teacache_kwargs(on_neuron, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import asyncio

    from difflet.backends.neuron.runtime import PG_TIMEOUT_ENV
    from difflet.pipeline import difflet_pipeline

    seen = {}

    class _Pipe:
        def load(self, skip_warmup):
            seen["loaded"] = not skip_warmup

    def fake_from_pretrained(model_id, **kwargs):
        seen.update(kwargs)
        return _Pipe()

    monkeypatch.setattr(difflet_pipeline.DiffletPipeline, "from_pretrained",
                        staticmethod(fake_from_pretrained))
    # _export_rank_environment keeps operator values: preset both, so nothing leaks.
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    monkeypatch.setenv(PG_TIMEOUT_ENV, "60")
    profile = dataclasses.replace(_profile(WAN21), teacache_cadence=2)
    runtime = SimpleNamespace(profile=profile,
                              source=SimpleNamespace(resolved_source_id="abc123"))

    asyncio.run(native.NeuronServingStageAdapter(WAN21).create_loaded_runners(runtime))

    assert seen["loaded"]
    assert seen["application_kwargs"]["teacache_cadence"] == 2
    assert seen["application_kwargs"]["enable_vae_decoder"] is True  # the model's own kwargs


# ------------------------------------------------------------------ CFG parallel


def _cfg_profile(model_id: str):
    options = ServeOptions(model_id=model_id, tp_degree=2, cfg_parallel=True)
    return resolve_serving_model(options).profile


@pytest.mark.parametrize("model_id", [WAN21, LTX2])
def test_cfg_parallel_is_served_natively_for_true_cfg_models(on_neuron, model_id):
    profile = _cfg_profile(model_id)

    assert profile.parallel.cfg_parallel_enabled is True
    assert profile.world_size == 4  # TP2 x CFG2: one device
    native.refuse_unsupported_profile(profile, native._spec(profile.model_type))


@pytest.mark.parametrize("model_id,reason", [
    (FLUX, "guidance-distilled"), (HV, "guidance-distilled"),
    ("MiniMaxAI/MiniMax-H3", "guidance-distilled"), (QWEN, "planned"),
])
def test_cfg_parallel_is_refused_where_there_is_no_cfg_branch(on_neuron, model_id, reason):
    with pytest.raises(DiffletServingError, match=reason):
        _cfg_profile(model_id)


def test_cfg_parallel_stays_refused_off_the_native_backend(monkeypatch):
    monkeypatch.setattr(model_registry, "current_backend", lambda: "trainium")
    monkeypatch.setattr("difflet.backends.registry.current_backend", lambda: "trainium")

    with pytest.raises(DiffletServingError, match="true-CFG request path"):
        _cfg_profile(WAN21)


def test_cfg_parallel_takes_one_rank_per_core_and_enters_the_identity(on_neuron, monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "8-11")
    seen = []
    real = native.CompileArtifactIdentity.from_cache_inputs
    monkeypatch.setattr(native.CompileArtifactIdentity, "from_cache_inputs",
                        staticmethod(lambda inputs: seen.append(dict(inputs)) or real(inputs)))

    cfg_plan = native.native_runtime_plan(_cfg_profile(WAN21), native._spec("wan"), "abc")
    plain = dataclasses.replace(_cfg_profile(WAN21),
                                parallel=dataclasses.replace(_cfg_profile(WAN21).parallel,
                                                             cfg_parallel_enabled=False))
    plain_plan = native.native_runtime_plan(plain, native._spec("wan"), "abc")

    assert cfg_plan.allocations[0].world_size == 4
    assert cfg_plan.environment.available_core_ids == (8, 9, 10, 11)
    assert seen[0]["cfg_parallel"] is True
    assert "cfg_parallel" not in seen[1] and "sp" not in seen[1] and "cp_mode" not in seen[1]
    assert cfg_plan.profile_identity != plain_plan.profile_identity


def _validator(profile, monkeypatch):
    runtime = SimpleNamespace(profile=profile, source=SimpleNamespace(pinned_model_path="/none"))
    validator = native.NeuronServingRequestValidator(runtime)
    monkeypatch.setattr(validator, "_validate_prompts", lambda request, video: None)
    return validator


def test_ltx2_under_cfg_parallel_refuses_a_request_without_cfg(on_neuron, monkeypatch):
    profile = _cfg_profile(LTX2)
    validator = _validator(profile, monkeypatch)

    validator.validate(_request(LTX2, profile))  # guidance 3.0
    no_cfg = dataclasses.replace(_request(LTX2, profile), guidance_scale=1.0)
    with pytest.raises(DiffletServingError, match="guidance_scale must be > 1"):
        validator.validate(no_cfg)


def test_wan_under_cfg_parallel_keeps_guidance_one_valid(on_neuron, monkeypatch):
    profile = _cfg_profile(WAN21)

    _validator(profile, monkeypatch).validate(
        dataclasses.replace(_request(WAN21, profile), guidance_scale=1.0))


# ------------------------------------------------------------------ const-max attention

CONSTMAX_ENV = "DIFFLET_EXP_ATTN_CONSTMAX"


def _attention_profile(model_id: str, attention: str | None):
    return resolve_serving_model(
        ServeOptions(model_id=model_id, tp_degree=4, attention=attention)).profile


@pytest.mark.parametrize("model_id", NATIVE_IDS)
def test_const_max_is_served_for_every_native_model(on_neuron, model_id):
    assert _attention_profile(model_id, "const-max").attention == "const-max"
    assert _attention_profile(model_id, "cte").attention is None  # cte is the default


def test_const_max_is_refused_without_a_table(on_neuron):
    with pytest.raises(DiffletServingError, match="const-max table"):
        _attention_profile("MiniMaxAI/MiniMax-H3", "const-max")


def test_const_max_is_refused_off_the_native_backend(monkeypatch):
    monkeypatch.setattr(model_registry, "current_backend", lambda: "trainium")
    monkeypatch.setattr("difflet.backends.registry.current_backend", lambda: "trainium")

    with pytest.raises(DiffletServingError, match="native neuron backend only"):
        _attention_profile(WAN21, "const-max")


def test_const_max_enters_the_identity_and_cte_keeps_it(on_neuron, monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0-3")
    seen = []
    real = native.CompileArtifactIdentity.from_cache_inputs
    monkeypatch.setattr(native.CompileArtifactIdentity, "from_cache_inputs",
                        staticmethod(lambda inputs: seen.append(dict(inputs)) or real(inputs)))

    native.native_runtime_plan(_attention_profile(FLUX, "const-max"), native._spec("flux"), "a")
    native.native_runtime_plan(_attention_profile(FLUX, None), native._spec("flux"), "a")

    assert seen[0]["attention"] == "const-max"
    assert "attention" not in seen[1]


def test_the_parent_exports_const_max_before_the_ranks_spawn(on_neuron, monkeypatch):
    monkeypatch.setenv(CONSTMAX_ENV, "0")  # monkeypatch restores the variable afterwards

    native.export_attention_environment(_attention_profile(QWEN, "const-max"))

    import os
    assert os.environ[CONSTMAX_ENV] == "1"


def test_an_exported_const_max_without_the_flag_is_refused(on_neuron, monkeypatch):
    monkeypatch.setenv(CONSTMAX_ENV, "1")

    with pytest.raises(DiffletServingError, match="--attention const-max"):
        native.export_attention_environment(_attention_profile(QWEN, None))


@pytest.mark.parametrize("module_constmax,attention,ok", [
    (True, "const-max", True), (False, None, True),
    (False, "const-max", False), (True, None, False),
])
def test_a_rank_refuses_an_attention_kernel_the_profile_does_not_name(
        on_neuron, monkeypatch, module_constmax, attention, ok):
    pytest.importorskip("torch")
    from difflet.backends.neuron.ops_impl import attention as attention_module

    monkeypatch.setattr(attention_module, "_CONSTMAX", module_constmax)
    profile = dataclasses.replace(_profile(WAN21), attention=attention)
    if ok:
        native.check_attention_kernel(profile)
    else:
        with pytest.raises(RuntimeError, match="must be set before the rank imports it"):
            native.check_attention_kernel(profile)


@pytest.mark.parametrize("model_id", NATIVE_IDS)
def test_default_profiles_pass_the_refusals(on_neuron, model_id):
    profile = _profile(model_id)  # HunyuanVideo's defaults CLIP to "host"

    native.refuse_unsupported_profile(profile, native._spec(profile.model_type))


def test_device_clip_placement_is_refused(on_neuron):
    profile = dataclasses.replace(_profile(HV), clip_placement="neuron")

    with pytest.raises(DiffletServingError, match="clip-placement"):
        native.refuse_unsupported_profile(profile, native._spec("hunyuan_video"))


def test_validator_refuses_guidance_scale_2_for_wan21(on_neuron, monkeypatch):
    profile = _profile(WAN21)
    runtime = SimpleNamespace(profile=profile, source=SimpleNamespace(pinned_model_path="/none"))
    validator = native.NeuronServingRequestValidator(runtime)
    monkeypatch.setattr(validator, "_validate_prompts", lambda request, video: None)

    validator.validate(_request(WAN21, profile))
    with pytest.raises(DiffletServingError) as exc:
        validator.validate(_request(WAN21, profile, guidance_scale_2=3.0))
    # The GPU flow's refusal, inherited: one wording for every backend that serves the spec.
    assert exc.value.code == "feature_not_supported"
    assert exc.value.message.endswith(guidance_scale_2_refusal(gpu_model_spec("wan"), WAN21))


# ------------------------------------------------------------------ adapter


def test_every_rank_runs_the_pipeline_and_only_the_primary_writes(on_neuron, monkeypatch):
    profile = _profile(FLUX)
    adapter = native.NeuronServingStageAdapter(FLUX)
    calls = []
    adapter.pipe = lambda **kwargs: calls.append(kwargs) or SimpleNamespace(images=["latent"])
    adapter.active_profile = profile
    monkeypatch.setenv("DIFFLET_REPLICA_RANK", "1")

    out = adapter.generate(_request(FLUX, profile))

    assert out.data == b""
    assert calls[0]["generator"].initial_seed() == 42


def test_qwen_tensor_becomes_the_cli_png(on_neuron):
    torch = pytest.importorskip("torch")
    from PIL import Image

    profile = _profile(QWEN)
    out = native.encode_output("qwen_image", SimpleNamespace(images=torch.zeros(1, 3, 8, 8)),
                               _request(QWEN, profile))

    image = Image.open(io.BytesIO(out.data))
    assert image.size == (8, 8)
    assert image.getpixel((0, 0)) == (128, 128, 128)  # (0 * 0.5 + 0.5) * 255 + 0.5


def test_module_does_not_import_torch():
    code = ("import sys, difflet.backends.neuron.serving, difflet.serving.model_registry; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "DIFFLET_BACKEND": "cpu",
                        "PATH": "/usr/bin:/bin", "PYTHONPATH": ":".join(sys.path)})
