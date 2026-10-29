"""Wan's two DiT-step levers on the neuron backend (F6a's proposals 1 and 2), CPU only:

* AoT's neuronx-cc flag ``--tensorizer-options=--enable-ccop-compute-overlap``, appended once to
  ``NEURON_CC_FLAGS`` by every Wan application built for the neuron device, and by no other
  model's (``DIFFLET_NEURON_WAN_CCOP_OVERLAP=0`` turns it off);
* the scatter-free RoPE: the interleaved q/k write built with ``stack`` + ``flatten`` instead of
  the shared helper's strided writes, which torch-neuronx lowers to scatters. Bit-identical to the
  shared ``apply_rotary_emb``; set on the Wan DiT's attentions by the neuron application only, so
  the shared helper, Flux and HunyuanVideo keep their graphs
  (``DIFFLET_NEURON_WAN_SCATTER_FREE_ROPE=0`` turns it off).

The tiny model is tests/unit/backends/_neuron_wan_toy.py's. Constructing an application for the
``"neuron"`` device touches no device (test_neuron_wan_w1.py checks that).
"""

import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace  # noqa: E402

from difflet.backends.cpu.ops_impl import embeddings as cpu_embeddings  # noqa: E402
from difflet.backends.neuron.ops_impl import embeddings as neuron_embeddings  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.backends.neuron.wan import compile_options  # noqa: E402
from tests.unit.backends._neuron_wan_toy import (  # noqa: E402
    TINY_LATENT_SHAPE,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    tiny_wan_config,
    tiny_wan_inputs,
    write_tiny_wan_model,
)

CCOP = "--tensorizer-options=--enable-ccop-compute-overlap"
CCOP_ENV = "DIFFLET_NEURON_WAN_CCOP_OVERLAP"
ROPE_ENV = "DIFFLET_NEURON_WAN_SCATTER_FREE_ROPE"


def _unset(monkeypatch, *names):
    """Unset ``names`` now and restore them after the test, also when the code under test
    assigns os.environ itself (delenv of an absent name records nothing to undo)."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    _unset(monkeypatch, "NEURON_CC_FLAGS", CCOP_ENV, ROPE_ENV, "DIFFLET_EXEC_MODE")
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("safetensors.torch")
    pytest.importorskip("accelerate")
    pytest.importorskip("transformers")
    root = tmp_path_factory.mktemp("tiny_wan_levers")
    write_tiny_wan_model(root)
    return SimpleNamespace(dir=str(root), transformer=root / "transformer")


# ------------------------------------------------------------------ (a) the RoPE, bit for bit


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("tables", ["same", "fp32"])
@pytest.mark.parametrize("shape", [(1, 7, 3, 12), (2, 12, 4, 128), (1, 5, 1, 2), (3, 1, 5, 6)],
                         ids=["odd-seq", "even", "head-dim-2", "one-token"])
def test_stacked_rope_equals_the_shared_helper_bit_for_bit(dtype, tables, shape):
    gen = torch.Generator().manual_seed(sum(shape))
    hidden = (torch.randn(shape, generator=gen) * 3).to(dtype)
    angles = torch.randn(1, shape[1], 1, shape[-1], generator=gen) * 10
    table_dtype = dtype if tables == "same" else torch.float32  # Wan's tables are fp32
    cos, sin = angles.cos().to(table_dtype), angles.sin().to(table_dtype)

    expected = cpu_embeddings.apply_rotary_emb(hidden, cos, sin)
    got = neuron_embeddings.apply_rotary_emb_stacked(hidden, cos, sin)

    assert got.dtype == expected.dtype == dtype and got.shape == expected.shape
    assert got.is_contiguous()
    assert torch.equal(_bits(got), _bits(expected))


def test_the_shared_helper_is_unchanged():
    # Flux, HunyuanVideo 1.0/1.5, the CPU backend and the neuron backend's difflet.ops all keep
    # the one shared function, strided writes included.
    assert neuron_embeddings.apply_rotary_emb is cpu_embeddings.apply_rotary_emb
    assert neuron_embeddings.apply_rotary_emb_stacked is not cpu_embeddings.apply_rotary_emb
    assert "slice_scatter" in " ".join(_aten_targets(cpu_embeddings.apply_rotary_emb))
    assert "scatter" not in " ".join(_aten_targets(neuron_embeddings.apply_rotary_emb_stacked))


def test_flux_and_hunyuan_video_call_the_shared_helper():
    from difflet.models.flux import modeling_flux
    from difflet.models.hunyuan_video import modeling_hunyuan_video

    for module in (modeling_flux, modeling_hunyuan_video):
        fn = module.apply_rotary_emb
        assert fn is not neuron_embeddings.apply_rotary_emb_stacked
        assert (fn.__name__, fn.__module__.rsplit(".", 1)[-1]) == ("apply_rotary_emb",
                                                                   "embeddings")


# ------------------------------------------------- (b) the Wan DiT's traced graph has no scatter


def _aten_targets(fn, *args):
    """The call targets of ``fn``'s functionalized aten graph (what torch-neuronx lowers)."""
    from torch._functorch.aot_autograd import aot_function
    from functorch.compile import make_boxed_func

    graphs = []

    def compiler(gm, example_inputs):
        del example_inputs
        graphs.append([str(n.target) for n in gm.graph.nodes if n.op == "call_function"])
        return make_boxed_func(gm.forward)

    if not args:
        args = (torch.randn(1, 3, 2, 8), torch.randn(1, 3, 1, 8), torch.randn(1, 3, 1, 8))
    with torch.no_grad():
        aot_function(fn, fw_compiler=compiler)(*args)
    [targets] = graphs
    return targets


def _aten_recording_backend(graphs: list):
    """A Dynamo backend recording each compiled region's functionalized aten graph."""
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    def compiler(gm, example_inputs):
        del example_inputs
        graphs.append([str(n.target) for n in gm.graph.nodes if n.op == "call_function"])
        return make_boxed_func(gm.forward)

    return aot_autograd(fw_compiler=compiler)


def _compiled_wan_dit(tiny):
    """The tiny Wan DiT application at TP1, compile mode, its blocks' aten graphs recorded."""
    from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication

    torch._dynamo.reset()
    graphs: list = []
    app = NeuronWanTransformerApplication(model_path=tiny.transformer, config=tiny_wan_config(),
                                          dtype=torch.float32, exec_mode="compile", device="cpu")
    app.compile_backend = _aten_recording_backend(graphs)
    app.load()
    out = app(*tiny_wan_inputs())
    torch._dynamo.reset()
    return app, graphs, out


def test_the_wan_dit_block_graph_has_no_scatter(tiny, monkeypatch):
    app, graphs, out = _compiled_wan_dit(tiny)
    [block] = graphs  # one graph for both blocks
    assert not [t for t in block if "scatter" in t]
    for wan_block in app.module.blocks:
        assert wan_block.attn1.rotary_fn is neuron_embeddings.apply_rotary_emb_stacked

    # The opt-out keeps the shared helper and its strided writes; the output is the same bits.
    monkeypatch.setenv(ROPE_ENV, "0")
    parallel_mesh.destroy_parallel_mesh()
    shared_app, shared_graphs, shared_out = _compiled_wan_dit(tiny)
    [shared_block] = shared_graphs
    assert [t for t in shared_block if "slice_scatter" in t]
    for wan_block in shared_app.module.blocks:
        assert wan_block.attn1.rotary_fn is cpu_embeddings.apply_rotary_emb
    assert torch.equal(_bits(out), _bits(shared_out))


def test_wan_modeling_keeps_the_shared_helper_outside_the_neuron_application():
    # The modeling file is the AoT (Trainium) and TPU paths' too: built directly, every
    # attention applies difflet.ops.apply_rotary_emb as before.
    from difflet.models.wan import modeling_wan
    from tests.unit.backends._neuron_toy import tp1_mesh
    from tests.unit.backends._neuron_wan_toy import tiny_wan_tp1_model

    assert modeling_wan.WanAttention.rotary_fn is modeling_wan.apply_rotary_emb
    with tp1_mesh():
        model = tiny_wan_tp1_model()
    for block in model.blocks:
        assert block.attn1.rotary_fn is modeling_wan.apply_rotary_emb


# ------------------------------------------------------------- (c) the compiler flag, Wan only


@pytest.mark.parametrize("user, expected", [
    (None, CCOP),
    ("", CCOP),
    ("  ", CCOP),
    ("--retry_failed_compilation", f"--retry_failed_compilation {CCOP}"),
    (f"{CCOP}", CCOP),
    (f"--retry_failed_compilation {CCOP} -O2", f"--retry_failed_compilation {CCOP} -O2"),
])
def test_append_cc_flags_appends_each_option_once(user, expected):
    from difflet.backends.neuron.compile import append_cc_flags

    env = {} if user is None else {"NEURON_CC_FLAGS": user}
    assert append_cc_flags(CCOP, env=env) == expected
    assert append_cc_flags(CCOP, env=env) == expected  # idempotent
    assert env == {"NEURON_CC_FLAGS": expected}


def test_a_user_tensorizer_option_is_left_alone(caplog):
    from difflet.backends.neuron.compile import append_cc_flags

    # neuronx-cc takes one --tensorizer-options; a second would replace the user's.
    env = {"NEURON_CC_FLAGS": "--tensorizer-options=--skip-pass=Foo"}
    with caplog.at_level("WARNING"):
        assert append_cc_flags(CCOP, env=env) == "--tensorizer-options=--skip-pass=Foo"
    assert env == {"NEURON_CC_FLAGS": "--tensorizer-options=--skip-pass=Foo"}
    assert "--tensorizer-options" in caplog.text


def _wan_dit(tiny, device):
    from difflet.backends.neuron.wan.transformer import NeuronWanTransformerApplication

    dtype = torch.bfloat16 if device == "neuron" else torch.float32
    return NeuronWanTransformerApplication(model_path=tiny.transformer, config=tiny_wan_config(),
                                           dtype=dtype, device=device)


def _wan_umt5(tiny, device):
    from difflet.backends.neuron.wan.text_encoder import NeuronWanTextEncoderApplication

    return NeuronWanTextEncoderApplication(model_path=f"{tiny.dir}/text_encoder",
                                           text_seq_len=TINY_TEXT_SEQ_LEN, dtype=torch.bfloat16,
                                           device=device)


def _wan_vae(tiny, device):
    # As Wan's callers build it (the CLI vae stage, TorchNeuronWanApplication): opted in.
    from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication

    return NeuronWanVaeDecoderApplication(model_path=f"{tiny.dir}/vae",
                                          latent_shape=TINY_LATENT_SHAPE, dtype=torch.bfloat16,
                                          device=device, ccop_overlap=True)


def _wan_composed(tiny, device):
    from difflet.models.wan.neuron_application import TorchNeuronWanApplication
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    return TorchNeuronWanApplication(model_path=tiny.dir, parallel=DiffletParallelConfig(),
                                     dtype=torch.bfloat16, shape=dict(TINY_SHAPE),
                                     text_seq_len=TINY_TEXT_SEQ_LEN, device=device)


WAN_APPS = {"dit": _wan_dit, "umt5": _wan_umt5, "vae": _wan_vae, "composed": _wan_composed}


@pytest.mark.parametrize("build", WAN_APPS.values(), ids=WAN_APPS.keys())
def test_every_wan_application_on_the_device_appends_the_flag_once(tiny, monkeypatch, build):
    import os

    apps = [build(tiny, "neuron")]  # held: the flag lives as long as a Wan application does
    assert os.environ["NEURON_CC_FLAGS"] == CCOP
    apps.append(build(tiny, "neuron"))  # a second application in the process: still once
    assert os.environ["NEURON_CC_FLAGS"] == CCOP

    monkeypatch.setenv("NEURON_CC_FLAGS", "--retry_failed_compilation")
    apps.append(build(tiny, "neuron"))
    assert os.environ["NEURON_CC_FLAGS"] == f"--retry_failed_compilation {CCOP}"


@pytest.mark.parametrize("build", WAN_APPS.values(), ids=WAN_APPS.keys())
@pytest.mark.parametrize("off", ["0", "off", "false", "no"])
def test_the_opt_out_leaves_neuron_cc_flags_alone(tiny, monkeypatch, build, off):
    import os

    monkeypatch.setenv(CCOP_ENV, off)
    apps = [build(tiny, "neuron")]
    assert "NEURON_CC_FLAGS" not in os.environ
    monkeypatch.setenv("NEURON_CC_FLAGS", "--retry_failed_compilation")
    apps.append(build(tiny, "neuron"))
    assert os.environ["NEURON_CC_FLAGS"] == "--retry_failed_compilation"


@pytest.mark.parametrize("build", [_wan_dit, _wan_vae], ids=["dit", "vae"])
def test_a_cpu_wan_application_leaves_neuron_cc_flags_alone(tiny, build):
    import os

    app = build(tiny, "cpu")
    assert app is not None and "NEURON_CC_FLAGS" not in os.environ


def test_other_neuron_applications_leave_neuron_cc_flags_alone():
    # The base every neuron model's application derives from (here its toy) adds nothing.
    import os

    from tests.unit.backends._neuron_toy import ToyApplication

    app = ToyApplication(device="neuron")
    assert app is not None and "NEURON_CC_FLAGS" not in os.environ


# --------------------------------- (c2) Qwen-Image's device VAE is Wan's class: no flag there


def test_the_wan_vae_class_adds_the_flag_only_when_asked(tiny):
    """Qwen-Image builds NeuronWanVaeDecoderApplication too; only Wan's callers opt in."""
    import os

    from difflet.backends.neuron.wan.vae import NeuronWanVaeDecoderApplication

    app = NeuronWanVaeDecoderApplication(model_path=f"{tiny.dir}/vae",
                                         latent_shape=TINY_LATENT_SHAPE, dtype=torch.bfloat16,
                                         device="neuron")
    assert app.device.type == "neuron" and "NEURON_CC_FLAGS" not in os.environ


def _vae_stage_flags(monkeypatch, orchestrator_module, orchestrator, args):
    """Run ``orchestrator``'s neuron vae stage up to its decoder: ``run_vae_stage`` replaced by
    one that builds the decoder exactly as the stage does (device "neuron"; constructing touches
    no device) and returns; the NEURON_CC_FLAGS the build left, with the decoder still alive."""
    import os

    from difflet.pipeline.parallel_config import DiffletParallelConfig

    seen = {}

    def fake_run_vae_stage(stage_args, *, build, finish_compile, prepare_decode):
        seen["app"] = build(DiffletParallelConfig(tp_degree=1))
        seen["flags"] = os.environ.get("NEURON_CC_FLAGS")

    monkeypatch.setattr(orchestrator_module, "run_vae_stage", fake_run_vae_stage)
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    orchestrator(args)._stage_vae_neuron(args)
    assert seen["app"].device.type == "neuron"
    return seen["flags"]


def test_qwen_image_vae_stage_leaves_neuron_cc_flags_alone(monkeypatch, tmp_path):
    pytest.importorskip("diffusers")
    from difflet.cli.orchestrators import qwen_image as qwen_mod
    from tests.unit.cli.test_orchestrator_qwen_neuron import _qwen_args, _tiny_vae_snapshot

    model = tmp_path / "model"
    _tiny_vae_snapshot(model)
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: str(model))
    args = _qwen_args(height=32, width=48, work_dir=str(tmp_path / "w"))
    assert _vae_stage_flags(monkeypatch, qwen_mod, qwen_mod.QwenImageOrchestrator, args) is None


def test_wan_vae_stage_sets_the_flag(monkeypatch, tiny):
    import argparse

    from difflet.cli.orchestrators import wan as wan_mod

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path",
                        lambda *a, **k: tiny.dir)
    args = argparse.Namespace(model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", revision=None,
                              height=TINY_SHAPE["height"], width=TINY_SHAPE["width"],
                              num_frames=TINY_SHAPE["num_frames"], vae_dtype="bf16",
                              vae_exec_mode=None)
    assert _vae_stage_flags(monkeypatch, wan_mod, wan_mod.WanOrchestrator, args) == CCOP


@pytest.mark.parametrize("enable_transformer", [False, True])
def test_qwen_image_composed_app_with_the_device_vae_leaves_neuron_cc_flags_alone(
        tmp_path_factory, enable_transformer):
    """The composed Qwen-Image application builds Wan's decoder class for its device VAE after
    its DiT; neither may bring Wan's flag into the process."""
    import os

    pytest.importorskip("diffusers")
    from difflet.models.qwen_image import neuron_application as qna
    from difflet.pipeline.parallel_config import DiffletParallelConfig
    from tests.unit.backends._neuron_qwen_app_toy import write_tiny_qwen_snapshot
    from tests.unit.backends._neuron_qwen_toy import TINY_SHAPE as QWEN_SHAPE
    from tests.unit.backends._neuron_qwen_toy import TINY_TEXT_SEQ_LEN as QWEN_TEXT_SEQ_LEN

    root = tmp_path_factory.mktemp("tiny_qwen_levers")
    write_tiny_qwen_snapshot(root, text_encoder=False)
    app = qna.TorchNeuronQwenImageApplication(
        model_path=str(root), parallel=DiffletParallelConfig(), dtype=torch.bfloat16,
        shape=dict(QWEN_SHAPE), text_seq_len=QWEN_TEXT_SEQ_LEN, device="neuron",
        enable_text_encoder=False, enable_transformer=enable_transformer,
        enable_vae_decoder=True)
    assert app.vae_decoder is not None and "NEURON_CC_FLAGS" not in os.environ


# ----------------------------------- (c3) the flag leaves with the last Wan application


def test_the_flag_leaves_with_the_last_wan_application(tiny):
    """An API caller who drops its Wan applications and builds another model in the same
    process compiles that model without Wan's flag."""
    import gc
    import os

    from tests.unit.backends._neuron_toy import ToyApplication

    first, second = _wan_dit(tiny, "neuron"), _wan_umt5(tiny, "neuron")
    assert os.environ["NEURON_CC_FLAGS"] == CCOP
    del first
    gc.collect()
    assert os.environ["NEURON_CC_FLAGS"] == CCOP  # the text encoder still holds it
    del second
    gc.collect()
    assert "NEURON_CC_FLAGS" not in os.environ
    other = ToyApplication(device="neuron")
    assert other is not None and "NEURON_CC_FLAGS" not in os.environ


def test_dropping_the_composed_wan_application_restores_neuron_cc_flags(tiny):
    import gc
    import os

    app = _wan_composed(tiny, "neuron")
    assert os.environ["NEURON_CC_FLAGS"] == CCOP
    del app
    gc.collect()
    assert "NEURON_CC_FLAGS" not in os.environ


@pytest.mark.parametrize("user", ["--retry_failed_compilation", "", CCOP,
                                  f"--retry_failed_compilation {CCOP}"])
def test_release_restores_the_users_flags(tiny, monkeypatch, user):
    """``release_ccop_overlap`` (or the collection of the application) gives back the user's
    value as it was; a flag the user exported stays."""
    import os

    monkeypatch.setenv("NEURON_CC_FLAGS", user)
    app = _wan_dit(tiny, "neuron")
    assert CCOP in os.environ["NEURON_CC_FLAGS"].split()
    compile_options.release_ccop_overlap(app)
    assert os.environ["NEURON_CC_FLAGS"] == user
    compile_options.release_ccop_overlap(app)  # twice: nothing more
    assert os.environ["NEURON_CC_FLAGS"] == user


def test_release_keeps_flags_set_since(tiny, monkeypatch):
    # Someone changed NEURON_CC_FLAGS after the flag went in: only the flag leaves.
    import os

    app = _wan_dit(tiny, "neuron")
    monkeypatch.setenv("NEURON_CC_FLAGS", f"{CCOP} -O1")
    compile_options.release_ccop_overlap(app)
    assert os.environ["NEURON_CC_FLAGS"] == "-O1"


# ------------------------------ (c4) the stage key and the applied flag decide the same way


@pytest.mark.parametrize("user", [None, "", "--retry_failed_compilation", CCOP,
                                  "--tensorizer-options=--skip-pass=Foo",
                                  "--retry_failed_compilation --tensorizer-options=--foo",
                                  f"--tensorizer-options=--foo {CCOP}"])
@pytest.mark.parametrize("switch", [None, "0"])
def test_the_stage_key_names_the_flag_iff_a_wan_application_compiles_with_it(
        tiny, monkeypatch, user, switch):
    import os

    if user is not None:
        monkeypatch.setenv("NEURON_CC_FLAGS", user)
    if switch is not None:
        monkeypatch.setenv(CCOP_ENV, switch)
    keyed = "neuron_cc_flags" in compile_options.stage_key_fields("transformer")
    assert keyed == ("neuron_cc_flags" in compile_options.stage_key_fields("vae"))
    assert keyed == compile_options.ccop_overlap_applies()
    app = _wan_dit(tiny, "neuron")
    compiled_with = CCOP in (os.environ.get("NEURON_CC_FLAGS") or "").split()
    assert app is not None
    if switch is None:
        assert keyed == compiled_with
    else:
        assert not keyed  # off: never keyed (a flag the user exported is the user's business)
    if switch is None and user in ("--tensorizer-options=--skip-pass=Foo",
                                   "--retry_failed_compilation --tensorizer-options=--foo"):
        assert not keyed and not compiled_with  # the user's tensorizer option wins


@pytest.mark.parametrize("value, ccop, rope", [
    (None, True, True), ("", True, True), ("1", True, True), ("yes", True, True),
    ("0", False, False), ("off", False, False), ("False", False, False), ("no", False, False),
])
def test_switches_default_on(monkeypatch, value, ccop, rope):
    from difflet import envs

    for name in (CCOP_ENV, ROPE_ENV):
        if value is not None:
            monkeypatch.setenv(name, value)
    assert envs.DIFFLET_NEURON_WAN_CCOP_OVERLAP is ccop
    assert envs.DIFFLET_NEURON_WAN_SCATTER_FREE_ROPE is rope
    assert compile_options.ccop_overlap_enabled() is ccop
    assert compile_options.scatter_free_rope_enabled() is rope


def test_compile_options_is_torch_free():
    # The CLI driver keys the Wan stages with it and stays torch-free.
    import subprocess
    import sys

    code = ("import sys; import difflet.backends.neuron.wan.compile_options; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
