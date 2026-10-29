"""The neuron DiT stage cache keys, one field at a time (review T1), for Wan, Qwen-Image, LTX-2
and HunyuanVideo.

Each case starts from one pinned command line (tp2, cp1 or cp2 gather_kv, exec mode compile,
every flag off), changes exactly one thing, and asserts that the stage cache inputs differ in
exactly the key that thing owns and that the stage dir changes, or, for a flag a model does not
key, that nothing changes. The old tests compared configurations that differed in several
fields at once (tp and the exec mode as well), so they would have passed with cp or cp_mode
dropped from the key.
"""

import importlib

import pytest

WAN = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
QWEN = "Qwen/Qwen-Image"
LTX = "Lightricks/LTX-2"
HV = "hunyuanvideo-community/HunyuanVideo"


def _key(monkeypatch, model, flags, exec_mode="compile"):
    """``(stage cache inputs, stage dir)`` of the model's DiT stage on the neuron backend."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", exec_mode)
    args = cli_main._build_parser().parse_args(
        ["generate", "--model-id", model, "--cache-dir", "/c", "--tp-degree", "2", *flags])
    orch = cli_main._get_orchestrator(args)
    if model == LTX:
        return orch._stage_cache_inputs(args), orch._stage_compiled_dir(args)
    stage = "transformer" if model == WAN else "generate"
    return orch._stage_cache_inputs(stage, args), orch._stage_compiled_dir(stage, args)


def _changed(before, after):
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


CP2 = ["--cp-degree", "2"]

#: (model, baseline flags, changed flags, the keys that must change, or [] for "not keyed").
CASES = [
    # cp degree: every model keys it.
    (WAN, [], CP2, ["cp"]),
    (QWEN, [], CP2, ["cp"]),
    (LTX, [], CP2, ["cp"]),
    (HV, [], CP2, ["cp"]),
    # cp mode at cp2: every model keys it (LTX-2 runs gather_kv and ulysses).
    (WAN, CP2, [*CP2, "--cp-mode", "ring"], ["cp_mode"]),
    (WAN, CP2, [*CP2, "--cp-mode", "ulysses"], ["cp_mode"]),
    (QWEN, CP2, [*CP2, "--cp-mode", "ring"], ["cp_mode"]),
    (QWEN, CP2, [*CP2, "--cp-mode", "ulysses"], ["cp_mode"]),
    (LTX, CP2, [*CP2, "--cp-mode", "ulysses"], ["cp_mode"]),
    (HV, CP2, [*CP2, "--cp-mode", "ring"], ["cp_mode"]),
    (HV, CP2, [*CP2, "--cp-mode", "ulysses"], ["cp_mode"]),
    # SP: Wan, Qwen-Image and HunyuanVideo key it (SP builds another graph); LTX-2 refuses SP in the driver,
    # so its key has no SP field.
    (WAN, [], ["--sp"], ["sp"]),
    (QWEN, [], ["--sp"], ["sp"]),
    (LTX, [], ["--sp"], []),
    (HV, [], ["--sp"], ["sp"]),
    # CFG parallelism: every model keys it (Qwen-Image additively: only when it is on).
    (WAN, [], ["--cfg-parallel"], ["cfg_parallel"]),
    (QWEN, [], ["--cfg-parallel"], ["cfg_parallel"]),
    (LTX, [], ["--cfg-parallel"], ["cfg_parallel"]),
    # HunyuanVideo is guidance-distilled: the driver refuses --cfg-parallel, so no key field.
    (HV, [], ["--cfg-parallel"], []),
    # Text encoder placement: LTX-2 keys a device encoder (additively); Wan's and Qwen-Image's
    # keys never named it (the neuron generate stage requires no artifact, and their NEFF
    # caches are keyed by the graphs themselves), so the flags leave their keys unchanged.
    (LTX, [], ["--device-text-encoder"], ["text_encoder_device"]),
    (WAN, [], ["--host-text-encoder"], []),
    (QWEN, [], ["--device-text-encoder"], []),
    # Where the VAE decodes is the vae stage's business, not the DiT stage's key...
    (WAN, [], ["--host-vae"], []),
    (QWEN, [], ["--device-vae"], []),
    (LTX, [], ["--device-vae"], []),
    # ...except HunyuanVideo's default, which decodes in the generate stage: its key names the
    # decoder ("vae"), which a placement outside the stage drops and a decoder option changes;
    # with the vae launch the decoder options are the vae stage's.
    (HV, [], ["--device-vae"], ["vae"]),
    (HV, [], ["--host-vae"], ["vae"]),
    (HV, [], ["--vae-dtype", "fp32"], ["vae"]),
    (HV, [], ["--vae-exec-mode", "eager"], ["vae"]),
    (HV, ["--device-vae"], ["--device-vae", "--vae-dtype", "fp32"], []),
]


@pytest.mark.parametrize(("model", "base", "changed", "keys"), CASES,
                         ids=[f"{m.split('/')[0]}-{' '.join(c[len(b):]) or c}"
                              for m, b, c, _ in CASES])
def test_one_field_at_a_time(monkeypatch, model, base, changed, keys):
    before, before_dir = _key(monkeypatch, model, base)
    after, after_dir = _key(monkeypatch, model, changed)
    assert _changed(before, after) == keys
    assert (after_dir != before_dir) is bool(keys)
    assert after_dir.parent == before_dir.parent  # the same component directory


@pytest.mark.parametrize("model", [WAN, QWEN, LTX, HV])
def test_exec_mode_alone_changes_the_key(monkeypatch, model):
    compiled, compiled_dir = _key(monkeypatch, model, [], exec_mode="compile")
    eager, eager_dir = _key(monkeypatch, model, [], exec_mode="eager")
    assert _changed(compiled, eager) == ["exec_mode"]
    assert (compiled["exec_mode"], eager["exec_mode"]) == ("compile", "eager")
    assert eager_dir != compiled_dir


# ------------------------------------------- Wan's DiT-step levers (F6a's proposals 1 and 2)

CCOP = "--tensorizer-options=--enable-ccop-compute-overlap"
LEVER_ENVS = ("DIFFLET_NEURON_WAN_CCOP_OVERLAP", "DIFFLET_NEURON_WAN_SCATTER_FREE_ROPE")


def _levers(monkeypatch, ccop=None, rope=None):
    """Set the two switches (None: unset, the default)."""
    for name, value in zip(LEVER_ENVS, (ccop, rope), strict=True):
        monkeypatch.setenv(name, "")
        if value is None:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, value)


def _wan_stage_key(monkeypatch, stage, backend="neuron"):
    cli_main = importlib.import_module("difflet.cli.main")
    if backend is None:
        monkeypatch.setenv("DIFFLET_BACKEND", "")
        monkeypatch.delenv("DIFFLET_BACKEND")
    else:
        monkeypatch.setenv("DIFFLET_BACKEND", backend)
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")
    args = cli_main._build_parser().parse_args(
        ["generate", "--model-id", WAN, "--cache-dir", "/c", "--tp-degree", "2"])
    orch = cli_main._get_orchestrator(args)
    return orch._stage_cache_inputs(stage, args), orch._stage_compiled_dir(stage, args)


def test_wan_levers_key_the_wan_neuron_stages(monkeypatch):
    """Both levers are on by default and each adds its field to the stage it changes: the
    compiler flag to both stages (every Wan neuron process compiles with it), the RoPE to the
    transformer stage. Both off is the key the stages had before the levers existed."""
    _levers(monkeypatch, "0", "0")
    off = {s: _wan_stage_key(monkeypatch, s) for s in ("transformer", "vae")}
    assert not {"neuron_cc_flags", "rope"} & (off["transformer"][0].keys() | off["vae"][0].keys())

    _levers(monkeypatch)
    on = {s: _wan_stage_key(monkeypatch, s) for s in ("transformer", "vae")}
    assert on["transformer"][0] == {**off["transformer"][0], "neuron_cc_flags": CCOP,
                                    "rope": "stack"}
    assert on["vae"][0] == {**off["vae"][0], "neuron_cc_flags": CCOP}
    for stage in ("transformer", "vae"):
        assert on[stage][1] != off[stage][1] and on[stage][1].parent == off[stage][1].parent

    _levers(monkeypatch, ccop="0")
    no_ccop = {s: _wan_stage_key(monkeypatch, s) for s in ("transformer", "vae")}
    assert _changed(on["transformer"][0], no_ccop["transformer"][0]) == ["neuron_cc_flags"]
    assert no_ccop["vae"] == off["vae"]
    _levers(monkeypatch, rope="0")
    no_rope = {s: _wan_stage_key(monkeypatch, s) for s in ("transformer", "vae")}
    assert _changed(on["transformer"][0], no_rope["transformer"][0]) == ["rope"]
    assert no_rope["vae"] == on["vae"]


@pytest.mark.parametrize("backend", [None, "trainium", "tpu"])
def test_wan_levers_leave_the_aot_wan_keys_alone(monkeypatch, backend):
    keys = []
    for value in (None, "0"):
        _levers(monkeypatch, value, value)
        keys.append({s: _wan_stage_key(monkeypatch, s, backend) for s in ("transformer", "vae")})
    assert keys[0] == keys[1]


@pytest.mark.parametrize("model", [QWEN, LTX, HV])
def test_wan_levers_leave_other_models_keys_alone(monkeypatch, model):
    _levers(monkeypatch)
    on = _key(monkeypatch, model, [])
    _levers(monkeypatch, "0", "0")
    assert _key(monkeypatch, model, []) == on
    assert not {"neuron_cc_flags", "rope"} & on[0].keys()


@pytest.mark.parametrize("user, keyed", [
    ("--tensorizer-options=--skip-pass=Foo", False),  # the user's option wins: no flag compiled
    ("--retry_failed_compilation --tensorizer-options=--foo", False),
    ("--retry_failed_compilation", True),
    (CCOP, True),  # the user exported the flag itself: compiled with it
])
def test_the_wan_key_names_the_flag_only_when_the_stages_compile_with_it(monkeypatch, user, keyed):
    """A user --tensorizer-options in NEURON_CC_FLAGS keeps the flag out of the Wan stage
    processes (append_cc_flags does not replace it), so the keys must not claim it: with one, the
    keys are the switch-off keys (the RoPE still named on the transformer stage), and a later
    default run is told to compile instead of accepting the manifest."""
    _levers(monkeypatch, ccop="0")
    monkeypatch.setenv("NEURON_CC_FLAGS", "")
    monkeypatch.delenv("NEURON_CC_FLAGS")
    off = {s: _wan_stage_key(monkeypatch, s) for s in ("transformer", "vae")}
    _levers(monkeypatch)
    monkeypatch.setenv("NEURON_CC_FLAGS", user)
    keys = {s: _wan_stage_key(monkeypatch, s) for s in ("transformer", "vae")}
    for stage in ("transformer", "vae"):
        expected = {**off[stage][0], **({"neuron_cc_flags": CCOP} if keyed else {})}
        if stage == "transformer":
            expected["rope"] = "stack"
        assert keys[stage][0] == expected
    assert (keys["vae"][1] == off["vae"][1]) is not keyed
