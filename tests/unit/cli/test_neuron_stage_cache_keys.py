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
    # Where the VAE decodes is the vae stage's business, not the DiT stage's key.
    (WAN, [], ["--host-vae"], []),
    (QWEN, [], ["--device-vae"], []),
    (LTX, [], ["--device-vae"], []),
    (HV, [], ["--device-vae"], []),
    (HV, [], ["--host-vae"], []),
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
