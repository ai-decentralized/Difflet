"""Recipes: loading, ``extends``, mechanical translation, parser-derived routing, refusals."""
from __future__ import annotations

import textwrap

import pytest

from difflet.bench import recipe as R

REV = "75e0b4be04f60ec59a75f475837eced720f823b6"


def _write(tmp_path, text):
    path = tmp_path / "recipes.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def _base(**args):
    a = {"backend": "neuron", "model-id": "Qwen/Qwen-Image", "revision": REV,
         "prompt": "a cat", "tp-degree": 4}
    a.update(args)
    return R.Recipe(name="t", args=a)


def test_to_argv_switches_values_and_false():
    assert R.to_argv({"tp-degree": 4, "sp": True, "host-vae": False, "seed": None,
                      "guidance-scale": 7.5}) == ["--tp-degree", "4", "--sp",
                                                  "--guidance-scale", "7.5"]


def test_extends_merges_and_a_later_false_removes_a_switch(tmp_path):
    path = _write(tmp_path, f"""
        _base:
          args: {{backend: neuron, seed: 42, prompt: p, sp: true}}
        a:
          extends: _base
          threads: 8
          env: {{force: {{X: "1"}}}}
          args: {{model-id: Qwen/Qwen-Image, revision: "{REV}", tp-degree: 4}}
        b:
          extends: a
          args: {{sp: false, tp-degree: 2}}
          env: {{default: {{Y: "2"}}}}
    """)
    recipes = R.load_recipes(path)
    assert set(recipes) == {"a", "b"}  # _base is abstract
    b = recipes["b"]
    assert b.args["tp-degree"] == 2 and b.args["seed"] == 42
    assert "--sp" not in b.argv() and "--sp" in recipes["a"].argv()
    assert b.threads == 8 and b.env_force == {"X": "1"} and b.env_default == {"Y": "2"}


def test_extends_cycle_and_unknown_parent_are_refused(tmp_path):
    with pytest.raises(R.RecipeError, match="cycle"):
        R.load_recipes(_write(tmp_path, "a: {extends: b}\nb: {extends: a}\n"))
    with pytest.raises(R.RecipeError, match="unknown recipe 'nope'"):
        R.load_recipes(_write(tmp_path, "a: {extends: nope}\n"))


@pytest.mark.parametrize("text, match", [
    ("a: {argz: {seed: 1}}\n", "unknown keys"),
    ("a: {args: {tp_degree: 4}}\n", "without the leading dashes"),
    ("a: {args: {--seed: 1}}\n", "without the leading dashes"),
    ("a: {args: {seed: [1, 2]}}\n", "one value or none"),
    ("a: {env: {forced: {X: 1}}}\n", "env must be a mapping"),
    ("a: {threads: 0}\n", "positive integer"),
])
def test_structural_errors(tmp_path, text, match):
    with pytest.raises(R.RecipeError, match=match):
        R.load_recipes(_write(tmp_path, text))


def test_route_sends_every_flag_to_generate_and_compile_known_ones_to_compile():
    compile_argv, generate_argv = R.route(
        ["--model-id", "m", "--tp-degree", "4", "--sp", "--teacache-cadence", "2",
         "--steps", "28", "--guidance-scale", "3.5", "--seed", "-1"])
    assert generate_argv == ["--model-id", "m", "--tp-degree", "4", "--sp",
                             "--teacache-cadence", "2", "--steps", "28",
                             "--guidance-scale", "3.5", "--seed", "-1"]
    assert compile_argv == ["--model-id", "m", "--tp-degree", "4", "--sp",
                            "--teacache-cadence", "2"]


def test_route_accepts_equals_form():
    assert R.route(["--tp-degree=4"]) == (["--tp-degree", "4"], ["--tp-degree", "4"])


@pytest.mark.parametrize("tokens, match", [
    (["--teacache-cadance", "2"], "unknown flag '--teacache-cadance'"),
    (["--tea", "2"], "unknown flag '--tea'"),  # argparse would take the prefix; bench does not
    (["--steps"], "needs a value"),
    (["--sp=1"], "takes no value"),
    (["28"], "expected a --flag"),
])
def test_split_flags_refusals(tokens, match):
    with pytest.raises(R.RecipeError, match=match):
        R.split_flags(tokens)


@pytest.mark.parametrize("tokens, match", [
    (["--output", "x.png"], "managed by bench"),
    (["--cache-dir", "/tmp/c"], "managed by bench"),
    (["--requests", "r.jsonl"], "managed by bench"),
    (["--dp", "2"], "deferred throughput"),
    (["--mode", "latency"], "--mode"),
    (["--backend", "trainium"], "measures neuron only"),
])
def test_check_tokens_refusals(tokens, match):
    with pytest.raises(R.RecipeError, match=match):
        R.check_tokens(tokens, where="x")


def test_dp_one_is_allowed():
    R.check_tokens(["--dp", "1"], where="x")


@pytest.mark.parametrize("args, match", [
    ({"revision": None}, "missing required"),
    ({"backend": None}, "missing required"),
    ({"revision": 1234567890}, "quoted 40-character"),
    ({"revision": "main"}, "quoted 40-character"),
    ({"sp": "yes"}, "is a switch"),
    ({"steps": True}, "takes a value"),
    ({"no-such-flag": 1}, "unknown flag --no-such-flag"),
])
def test_check_recipe_refusals(args, match):
    with pytest.raises(R.RecipeError, match=match):
        R.check_recipe(_base(**args))


def test_every_shipped_recipe_passes_the_structural_checks():
    recipes = R.load_recipes()
    assert {"flux", "qwen_image", "ltx_2", "wan21", "wan22", "hunyuan_video",
            "hunyuan_video_15"} <= set(recipes)
    for recipe in recipes.values():
        R.check_recipe(recipe)
        assert recipe.args["backend"] == "neuron"


def test_shipped_negative_prompts_are_not_mangled():
    recipes = R.load_recipes()
    assert recipes["wan22"].args["negative-prompt"].startswith("色调艳丽，过曝")
    assert recipes["wan21"].args["negative-prompt"].endswith("walking backwards")
    assert recipes["ltx_2"].args["negative-prompt"].endswith("transition, static.")


def test_qwen_recipes_run_true_cfg():
    recipes = R.load_recipes()
    for name in ("qwen_image", "qwen_image_mxfp8"):
        argv = recipes[name].argv()
        assert argv[argv.index("--true-cfg-scale") + 1] == "4.0"  # the model card's
        assert argv[argv.index("--negative-prompt") + 1] == " "
        assert "--guidance-scale" not in argv  # ignored by Qwen-Image (guidance_embeds=false)


def test_parse_turns_argparse_errors_into_recipe_errors():
    with pytest.raises(R.RecipeError, match="invalid choice"):
        R.parse("generate", ["--model-id", "Qwen/Qwen-Image", "--prompt", "p",
                             "--cp-mode", "spiral"])


def test_merge_args_and_fingerprint_are_stable():
    merged = R.merge_args(_base(sp=True), {"sp": False, "quant": "mxfp8"}, name="t@s.mx")
    assert merged.name == "t@s.mx" and "--sp" not in merged.argv()
    assert "--quant" in merged.argv()
    assert R.fingerprint(["--steps", "8"]) == R.fingerprint(["--steps", "8"])
    assert R.fingerprint(["--steps", "8"]) != R.fingerprint(["--steps", "9"])
    assert _base().sha256() == _base().sha256() != _base(seed=1).sha256()
