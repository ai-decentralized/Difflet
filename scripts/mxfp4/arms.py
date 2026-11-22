#!/usr/bin/env python3
"""Phase-0 arm wiring: put one arm on a diffusers DiT in place, and the original linears back.

Arms: ``fp32`` (nothing), ``mxfp8`` (every target is difflet's MXFP8 CPU reference layer,
``FakeQuantMXLinear``), ``fp4-ocp`` / ``fp4-mse`` (the same layers with the same MXFP8 activation
law, and on the layers ``fp4_targets`` names the dequantized weight replaced by the MXFP4 law's,
``fp4.py``, in the layer's own K shards). So an FP4 arm differs from the mxfp8 arm in the weight
format of those layers and in nothing else.

    python scripts/mxfp4/arms.py    # self-test on small random FLUX and Qwen-Image models
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.quant import mx
from difflet.quant.fake_linear import (FakeQuantMXLinear, fake_quantize_model_,
                                       split_linear_columns)
from difflet.quant.spec import QuantSpec
from difflet.quant.targets import default_spec
from scripts.mxfp4.fp4 import fake_quantize_fp4_sharded

ARMS = ("fp32", "mxfp8", "fp4-ocp", "fp4-mse")


def _originals(model: nn.Module, spec: QuantSpec) -> dict[str, tuple[str, torch.Tensor]]:
    """``{name: (module path after the swap, original weight)}`` of every name
    ``fake_quantize_model_`` will quantize, in its order. A split linear counts as its halves
    ``<name>_attn`` / ``<name>_mlp``, which become ``<name>.attn`` / ``<name>.mlp`` and hold
    columns ``[:, :N]`` / ``[:, N:]`` of the fused weight (``fake_linear.split_linear_columns``)."""
    found = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight, n = module.weight.detach(), module.out_features
        if spec.is_split(name):
            found[f"{name}_attn"] = (f"{name}.attn", weight[:, :n])
            found[f"{name}_mlp"] = (f"{name}.mlp", weight[:, n:])
        else:
            found[name] = (name, weight)
    return {name: entry for name, entry in found.items() if spec.matches(name)}


def apply_arm(model: nn.Module, model_key: str, arm: str, *, targets=None, fp4_targets=None,
              tp: int = 4) -> dict[str, list[str]]:
    """Swap the target linears of ``model`` in place for ``arm``; returns the names that are
    quantized and, among them, those whose weight is FP4. ``targets``: regexes replacing the
    model's default target set (one group on an fp32 background). ``fp4_targets``: regexes of
    the quantized names an fp4 arm makes FP4, the others staying MXFP8 (default: all of them); one
    that matches no quantized name is refused before anything is replaced, as
    ``fake_quantize_model_`` refuses a dead target. An FP4 layer keeps the MXFP8 layer's fp8
    ``weight`` and ``weight_scale`` buffers, which its forward does not read."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    law = arm.removeprefix("fp4-") if arm.startswith("fp4-") else None
    if fp4_targets and law is None:
        raise ValueError(f"fp4_targets needs an fp4 arm, not {arm!r}")
    if targets and arm == "fp32":
        raise ValueError("the fp32 arm quantizes nothing: it takes no targets")
    if arm == "fp32":
        return {"quantized": [], "fp4": []}
    spec = default_spec(model_key, tp_degree=tp, **({"targets": tuple(targets)} if targets else {}))
    # Taken before the swap: afterwards the model no longer holds the fp32 weights.
    originals = _originals(model, spec)
    fp4 = []
    if law:
        dead = [p for p in fp4_targets or () if not any(re.fullmatch(p, n) for n in originals)]
        if dead:
            raise ValueError(f"fp4_targets match no quantized linear: {'; '.join(dead)}")
        fp4 = [n for n in originals
               if not fp4_targets or any(re.fullmatch(p, n) for p in fp4_targets)]
    names = fake_quantize_model_(model, spec)
    assert names == list(originals), "fake_quantize_model_ quantized other names than expected"
    for name in fp4:
        path, weight = originals[name]
        layer = model.get_submodule(path)
        layer.weight_dequant = fake_quantize_fp4_sharded(weight, layer.k_shards, law)
    return {"quantized": names, "fp4": fp4}


def snapshot_linears(model: nn.Module) -> dict[str, nn.Linear]:
    """The model's nn.Linear modules by name, for :func:`restore`. Take it before any swap."""
    return {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}


def restore(model: nn.Module, saved: dict[str, nn.Linear]) -> None:
    for name, module in saved.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, module)


def flux_toy(heads: int = 24, tokens: int = 1024) -> tuple[nn.Module, dict]:
    """A random FLUX DiT of one dual and one single block, ``heads`` x 128 wide (24: FLUX.1's
    own width), and the keyword arguments of one forward as ``FluxPipeline`` passes them."""
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel(num_layers=1, num_single_layers=1, num_attention_heads=heads,
                                   guidance_embeds=True).eval()
    kwargs = dict(hidden_states=torch.randn(1, tokens, 64), timestep=torch.tensor([0.5]),
                  guidance=torch.tensor([3.5]), pooled_projections=torch.randn(1, 768),
                  encoder_hidden_states=torch.randn(1, 64, 4096), txt_ids=torch.zeros(64, 3),
                  img_ids=torch.zeros(tokens, 3), joint_attention_kwargs={}, return_dict=False)
    return model, kwargs


def qwen_toy() -> tuple[nn.Module, dict]:
    """A random Qwen-Image DiT of two blocks, 256 wide (every target's K per rank at tp4 is a
    multiple of 128: the MLP-down's is the width), and the keyword arguments of one forward as
    ``QwenImagePipeline`` passes them."""
    from diffusers import QwenImageTransformer2DModel

    model = QwenImageTransformer2DModel(num_layers=2, num_attention_heads=2,
                                        joint_attention_dim=64).eval()
    kwargs = dict(hidden_states=torch.randn(1, 256, 64), timestep=torch.tensor([0.5]),
                  guidance=None, encoder_hidden_states_mask=None,
                  encoder_hidden_states=torch.randn(1, 11, 64), img_shapes=[[(1, 16, 16)]],
                  attention_kwargs={}, return_dict=False)
    return model, kwargs


def _self_test(model_key: str, model: nn.Module, kwargs: dict, mixed: tuple[str, ...]) -> None:
    spec = default_spec(model_key, tp_degree=4)
    ref = model(**kwargs)[0]
    assert torch.isfinite(ref).all() and ref.abs().max() > 0
    saved = snapshot_linears(model)

    def find(name: str) -> tuple[nn.Module, torch.Tensor]:
        """The swapped layer and its original weight, by difflet's own split rather than
        apply_arm's lookup."""
        if name in saved:
            return model.get_submodule(name), saved[name].weight
        fused, half = name.rsplit("_", 1)
        attn, mlp = split_linear_columns(saved[fused])
        return getattr(model.get_submodule(fused), half), (attn if half == "attn" else mlp).weight

    def rel(a: torch.Tensor) -> float:
        return float((a - ref).norm() / ref.norm())

    # the mxfp8 reference: fake_quantize_model_ called directly
    names = fake_quantize_model_(model, spec)
    mxfp8 = {n: find(n)[0].weight_dequant for n in names}
    outputs = {"direct": model(**kwargs)[0]}
    restore(model, saved)
    assert torch.equal(model(**kwargs)[0], ref), "restore did not give the original model back"
    splits = [n for n in names if n not in saved]
    print(f"[{model_key}] {len(names)} target linears ({len(splits)} split halves), "
          f"k_shards {sorted({spec.k_shards(n) for n in names})}")

    def check(arm: str, targets=None, fp4_targets=None) -> None:
        """Apply, hold every quantized layer against what it must be, restore."""
        got = apply_arm(model, model_key, arm, targets=targets, fp4_targets=fp4_targets)
        quantized = [n for n in names if not targets or any(re.fullmatch(p, n) for p in targets)]
        fp4 = [n for n in quantized if arm != "mxfp8"
               and (not fp4_targets or any(re.fullmatch(p, n) for p in fp4_targets))]
        assert got == {"quantized": quantized, "fp4": fp4}, got
        assert sum(isinstance(m, FakeQuantMXLinear) for m in model.modules()) == len(quantized)
        for n in quantized:
            layer, weight = find(n)
            want = mxfp8[n]
            if n in fp4:
                want = fake_quantize_fp4_sharded(weight, spec.k_shards(n), arm.removeprefix("fp4-"))
                assert not torch.equal(want, mxfp8[n])
            assert torch.equal(layer.weight_dequant, want), (arm, n)
            # the activation path is the MXFP8 layer's own: block-quantized input, this weight
            x = torch.randn(3, layer.in_features)
            xq = mx.fake_quantize_mx_sharded(x, spec.k_shards(n))
            assert torch.equal(layer(x), F.linear(xq, want, layer.bias)), (arm, n)
        tag = f"{arm} targets={targets} fp4_targets={fp4_targets}"
        outputs[tag] = model(**kwargs)[0]
        restore(model, saved)
        assert torch.equal(model(**kwargs)[0], ref), "restore did not give the original model back"
        print(f"[{model_key}] {tag}: {len(quantized)} quantized, {len(fp4)} fp4, "
              f"output rel L2 vs fp32 {rel(outputs[tag]):.4f}")

    check("mxfp8")
    assert torch.equal(outputs["mxfp8 targets=None fp4_targets=None"], outputs["direct"])
    check("fp4-ocp")
    check("fp4-mse")
    check("fp4-ocp", fp4_targets=list(mixed))  # MXFP8 everywhere, FP4 on some
    check("mxfp8", targets=[mixed[0]])  # one group on the fp32 background
    check("fp4-ocp", targets=[mixed[0]])
    assert len({rel(o) for o in outputs.values()}) == len(outputs) - 1  # only mxfp8 == direct

    untargeted = r"transformer_blocks\.\d+\.attn\.to_out\.0"  # a linear, but never quantized
    assert any(re.fullmatch(untargeted, n) for n in saved)
    for arm, kw in (("fp4-ocp", {"fp4_targets": [r"no\.such\.layer"]}),
                    ("fp4-ocp", {"fp4_targets": [mixed[0], untargeted]}),
                    ("fp4-ocp", {"targets": [mixed[0]], "fp4_targets": [mixed[1]]}),
                    ("mxfp8", {"fp4_targets": [mixed[0]]}),
                    ("fp32", {"targets": [mixed[0]]})):
        try:
            apply_arm(model, model_key, arm, **kw)
        except ValueError as e:
            print(f"[{model_key}] refused {arm} {kw}: {e}")
        else:
            raise AssertionError(f"{arm} {kw} was not refused")
        assert snapshot_linears(model) == saved, "a refused arm changed the model"
    assert torch.equal(model(**kwargs)[0], ref)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    _self_test("flux", *flux_toy(), mixed=(r"single_transformer_blocks\.\d+\.proj_out_attn",
                                           r"transformer_blocks\.\d+\.ff\.net\.2"))
    _self_test("qwen_image", *qwen_toy(), mixed=(r"transformer_blocks\.\d+\.img_mlp\.net\.2",
                                                 r"transformer_blocks\.\d+\.attn\.to_q"))
    print("arms self-test ok")
