"""The quantized checkpoint directory, written from the tiny diffusers-layout Qwen-Image DiT."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from difflet.backends._torch_megatron.checkpoint import build_weight_map
from difflet.backends.neuron.core import weights_bf16
from difflet.backends.neuron.core.weights_bf16 import ensure_bf16_copy
from difflet.quant import mx
from difflet.quant.checkpoint import (
    BF16_MANIFEST,
    MANIFEST,
    check_quantized_dir,
    ensure_quantized_dir,
    prepare_quantized_checkpoint,
    quantized_digest,
    quantized_dir,
    quantized_linear_names,
    read_manifest,
    source_fingerprint,
    write_quantized_dir,
)
from difflet.quant.fake_linear import FakeQuantMXLinear, fake_quantize_model_
from difflet.quant.spec import QuantSpec
from difflet.quant.targets import default_spec

pytest.importorskip("diffusers")

from tests.unit.backends._neuron_qwen_toy import (  # noqa: E402
    TINY_PACKED,
    tiny_qwen_inputs,
    write_tiny_qwen_model,
)

#: 2 blocks x (6 attention projections + 4 MLP linears); to_out / to_add_out stay bf16.
TINY_TARGET_COUNT = 2 * 10


def _diffusers_model(transformer_dir):
    from diffusers import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(
        str(transformer_dir), torch_dtype=torch.float32).eval().requires_grad_(False)


def _forward(model, dtype=torch.float32):
    hidden, timestep, text = tiny_qwen_inputs(dtype=dtype)
    with torch.no_grad():
        return model(hidden_states=hidden, timestep=timestep, encoder_hidden_states=text,
                     encoder_hidden_states_mask=None, img_shapes=[[(1, *TINY_PACKED)]],
                     return_dict=False)[0]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_qwen")
    reference = write_tiny_qwen_model(root)
    transformer_dir = root / "transformer"
    return {"root": root, "transformer_dir": transformer_dir, "spec": default_spec("qwen_image"),
            "reference": reference}


def test_write_quantized_dir_layout_and_manifest(tiny, tmp_path):
    out = tmp_path / "mxfp8"
    manifest = write_quantized_dir(tiny["transformer_dir"], out, tiny["spec"], shard_bytes=64 * 1024)
    assert (out / MANIFEST).is_file() and (out / "config.json").is_file()
    assert manifest["tensors"]["quantized"] == TINY_TARGET_COUNT
    assert len(manifest["files"]) > 1, "the small shard size must produce several shards"
    assert manifest["spec"] == tiny["spec"].to_dict()
    assert manifest["schema"] == tiny["spec"].schema
    assert quantized_linear_names(out) == manifest["quantized_linears"]
    assert manifest["bytes"] < manifest["source"]["bytes"]

    weight_map = build_weight_map(out)
    source_map = build_weight_map(tiny["transformer_dir"])
    index = json.loads((out / "diffusion_pytorch_model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(weight_map)
    # Every source key is present; every target weight brings its block scales.
    assert set(source_map) <= set(weight_map)
    extra = set(weight_map) - set(source_map)
    assert len(extra) == TINY_TARGET_COUNT
    assert all(k.endswith(".weight_scale") for k in extra)


def test_quantized_tensors_match_the_fake_quant_reference(tiny, tmp_path):
    from safetensors import safe_open

    out = tmp_path / "mxfp8"
    write_quantized_dir(tiny["transformer_dir"], out, tiny["spec"])
    model = _diffusers_model(tiny["transformer_dir"])
    plain = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    fake_quantize_model_(model, tiny["spec"])
    fakes = {n: m for n, m in model.named_modules() if isinstance(m, FakeQuantMXLinear)}
    assert len(fakes) == TINY_TARGET_COUNT

    weight_map = build_weight_map(out)
    handles = {}
    try:
        def get(key):
            path = weight_map[key]
            if path not in handles:
                handles[path] = safe_open(path, framework="pt").__enter__()
            return handles[path].get_tensor(key)

        for name, fake in fakes.items():
            w = get(f"{name}.weight")
            assert w.dtype == mx.FP8_DTYPE and torch.equal(w, fake.weight)
            s = get(f"{name}.weight_scale")
            assert s.dtype == torch.uint8 and s.shape == (w.shape[0], w.shape[1] // 32)
            assert torch.equal(s, fake.weight_scale)
            assert torch.equal(get(f"{name}.bias"), plain[name].bias.detach())
        # A kept tensor is copied byte for byte, in its own dtype.
        mod = get("transformer_blocks.0.img_mod.1.weight")
        assert mod.dtype == torch.float32
        assert torch.equal(mod, plain["transformer_blocks.0.img_mod.1"].weight.detach())
        assert "transformer_blocks.0.attn.to_out.0.weight_scale" not in weight_map
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)


def test_fake_quant_model_stays_close_to_the_reference(tiny):
    model = _diffusers_model(tiny["transformer_dir"])
    fake_quantize_model_(model, tiny["spec"])
    out = _forward(model)
    ref = tiny["reference"]
    cosine = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    assert cosine > 0.99, cosine
    assert not torch.equal(out, ref)


def test_row_parallel_targets_are_quantized_per_rank_k_range(tiny, tmp_path):
    from safetensors import safe_open

    spec = default_spec("qwen_image", tp_degree=2)
    out = tmp_path / "tp2"
    write_quantized_dir(tiny["transformer_dir"], out, spec)
    weight_map = build_weight_map(out)
    key = "transformer_blocks.0.img_mlp.net.2"
    with safe_open(weight_map[f"{key}.weight"], framework="pt") as h:
        q, s = h.get_tensor(f"{key}.weight"), h.get_tensor(f"{key}.weight_scale")
    with safe_open(build_weight_map(tiny["transformer_dir"])[f"{key}.weight"], framework="pt") as h:
        w = h.get_tensor(f"{key}.weight").float()
    expect_q, expect_s = mx.quantize_mx_sharded(w, 2)
    assert torch.equal(q, expect_q) and torch.equal(s, expect_s)
    assert not torch.equal(s, mx.quantize_mx(w)[1])  # another block layout than tp1's


def test_a_target_whose_per_rank_k_has_no_layout_is_refused_by_name(tmp_path):
    """The writer runs on the driver before any rank builds a layer, so this is the refusal a
    ``difflet compile --quant`` user meets first: it names the layer, its K and the tp."""
    from safetensors.torch import save_file

    src = tmp_path / "src"
    src.mkdir()
    save_file({"net.2.weight": torch.randn(16, 384), "proj.weight": torch.randn(16, 192)},
              str(src / "model.safetensors"))
    row = QuantSpec(targets=(r"net\.2",), row_parallel=(r"net\.2",), tp_degree=2)
    with pytest.raises(ValueError, match=r"net\.2 \(K=384 split over tp=2\): K=192 must be a "
                                         r"multiple of 128"):
        write_quantized_dir(src, tmp_path / "tp2", row)
    assert not (tmp_path / "tp2").exists()  # nothing half-written
    with pytest.raises(ValueError, match=r"proj: K=192 must be a multiple of 128"):
        write_quantized_dir(src, tmp_path / "plain", QuantSpec(targets=("proj",)))
    # Whole at tp=1 the same K has a layout.
    write_quantized_dir(src, tmp_path / "tp1", dataclasses.replace(row, tp_degree=1))


def test_a_target_pattern_that_matches_no_linear_weight_is_refused_before_writing(tiny, tmp_path):
    """A dead pattern (a ``DIFFLET_QUANT_TARGETS`` typo) is refused by name, verbatim, before
    anything is quantized or written: the directory would otherwise hold less than the spec
    claims, and every rank would refuse it (``quantize_module_``) only after the driver spent
    minutes writing it. A pattern that matches only a 1-D weight (a norm) is dead too: the
    writer quantizes 2-D weights, the ranks their linears."""
    spec = tiny["spec"]
    for dead in (r"transformer_blocks\.\d+\.attn\.to_qkv", r"transformer_blocks\.\d+\.attn\.norm_q"):
        with pytest.raises(ValueError, match="match no linear weight") as exc:
            write_quantized_dir(tiny["transformer_dir"], tmp_path / "out",
                                dataclasses.replace(spec, targets=(*spec.targets, dead)))
        assert str(exc.value).endswith(f"{tiny['transformer_dir']}: {dead}")
        assert list(tmp_path.iterdir()) == []  # nothing written, no temporary sibling
    both = (r"transformer_blocks\.\d+\.attn\.to_qkv", r"transformer_blocks\.\d+\.attn\.norm_q")
    with pytest.raises(ValueError) as exc:  # every dead pattern, as the env var separates them
        write_quantized_dir(tiny["transformer_dir"], tmp_path / "out",
                            dataclasses.replace(spec, targets=both))
    assert str(exc.value).endswith(f": {both[0]}; {both[1]}")


def test_check_and_ensure(tiny, tmp_path):
    cache = tmp_path / "cache"
    spec, src = tiny["spec"], tiny["transformer_dir"]
    target = quantized_dir(cache, src, spec)
    assert target.parent.name == "transformer" and target.name.startswith("mxfp8-")
    with pytest.raises(ValueError, match="never written"):
        ensure_quantized_dir(cache, src, spec, create=False)
    made = ensure_quantized_dir(cache, src, spec, create=True)
    assert made == target and (made / MANIFEST).is_file()
    # Idempotent: a second call finds the valid directory and writes nothing.
    stamp = (made / MANIFEST).stat().st_mtime_ns
    assert ensure_quantized_dir(cache, src, spec, create=False) == made
    assert (made / MANIFEST).stat().st_mtime_ns == stamp
    assert prepare_quantized_checkpoint(src, spec, create=False, cache_root=cache) == made

    check_quantized_dir(made, src, spec)
    # The compute path does not change the weights: dequant and nki share the directory.
    other = dataclasses.replace(spec, compute="dequant")
    assert quantized_dir(cache, src, other) == target
    check_quantized_dir(made, src, other)
    with pytest.raises(ValueError, match="another quantization spec"):
        check_quantized_dir(made, src, dataclasses.replace(spec, tp_degree=2))
    assert quantized_dir(cache, src, dataclasses.replace(spec, tp_degree=2)) != target
    with pytest.raises(ValueError, match="schema"):
        check_quantized_dir(made, src, dataclasses.replace(spec, schema=spec.schema + 1))
    with pytest.raises(FileExistsError):
        write_quantized_dir(src, made, spec)
    assert read_manifest(made)["spec"]["format"] == "mxfp8"


def test_a_narrower_target_set_is_another_directory(tiny, tmp_path):
    cache = tmp_path / "cache"
    spec = tiny["spec"]
    narrow = QuantSpec(targets=spec.targets[:1], row_parallel=spec.row_parallel)
    assert quantized_dir(cache, tiny["transformer_dir"], narrow) != \
        quantized_dir(cache, tiny["transformer_dir"], spec)


def test_source_fingerprint_ignores_mtime_but_not_content(tiny, tmp_path):
    import os
    import shutil

    src = tiny["transformer_dir"]
    before = source_fingerprint(src)
    copy = tmp_path / "copy"
    shutil.copytree(src, copy)
    for path in copy.iterdir():
        os.utime(path, (0, 0))
    assert source_fingerprint(copy) == before
    shard = sorted(copy.glob("*.safetensors"))[0]
    shard.write_bytes(shard.read_bytes() + b"\0")
    assert source_fingerprint(copy) != before


def _file_identity(model_dir):
    """The fingerprint as it was before a bf16 copy's manifest joined it: every safetensors
    file's name, size and blob name, every index's sha256. Computed here, not by the function
    under test, because every Qwen, FLUX and LTX-2 digest hangs on it staying byte-identical."""
    entries = []
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        entries.append([path.name, path.stat().st_size, os.path.basename(os.path.realpath(path))])
    for path in sorted(Path(model_dir).glob("*.safetensors.index.json")):
        entries.append([path.name, hashlib.sha256(path.read_bytes()).hexdigest()])
    return entries


def test_a_bf16_copy_is_fingerprinted_with_its_manifest(tiny, tmp_path):
    """A bf16 copy's safetensors are written by the copy, so they carry no hub blob name; its
    manifest (the fp32 source's file names, sizes and mtimes) is what ties a quantized
    directory built from the copy to the hub snapshot, and joins the fingerprint as the last
    entry. A directory without one (every hub checkpoint) is fingerprinted as before."""
    src = tiny["transformer_dir"]
    assert source_fingerprint(src) == _file_identity(src)
    # The toy's digest from before the manifest entry existed (its files' names and sizes and
    # the Qwen targets fix it, not the random weights): the literal every Qwen, FLUX and LTX-2
    # directory written since hangs on, which a formula mirrored in the test cannot pin.
    assert quantized_digest(tiny["spec"], source_fingerprint(src)) == "e11f9d28946c0cb7"
    copy = ensure_bf16_copy(src, cache_root=tmp_path / "cache")
    manifest = copy / BF16_MANIFEST
    fingerprint = source_fingerprint(copy)
    assert fingerprint[-1] == [BF16_MANIFEST, hashlib.sha256(manifest.read_bytes()).hexdigest()]
    assert fingerprint[:-1] == _file_identity(copy)
    # The copy's files without the manifest: a plain bf16 checkpoint (LTX-2's), as before.
    plain = tmp_path / "plain"
    shutil.copytree(copy, plain)
    (plain / BF16_MANIFEST).unlink()
    assert source_fingerprint(plain) == _file_identity(plain) == fingerprint[:-1]
    # Another fp32 source behind the same bf16 bytes is another manifest, so another fingerprint.
    stale = json.loads(manifest.read_text())
    stale["files"]["config.json"][1] += 1
    manifest.write_text(json.dumps(stale))
    changed = source_fingerprint(copy)
    assert changed[:-1] == fingerprint[:-1] and changed[-1] != fingerprint[-1]


def test_a_quantized_directory_built_from_a_copy_follows_the_copy_s_source(tiny, tmp_path):
    """``difflet compile`` remakes the copy when its fp32 source changed, with a new manifest:
    the quantized directory built from the old copy is then another checkpoint's (refused by
    the check, another digest for the next compile), never silently reused."""
    cache = tmp_path / "cache"
    src = tmp_path / "snap" / "transformer"
    shutil.copytree(tiny["transformer_dir"], src)
    copy = ensure_bf16_copy(src, cache_root=cache)
    spec = tiny["spec"]
    made = ensure_quantized_dir(cache, copy, spec, create=True)
    source = read_manifest(made)["source"]
    assert source["fingerprint"][-1][0] == BF16_MANIFEST
    # The copy's manifest stays behind (the quantized directory is no bf16 copy of anything);
    # the fp32 snapshot it named is recorded in the quant manifest instead.
    assert not (made / BF16_MANIFEST).exists() and (made / "config.json").is_file()
    assert source["bf16_copy_of"] == str(src.resolve())
    check_quantized_dir(made, copy, spec)
    os.utime(src / "config.json", ns=(1, 1))  # the source changed under the copy
    assert ensure_bf16_copy(src, cache_root=cache) == copy  # remade in place, new manifest
    with pytest.raises(ValueError, match="built from another checkpoint"):
        check_quantized_dir(made, copy, spec)
    assert quantized_dir(cache, copy, spec) != made


def test_a_hub_checkpoint_s_manifest_names_no_copy_source(tiny, tmp_path):
    manifest = write_quantized_dir(tiny["transformer_dir"], tmp_path / "mxfp8", tiny["spec"])
    assert "bf16_copy_of" not in manifest["source"]


def test_the_copy_s_manifest_name_is_one_constant():
    """``source_fingerprint`` names the copy's manifest without importing the backend (the
    neuron package's ``__init__`` loads the runtime), so the quant module owns the name and the
    copy writer takes it from there: one object, not two literals that could drift."""
    assert weights_bf16.MANIFEST is BF16_MANIFEST


def test_a_copy_built_directory_sits_under_its_fp32_source_s_label(tiny, tmp_path):
    """A bf16 copy lives under ``weights-bf16/<snapshot>/<component>``, which names no repo:
    the quantized directory built from it takes the label of the hub snapshot the copy's
    manifest names, beside where the fp32 directory's own would sit, at another digest."""
    cache = tmp_path / "cache"
    snapshot = "0123456789abcdef0123456789abcdef01234567"
    src = (tmp_path / "hub" / "models--Wan-AI--Wan2.1-T2V-14B-Diffusers" / "snapshots" / snapshot
           / "transformer")
    shutil.copytree(tiny["transformer_dir"], src)
    copy = ensure_bf16_copy(src, cache_root=cache)
    assert copy.parent.name == snapshot
    spec = tiny["spec"]
    of_copy, of_fp32 = quantized_dir(cache, copy, spec), quantized_dir(cache, src, spec)
    assert of_copy.parent.parent.name == "Wan-AI--Wan2.1-T2V-14B-Diffusers__01234567"
    assert of_copy.parent == of_fp32.parent and of_copy != of_fp32
    assert of_copy.parent.parent.parent == cache / "quantized"
