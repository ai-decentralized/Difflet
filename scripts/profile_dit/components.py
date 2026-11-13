"""Which DiT block component a profiled instruction belongs to (DiT block profile plan, Task 2).

``neuron-explorer view --output-format parquet`` writes one ``Instruction.parquet`` row per
device instruction with ``layer`` (the torch module path, ``<Class>[<attr>][<i>]/...``, then
``_<HLO op>``; ``Unknown`` when the compiler kept none), ``hlo_name`` (the HLO instruction),
``hlo_attrs`` (its Python source, ``[{"source_file": ..., "source_line": ...}]``) and
``nki_source_location`` (the kernel or Python line the instruction came from). ``classify`` puts
a row in one of ``COMPONENTS`` by the first rule that holds:

1. ``layer`` is ``Unknown``, or empty with no source: ``unattributed`` (never ``other``).
2. The HLO op (``hlo_name``'s, or the one ``layer`` ends with) is a collective: ``collective``.
3. An attention module's own ops (``<...>Attention[<attr>][<i>]_<op>``): the attention
   custom-call (by HLO op or kernel source) is ``attention``; the rest is ``rope`` or ``qk_norm`` when ``hlo_attrs`` names the
   model's source for it, else ``attn_glue`` (q pre-scale, bounds, concatenations).
4. Any other row whose ``nki_source_location`` is the attention kernel: ``attention``.
5. The module path, by the model's ordered table below. An MX linear's kernel rows
   (``matmul_mxfp8`` sources) carry the linear's path and so take its component.
6. Anything else: ``other``.

Two choices the plan leaves open: a LayerNorm inside an adaLN module (``norm1.norm``) is
``norm``, as Qwen-Image's and Wan's plain ``norm1`` are, and the adaLN linear, its SiLU and its
modulate are ``modulation``; the block's own ops (no module path: the adaLN modulate, the gates
and the residual adds) are ``modulation`` too, since that is where Wan and LTX-2 apply theirs.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

COMPONENTS = ("modulation", "norm", "qkv_proj", "qk_norm", "rope", "attn_glue", "attention",
              "out_proj", "mlp_up", "mlp_act", "mlp_down", "collective", "other", "unattributed")

_REPO = Path(__file__).resolve().parents[2]
_COLLECTIVES = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute")
_ATTENTION_KERNEL = re.compile(r"/nkilib/core/attention/")
_ATTENTION = re.compile(r"(?:^|/)\w*Attention\[(\w+)\]\[\d+\]")
_BARE_ATTENTION = re.compile(r"(?:^|/)\w*Attention\[\w+\]\[\d+\]_[^/]*$")


def _source(path: str, qualname: str) -> str:
    """An ``hlo_attrs`` pattern for the lines of ``qualname`` in the repo file ``path``, as the
    file is now (a capture of other code names other lines)."""
    node = ast.parse((_REPO / path).read_text())
    for name in qualname.split("."):
        node = next((n for n in node.body if getattr(n, "name", None) == name), None)
        if node is None:
            raise LookupError(f"{qualname} is not in {path}: update the classifier's source rules")
    lines = "|".join(str(i) for i in range(node.lineno, node.end_lineno + 1))
    return rf'{re.escape(path)}","source_line":"(?:{lines})"'


# Attention-module rows: (hlo_attrs pattern, component); the rest of them is attn_glue.
_ROPE_FILE = (r'/embeddings\.py"', "rope")  # difflet's ops_impl and diffusers' apply_rotary_emb
_WAN, _LTX = "difflet/models/wan/modeling_wan.py", "difflet/models/ltx_2/modeling_ltx_2.py"
_QWEN_SOURCES = ((_source("difflet/models/qwen_image/modeling_qwen_image.py", "_apply_qwen_rope_real"),
                  "rope"),)
_WAN_SOURCES = ((_source(_WAN, "WanAttention._global_rms_norm"), "qk_norm"),
                (_source(_WAN, "apply_rotary_emb_half_split"), "rope"), _ROPE_FILE)
_LTX_SOURCES = ((_source(_LTX, "_LTX2TPAttnProcessor._global_rms_norm"), "qk_norm"),
                (_source(_LTX, "_difflet_apply_split_rotary_emb"), "rope"))

# Module-path rules: (pattern searched in layer, component), first match wins.
_ATTN = (
    (r"\[(?:to_q|to_k|to_v|add_q_proj|add_k_proj|add_v_proj|fused_qkv|fused_add_qkv)\]", "qkv_proj"),
    (r"\[norm_(?:q|k|added_q|added_k)\]", "qk_norm"),
    (r"Attention\[\w+\]\[\d+\]/\w*RowParallelLinear\[0\]|\[(?:to_out|to_add_out|proj_out_attn)\]",
     "out_proj"),
)
#: diffusers' FeedForward (GELU's ``proj``, ``net[2]``) and HunyuanVideo's / FLUX's twins of it.
_FEEDFORWARD = (
    (r"GELU\[0\]\[0\]/\w*Linear\[proj\]", "mlp_up"),
    (r"GELU\[0\]\[0\]_[^/]*$", "mlp_act"),
    (r"FeedForward\[\w+\]\[\d+\]/\w*Linear\[2\]", "mlp_down"),
)
#: FLUX's and HunyuanVideo's single-stream block.
_SINGLE = ((r"\[proj_mlp\]", "mlp_up"), (r"\[act_mlp\]", "mlp_act"), (r"\[proj_out_mlp\]", "mlp_down"))
_WAN_FFN = ((r"\[net_in\]", "mlp_up"), (r"WanFeedForward\[\w+\]\[\d+\]_[^/]*$", "mlp_act"),
            (r"\[net_out\]", "mlp_down"))
_NORM_AND_MODULATION = (
    (r"(?:LayerNorm|RMSNorm)\[\w+\]\[\d+\]_[^/]*$", "norm"),  # a norm module's own ops
    (r"AdaLayerNorm|\[(?:norm1\w*|img_mod|txt_mod)\]", "modulation"),
    (r"^_", "modulation"),  # the block's own ops: modulate, gates, residual adds
)

_MODELS = {  # model -> (module-path rules, attention-module sources)
    "qwen_image": ((*_ATTN, *_FEEDFORWARD, *_NORM_AND_MODULATION), _QWEN_SOURCES),
    "flux": ((*_ATTN, *_FEEDFORWARD, *_SINGLE, *_NORM_AND_MODULATION), (_ROPE_FILE,)),
    "hunyuan_video": ((*_ATTN, *_FEEDFORWARD, *_SINGLE, *_NORM_AND_MODULATION), (_ROPE_FILE,)),
    "hunyuan_video_15": ((*_ATTN, *_FEEDFORWARD, *_NORM_AND_MODULATION), (_ROPE_FILE,)),
    "wan": ((*_ATTN, *_WAN_FFN, *_NORM_AND_MODULATION), _WAN_SOURCES),
    "ltx_2": ((*_ATTN, *_FEEDFORWARD, *_NORM_AND_MODULATION), _LTX_SOURCES),
}
# the plan's configuration keys
_MODELS.update(hunyuan_video_L=_MODELS["hunyuan_video"], hv15_official=_MODELS["hunyuan_video_15"],
               wan21_official=_MODELS["wan"], wan22_official=_MODELS["wan"])


def classify(layer: str, hlo_name: str, nki_source: str, hlo_attrs: str, model: str) -> str:
    """The component of one ``Instruction.parquet`` row of ``model`` (a model or a configuration
    key of the plan). A function of its strings alone: classify the unique rows, not every
    instruction (HunyuanVideo-1.5's block has 5.9M rows and 912 distinct ones)."""
    if model not in _MODELS:
        raise ValueError(f"no component rules for model {model!r}; known: {sorted(_MODELS)}")
    rules, sources = _MODELS[model]
    if layer == "Unknown" or not (layer or nki_source):
        return "unattributed"
    hlo_op = hlo_name.partition("= ")[2].partition("(")[0]
    layer_op = layer.rpartition("_")[2]
    if hlo_op.startswith(_COLLECTIVES) or layer_op.startswith(_COLLECTIVES):
        return "collective"
    if _BARE_ATTENTION.search(layer):
        if "custom-call" in (hlo_op, layer_op.split(".")[0]) or _ATTENTION_KERNEL.search(nki_source):
            return "attention"
        return next((c for p, c in sources if re.search(p, hlo_attrs)), "attn_glue")
    if _ATTENTION_KERNEL.search(nki_source):
        return "attention"
    return next((c for p, c in rules if re.search(p, layer)), "other")


# nkilib 2.14 attention_cte.py: (first, last line, phase); an instruction names its statement's
# first line. Loads, the P transpose, output write-back and buffer setup are "other".
_ATTENTION_PHASES = (
    (3112, 3169, "mask_softmax"),  # _update_max_impl: section / running max, flash correction
    (3299, 3399, "mask_softmax"),  # _exp_impl steps up to 1: -max, exp(s - max) with the row sum
    (3558, 3594, "mask_softmax"),  # _write_back_impl step 1: row sum and its reciprocal
    (3639, 3660, "mask_softmax"),  # _write_back_impl: flash correction of the earlier output (tp_out)
    (3668, 3675, "mask_softmax"),  # _write_back_impl: flash correction of the earlier output
    (3709, 3735, "mask_softmax"),  # _scale_reciprocal_write_back_impl: output x 1 / row sum
    (3962, 3992, "mask_softmax"),  # _emit_row_max_evict: scale + row max on the MM1 evict
    (4118, 4136, "qk_matmul"),  # _qk_and_max_large_tile_impl step 2: MM1 nc_matmul
    (4138, 4327, "mask_softmax"),  # _qk_and_max_large_tile_impl step 3: mask, scale, row max
    (4453, 4492, "pv_matmul"),  # _pv_large_tile_impl: MM2 nc_matmul and its PSUM evict
)


def attention_phase(nki_source: str) -> str:
    """``qk_matmul``, ``mask_softmax``, ``pv_matmul`` or ``other`` for an attention row's
    ``nki_source_location``."""
    path, _, line = nki_source.rpartition(":")
    if not path.endswith("/nkilib/core/attention/attention_cte.py") or not line.isdigit():
        return "other"
    return next((p for first, last, p in _ATTENTION_PHASES if first <= int(line) <= last), "other")


def attention_module(layer: str) -> str:
    """The attribute of the attention module a row's layer path runs in (``attn``, ``attn1``,
    ``audio_attn1``, ...), or "" outside one."""
    m = _ATTENTION.search(layer)
    return m.group(1) if m else ""
