"""Which DiT block component a profiled instruction belongs to.

``neuron-explorer view --output-format parquet`` writes one ``Instruction.parquet`` row per
device instruction with ``layer`` (the torch module path, ``<Class>[<attr>][<i>]/...``, then
``_<HLO op>``; ``Unknown`` when the compiler kept none), ``hlo_name`` (the HLO instruction),
``hlo_attrs`` (its Python source, ``[{"source_file": ..., "source_line": ...}]``) and
``nki_source_location`` (the kernel or Python line the instruction came from). ``classify`` puts
a row in one of ``COMPONENTS`` by the first rule that holds:

1. ``layer`` is ``Unknown``, or empty with no source: ``unattributed`` (never ``other``).
2. A row whose ``nki_source_location`` is the attention kernel: ``attention``, whatever HLO
   op the profiler files it under (Wan's mxfp8 block: kernel rows under the all-reduce after
   the kernel).
3. The HLO op (``hlo_name``'s, or the one ``layer`` ends with) is a collective: ``collective``.
4. An attention module's own ops (``<...>Attention[<attr>][<i>]_<op>``) and its compiler
   regions (``.../computation_<n>``): the attention custom-call is ``attention``; the rest is
   ``rope`` or ``qk_norm`` when ``hlo_attrs`` or ``nki_source_location`` names the model's source
   for it, else ``attn_glue`` (q pre-scale, bounds, concatenations).
5. The module path, by the model's ordered table below. An MX linear's kernel rows
   (``matmul_mxfp8`` sources) carry the linear's path and so take its component.
6. Anything else: ``other``.

Two choices made here: a LayerNorm inside an adaLN module (``norm1.norm``) is
``norm``, as Qwen-Image's and Wan's plain ``norm1`` are, and the adaLN linear, its SiLU and its
modulate are ``modulation``; the block's own ops (no module path: the adaLN modulate, the gates
and the residual adds) are ``modulation`` too, since that is where Wan and LTX-2 apply theirs.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

COMPONENTS = ("modulation", "norm", "qkv_proj", "qk_norm", "rope", "attn_glue", "attention",
              "out_proj", "mlp_up", "mlp_act", "mlp_down", "collective", "other", "unattributed")

_REPO = Path(__file__).resolve().parents[2]
_COLLECTIVES = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute")
#: nkilib's attention kernels, and Difflet's MXFP8 fork of attention_cte
_ATTENTION_KERNEL = re.compile(r"/nkilib/core/attention/|/ops_impl/mx_attention_kernel\.py:")
_ATTENTION = re.compile(r"(?:^|/)\w*Attention\[(\w+)\]\[\d+\]")
#: an attention module's own ops, and its compiler regions (``.../computation_154``)
_BARE_ATTENTION = re.compile(r"(?:^|/)\w*Attention\[\w+\]\[\d+\](?:_[^/]*|/computation_\d+)$")


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
# capture.sh's configuration keys
_MODELS.update(hunyuan_video_L=_MODELS["hunyuan_video"], hv15_official=_MODELS["hunyuan_video_15"],
               wan21_official=_MODELS["wan"], wan22_official=_MODELS["wan"])


def classify(layer: str, hlo_name: str, nki_source: str, hlo_attrs: str, model: str) -> str:
    """The component of one ``Instruction.parquet`` row of ``model`` (a model or a configuration
    key of ``capture.sh``). A function of its strings alone: classify the unique rows, not every
    instruction (HunyuanVideo-1.5's block has 5.9M rows and 912 distinct ones)."""
    if model not in _MODELS:
        raise ValueError(f"no component rules for model {model!r}; known: {sorted(_MODELS)}")
    rules, sources = _MODELS[model]
    if layer == "Unknown" or not (layer or nki_source):
        return "unattributed"
    # whatever HLO the profiler files it under: Wan's mxfp8 block puts kernel rows under the
    # all-reduce after the kernel (2026-11-13)
    if _ATTENTION_KERNEL.search(nki_source):
        return "attention"
    hlo_op = hlo_name.partition("= ")[2].partition("(")[0]
    layer_op = layer.rpartition("_")[2]
    if hlo_op.startswith(_COLLECTIVES) or layer_op.startswith(_COLLECTIVES):
        return "collective"
    if _BARE_ATTENTION.search(layer):
        if "custom-call" in (hlo_op, layer_op.split(".")[0]):
            return "attention"
        # a compiler region's rows name their source only in nki_source_location
        path, _, line = nki_source.rpartition(":")
        probe = f'{hlo_attrs} "source_file":"{path}","source_line":"{line}"'
        return next((c for p, c in sources if re.search(p, probe)), "attn_glue")
    return next((c for p, c in rules if re.search(p, layer)), "other")


# nkilib 2.14 attention_cte.py: (first, last line, phase), the first range holding the line wins;
# an instruction names its statement's first line. Loads, the P transpose, output write-back and
# buffer setup are "other".
_ATTENTION_PHASES = (
    (3381, 3381, "exp"),  # _exp_impl step 1: exp(s - max) with the row sum, VectorE
    (3389, 3389, "exp"),  # the same on ScalarE (allow_vector_exp off, or an odd tile)
    (3112, 3169, "mask_softmax"),  # _update_max_impl: section / running max, flash correction
    (3299, 3399, "mask_softmax"),  # _exp_impl steps up to 1: -max and the row-sum setup
    (3558, 3594, "mask_softmax"),  # _write_back_impl step 1: row sum and its reciprocal
    (3639, 3660, "mask_softmax"),  # _write_back_impl: flash correction of the earlier output (tp_out)
    (3668, 3675, "mask_softmax"),  # _write_back_impl: flash correction of the earlier output
    (3709, 3735, "mask_softmax"),  # _scale_reciprocal_write_back_impl: output x 1 / row sum
    (3962, 3992, "mask_softmax"),  # _emit_row_max_evict: scale + row max on the MM1 evict
    (4118, 4136, "qk_matmul"),  # _qk_and_max_large_tile_impl step 2: MM1 nc_matmul
    (4138, 4327, "mask_softmax"),  # _qk_and_max_large_tile_impl step 3: mask, scale, row max
    (4453, 4492, "pv_matmul"),  # _pv_large_tile_impl: MM2 nc_matmul and its PSUM evict
)


#: Difflet's fork of attention_cte (the op ``--attn-quant`` routes to): phase by the enclosing
#: function, then by the nisa call a statement makes; the fork's own MX work in three phases of
#: its own: Q and K (one helper, ``_mx_swizzle_quantize``, serves both, so they are not split),
#: V, and P, each with its loads, regroups, transposes, copies and ``quantize_mx``. Functions not
#: named here (loads, setup, output write-back, the dispatchers) are "other".
FORK = "difflet/backends/neuron/ops_impl/mx_attention_kernel.py"
_FORK_PHASES = {  # function: (its phase, {nisa call: phase of that statement})
    "_qk_and_max_large_tile_impl": ("mask_softmax", {"nc_matmul": "qk_matmul",
                                                     "nc_matmul_mx": "qk_matmul"}),
    "_emit_row_max_evict": ("mask_softmax", {}),
    "_update_max_impl": ("mask_softmax", {}),
    "_exp_impl": ("mask_softmax", {"exponential": "exp", "activation": "exp",
                                   "dma_transpose": "other"}),  # the bf16 path's P transpose
    "_write_back_impl": ("mask_softmax", {"dma_copy": "other"}),  # the earlier output's load
    "_scale_reciprocal_write_back_impl": ("mask_softmax", {}),
    "_pv_large_tile_impl": ("pv_matmul", {}),
    "_mx_load_q": ("qk_quantize", {}), "_mx_load_k": ("qk_quantize", {}),
    "_mx_swizzle_quantize": ("qk_quantize", {}),
    "_mx_load_v": ("v_quantize", {}),
    "_mx_quantize_p": ("p_quantize", {}),
}


@functools.lru_cache(maxsize=None)
def _fork_lines(path: str = "") -> dict[int, str]:
    """{line: phase} of the fork's lines in a function of ``_FORK_PHASES``: of the file the
    profile names when it is still there (a frozen tree keeps the captured lines), else of this
    checkout's file as it is now (re-analysing a capture after the fork changed needs its tree)."""
    src = Path(path) if path and Path(path).is_file() else _REPO / FORK
    tree = ast.parse(src.read_text())
    found, lines = set(), {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in _FORK_PHASES:
            continue
        found.add(fn.name)
        phase, calls = _FORK_PHASES[fn.name]
        lines.update(dict.fromkeys(range(fn.lineno, fn.end_lineno + 1), phase))
        for node in ast.walk(fn):
            call = node.value if isinstance(node, ast.Expr) else None
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr in calls):
                lines[node.lineno] = calls[call.func.attr]
    if missing := set(_FORK_PHASES) - found:
        raise LookupError(f"{sorted(missing)} are not in {FORK}: update _FORK_PHASES")
    return lines


def attention_phase(nki_source: str) -> str:
    """``qk_matmul``, ``mask_softmax``, ``exp``, ``pv_matmul``, the fork's ``qk_quantize``,
    ``v_quantize``, ``p_quantize``, or ``other`` for an attention row's ``nki_source_location``."""
    path, _, line = nki_source.rpartition(":")
    if not line.isdigit():
        return "other"
    if path.endswith("/" + FORK):
        return _fork_lines(path).get(int(line), "other")
    if not path.endswith("/nkilib/core/attention/attention_cte.py"):
        return "other"
    return next((p for first, last, p in _ATTENTION_PHASES if first <= int(line) <= last), "other")


def attention_module(layer: str) -> str:
    """The attribute of the attention module a row's layer path runs in (``attn``, ``attn1``,
    ``audio_attn1``, ...), or "" outside one."""
    m = _ATTENTION.search(layer)
    return m.group(1) if m else ""


@functools.lru_cache(maxsize=None)
def _functions(path: str) -> tuple[tuple[int, int, str], ...]:
    """(first line, last line, qualname) of every function in the Python file ``path``: as given,
    or a difflet file of another checkout at its place in this one; () when it cannot be read."""
    p = Path(path)
    if not p.is_file() and "/difflet/" in path:
        p = _REPO / path[path.rindex("/difflet/") + 1:]
    try:
        tree = ast.parse(p.read_text())
    except (OSError, SyntaxError, ValueError):
        return ()
    found = []

    def walk(node: ast.AST, prefix: str) -> None:
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not isinstance(n, ast.ClassDef):
                    found.append((n.lineno, n.end_lineno, prefix + n.name))
                walk(n, f"{prefix}{n.name}.")
    walk(tree, "")
    return tuple(found)


def enclosing_function(path: str, line: int) -> str | None:
    """The qualname of the innermost function of ``path`` that holds ``line`` (the inverse of the
    source rules above), or None (no such function, or the file cannot be read)."""
    inside = [(first, name) for first, last, name in _functions(path) if first <= line <= last]
    return max(inside)[1] if inside else None
