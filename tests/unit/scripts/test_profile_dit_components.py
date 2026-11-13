"""scripts/profile_dit/components.py: which DiT block component a profiled instruction belongs to.

The rows are real ``Instruction.parquet`` strings (2026-11-13): HunyuanVideo's dual block
(``/home/ubuntu/fp8-bench/hv1_tp4/profsp_V15/blk1``, an MXFP8 capture) and the Task 1 smokes
(Qwen-Image, FLUX dual and single, HunyuanVideo-1.5; bf16). No Wan or LTX-2 parquet exists yet:
their rows are written from the model code and Task 4 adds captured ones. A row put in the wrong
component misstates the breakdown the page draws, so each rule has a case here."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "profile_dit_components", REPO / "scripts" / "profile_dit" / "components.py")
components = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(components)

HV1 = "/home/ubuntu/fp8-bench/hv1_tp4/src-e/difflet/backends/neuron/ops_impl"
WT = "/home/ubuntu/Difflet/.claude/worktrees/mxfp8-sage-attention"
SP = "/opt/aws_neuronx_venv_torch_neuronx_2_14_0/lib/python3.12/site-packages"
DIFFUSERS = "/home/ubuntu/venvs/native-fp8/lib/python3.12/site-packages/diffusers/models"
CTE = f"{SP}/nkilib/core/attention/attention_cte.py"


def attrs(source_file: str, line: int | str) -> str:
    return f'[{{"op_type":"","source_file":"{source_file}","source_line":"{line}"}}]'


def line_of(path: str, text: str) -> int:
    """The line of ``path`` (repo-relative) holding ``text``: rules on a repo function's lines
    follow the file, so the cases find the line the way a capture of today's code names it."""
    return next(i for i, s in enumerate((REPO / path).read_text().splitlines(), 1) if text in s)


QWEN = "difflet/models/qwen_image/modeling_qwen_image.py"
WAN = "difflet/models/wan/modeling_wan.py"
LTX = "difflet/models/ltx_2/modeling_ltx_2.py"

CASES = [
    # HunyuanVideo dual block (hv1 blk1): the brief's four rows ...
    ("AdaLayerNormZero[norm1][0]/ColumnParallelLinear[linear][0]_dot",
     "%dot = dot(%reshape.6, %transpose.110)", f"{HV1}/linear.py:130", attrs(f"{HV1}/linear.py", 130),
     "hunyuan_video", "modulation"),
    ("AdaLayerNormZero[norm1][0]/ColumnParallelLinear[linear][0]_all-gather.115",
     "%all-gather.115 = all-gather(%reshape.114)", "", "", "hunyuan_video", "collective"),
    ("HunyuanVideoAttention[attn][0]_add.330", "%add.330 = add(%multiply.327, %multiply.329)", "",
     attrs(f"{HV1}/embeddings.py", 39), "hunyuan_video", "rope"),
    ("Unknown", "", "", "[]", "hunyuan_video", "unattributed"),
    # ... a collective whose HLO the profiler names as a fused custom-call: the layer's op decides
    ("AdaLayerNormZero[norm1][0]/ColumnParallelLinear[linear][0]_all-gather.115",
     "%custom-call.538 = custom-call(%reshape.537, %Arg_34.35, %Arg_35.36)",
     f"{SP}/torch/distributed/_functional_collectives.py:209", attrs(f"{HV1}/fp8_kernels.py", 122),
     "hunyuan_video", "collective"),
    # the attention custom-call, and a row of it the profiler files under another HLO op
    ("HunyuanVideoAttention[attn][0]_custom-call.411",
     "%custom-call.411 = custom-call(%reshape.266, %reshape.390, %reshape.392, %broadcast.104, %broadcast.407)",
     f"{CTE}:4472", attrs(f"{HV1}/attention.py", 201), "hunyuan_video", "attention"),
    ("HunyuanVideoAttention[attn][0]_add.330",
     "%custom-call.411 = custom-call(%reshape.266, %reshape.390, %reshape.392, %broadcast.104, %broadcast.407)",
     f"{CTE}:4128", attrs(f"{HV1}/attention.py", 201), "hunyuan_video", "attention"),
    # the attention module's own ops that are neither the kernel nor RoPE: the q pre-scale
    ("HunyuanVideoAttention[attn][0]_multiply.33", "%multiply.33 = multiply(%convert.11, %broadcast.129)",
     "", attrs(f"{HV1}/attention.py", 186), "hunyuan_video", "attn_glue"),
    ("HunyuanVideoAttention[attn][0]/ColumnParallelLinear[to_q][0]_dot.1",
     "%dot.1 = dot(%reshape.14, %transpose.215)", f"{HV1}/linear.py:130", attrs(f"{HV1}/linear.py", 130),
     "hunyuan_video", "qkv_proj"),
    ("HunyuanVideoAttention[attn][0]/FP8ColumnParallelLinear[add_k_proj][0]_custom-call.338",
     "%custom-call.338 = custom-call(%reshape.332, %Arg_21.22, %Arg_22.23)",
     f"{SP}/nkilib/experimental/matmul_mxfp8/matmul_mxfp8_generic_api.py:217",
     attrs(f"{HV1}/fp8_kernels.py", 122), "hunyuan_video", "qkv_proj"),
    ("HunyuanVideoAttention[attn][0]/RMSNorm[norm_q][0]_rsqrt.2", "%rsqrt.2 = rsqrt(%add.9)",
     f"{DIFFUSERS}/normalization.py:555", attrs(f"{DIFFUSERS}/normalization.py", 555),
     "hunyuan_video", "qk_norm"),
    ("HunyuanVideoAttention[attn][0]/RowParallelLinear[0][0]_dot.5",
     "%dot.5 = dot(%reshape.74, %transpose.418)", f"{HV1}/linear.py:192", attrs(f"{HV1}/linear.py", 192),
     "hunyuan_video", "out_proj"),
    # MLP up / act / down; the MX linears' kernel rows (matmul_mxfp8) take the linear's component
    ("HunyuanVideoFeedForward[ff][0]/HunyuanVideoGELU[0][0]/FP8ColumnParallelLinear[proj][0]_custom-call.538",
     "%custom-call.538 = custom-call(%reshape.537, %Arg_34.35, %Arg_35.36)",
     f"{SP}/nkilib/experimental/matmul_mxfp8/matmul_mxfp8_blocks.py:272",
     attrs(f"{HV1}/fp8_kernels.py", 122), "hunyuan_video", "mlp_up"),
    ("HunyuanVideoFeedForward[ff][0]/HunyuanVideoGELU[0][0]_multiply.550",
     "%multiply.550 = multiply(%add.549, %multiply.542)", "",
     attrs("/home/ubuntu/fp8-bench/hv1_tp4/src-e/difflet/models/hunyuan_video/modeling_hunyuan_video.py", 171),
     "hunyuan_video", "mlp_act"),
    ("HunyuanVideoFeedForward[ff][0]/FP8RowParallelLinear[2][0]_custom-call.552",
     "%custom-call.552 = custom-call(%reshape.551, %Arg_36.37, %Arg_37.38)",
     f"{SP}/nkilib/experimental/matmul_mxfp8/matmul_mxfp8_blocks.py:267",
     attrs(f"{HV1}/fp8_kernels.py", 122), "hunyuan_video", "mlp_down"),
    # norms, the adaLN's own LayerNorm included; the block's own ops (gate, residual) are modulation
    ("LayerNorm[norm2][0]_reduce.463", "%reduce.463 = reduce(%convert.458, %constant.102)", "", "",
     "hunyuan_video", "norm"),
    ("AdaLayerNormZero[norm1][0]/LayerNorm[norm][0]_reduce.140",
     "%reduce.140 = reduce(%multiply.135, %constant.102)", f"{DIFFUSERS}/normalization.py:169",
     attrs(f"{DIFFUSERS}/normalization.py", 169), "hunyuan_video", "norm"),
    ("_add.573", "%add.573 = add(%add.451, %multiply.572)", "",
     attrs("/home/ubuntu/fp8-bench/hv1_tp4/src-e/difflet/models/hunyuan_video/modeling_hunyuan_video.py", 663),
     "hunyuan_video", "modulation"),
    # a row with no module path but the attention kernel's source is the kernel's
    ("", "", f"{CTE}:4472", attrs(f"{WT}/difflet/backends/neuron/ops_impl/attention.py", 201),
     "hv15_official", "attention"),
    ("", "", "", "", "hv15_official", "unattributed"),
    # Qwen-Image: RoPE is lines of modeling_qwen_image.py (line 67 at capture), the bounds are glue
    ("Attention[attn][0]_multiply.319", "%multiply.319 = multiply(%reshape.163, %broadcast.318)", "",
     attrs(f"{WT}/{QWEN}", line_of(QWEN, "return (x.float() * cos + x_rotated.float() * sin)")),
     "qwen_image", "rope"),
    ("Attention[attn][0]_add.379", "%add.379 = add(%convert.372, %constant.65)", "",
     attrs(f"{WT}/{QWEN}", 544), "qwen_image", "attn_glue"),
    ("Sequential[img_mod][0]/ColumnParallelLinear[1][0]_dot.91", "%dot.1 = dot(%reshape, %transpose.90)",
     f"{WT}/difflet/backends/neuron/ops_impl/linear.py:130", "", "qwen_image", "modulation"),
    ("FeedForward[img_mlp][0]/GELU[0][0]_tanh.483", "%tanh.483 = tanh(%multiply.482)",
     f"{DIFFUSERS}/activations.py:85", attrs(f"{DIFFUSERS}/activations.py", 85), "qwen_image", "mlp_act"),
    ("_add.500", "%add.500 = add(%add.424, %multiply.499)", "",
     attrs(f"{DIFFUSERS}/transformers/transformer_qwenimage.py", 733), "qwen_image", "modulation"),
    # FLUX single-stream block: MLP and output projection are the block's own linears
    ("ColumnParallelLinear[proj_mlp][0]_dot.103", "%dot.5 = dot(%reshape.17, %transpose.101)", "", "",
     "flux", "mlp_up"),
    ("GELU[act_mlp][0]_tanh.112", "%tanh.112 = tanh(%multiply.111)", "", "", "flux", "mlp_act"),
    ("RowParallelLinear[proj_out_mlp][0]_dot.246", "%dot.6 = dot(%reshape.52, %transpose.244)", "", "",
     "flux", "mlp_down"),
    ("RowParallelLinear[proj_out_attn][0]_dot.243", "%dot.4 = dot(%reshape.46, %transpose.241)", "", "",
     "flux", "out_proj"),
    ("NeuronAdaLayerNormZeroSingle[norm][0]_add.100", "%add.100 = add(%multiply.95, %broadcast.99)",
     f"{WT}/difflet/layers/normalization.py:83", "", "flux", "modulation"),
    # HunyuanVideo-1.5: diffusers' apply_rotary_emb
    ("Attention[attn][0]_multiply.283", "%multiply.283 = multiply(%convert.282, %broadcast.265)",
     f"{DIFFUSERS}/embeddings.py:1231", attrs(f"{DIFFUSERS}/embeddings.py", 1231), "hv15_official", "rope"),
    # Wan (model code): the global q/k RMSNorm and RoPE run inside the attention module
    ("WanAttention[attn1][0]_multiply.12", "%multiply.12 = multiply(%convert.10, %broadcast.11)", "",
     attrs(f"{WT}/{WAN}", line_of(WAN, "x_normed = x.float() * torch.rsqrt(global_sq / full_dim + eps)")),
     "wan21_official", "qk_norm"),
    ("WanAttention[attn1][0]_add.20", "%add.20 = add(%multiply.18, %multiply.19)", "",
     attrs(f"{WT}/difflet/backends/neuron/ops_impl/embeddings.py", 48), "wan22_official", "rope"),
    ("WanAttention[attn2][0]/ColumnParallelLinear[to_k][0]_dot.3", "%dot.3 = dot(%reshape.2, %transpose.1)",
     "", "", "wan21_official", "qkv_proj"),
    ("WanFeedForward[ffn][0]/ColumnParallelLinear[net_in][0]_dot.7", "%dot.7 = dot(%reshape.6, %transpose.5)",
     "", "", "wan21_official", "mlp_up"),
    ("WanFeedForward[ffn][0]_tanh.9", "%tanh.9 = tanh(%multiply.8)", "", "", "wan21_official", "mlp_act"),
    ("WanFeedForward[ffn][0]/RowParallelLinear[net_out][0]_dot.11", "%dot.11 = dot(%reshape.10, %transpose.9)",
     "", "", "wan21_official", "mlp_down"),
    ("FP32LayerNorm[norm1][0]_reduce.4", "%reduce.4 = reduce(%convert.3, %constant.2)", "", "",
     "wan21_official", "norm"),
    # LTX-2 (model code): split RoPE and the global q/k RMSNorm in modeling_ltx_2.py
    ("LTX2Attention[audio_attn1][0]_multiply.30", "%multiply.30 = multiply(%reshape.28, %broadcast.29)", "",
     attrs(f"{WT}/{LTX}", line_of(LTX, "out = split_x * cos_u")), "ltx_2", "rope"),
    ("LTX2Attention[attn1][0]_multiply.40", "%multiply.40 = multiply(%convert.38, %rsqrt.39)", "",
     attrs(f"{WT}/{LTX}", line_of(LTX, "x_normed = x.float() * torch.rsqrt(global_sq / full_dim + eps)")),
     "ltx_2", "qk_norm"),
    ("LTX2Attention[attn2][0]/RowParallelLinear[0][0]_dot.50", "%dot.50 = dot(%reshape.49, %transpose.48)",
     "", "", "ltx_2", "out_proj"),
    ("FeedForward[audio_ff][0]/GELU[0][0]/ColumnParallelLinear[proj][0]_dot.60",
     "%dot.60 = dot(%reshape.59, %transpose.58)", "", "", "ltx_2", "mlp_up"),
]


@pytest.mark.parametrize("layer, hlo_name, nki_source, hlo_attrs, model, expected", CASES)
def test_classify(layer, hlo_name, nki_source, hlo_attrs, model, expected):
    assert components.classify(layer, hlo_name, nki_source, hlo_attrs, model) == expected


def test_components_are_the_contract():
    assert components.COMPONENTS == (
        "modulation", "norm", "qkv_proj", "qk_norm", "rope", "attn_glue", "attention", "out_proj",
        "mlp_up", "mlp_act", "mlp_down", "collective", "other", "unattributed")


def test_an_attributed_row_no_rule_names_is_other():
    assert components.classify(
        "HunyuanVideoFeedForward[ff][0]_concatenate.560",
        "%custom-call.411 = custom-call(%reshape.266, %reshape.390, %reshape.392, %broadcast.104, %broadcast.407)",
        f"{SP}/torch/_utils.py:1195", attrs(f"{HV1}/attention.py", 201), "hunyuan_video_L") == "other"


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="sd3"):
        components.classify("Unknown", "", "", "", "sd3")


@pytest.mark.parametrize("line, phase", [
    (4128, "qk_matmul"),     # MM1 nc_matmul
    (3381, "mask_softmax"),  # exp(s - max) with the fused row sum
    (3984, "mask_softmax"),  # scale + row max on the MM1 evict
    (4287, "mask_softmax"),  # range_select: the bounds mask with the row max
    (3137, "mask_softmax"),  # section / running max
    (3587, "mask_softmax"),  # 1 / row sum
    (3668, "mask_softmax"),  # flash correction of the earlier sections' output
    (3729, "mask_softmax"),  # output x 1 / row sum
    (4472, "pv_matmul"),     # MM2 nc_matmul
    (4492, "pv_matmul"),     # MM2 PSUM evict
    (3008, "other"),         # V load
    (3420, "other"),         # P transpose for MM2
    (3772, "other"),         # output write-back
])
def test_attention_phase(line, phase):
    assert components.attention_phase(f"{CTE}:{line}") == phase


def test_attention_phase_of_a_source_outside_the_kernel_is_other():
    assert components.attention_phase(f"{DIFFUSERS}/normalization.py:3420") == "other"
    assert components.attention_phase("") == "other"


@pytest.mark.parametrize("layer, module", [
    ("HunyuanVideoAttention[attn][0]_custom-call.411", "attn"),
    ("WanAttention[attn2][0]/ColumnParallelLinear[to_q][0]_dot", "attn2"),
    ("LTX2Attention[audio_attn1][0]_add.3", "audio_attn1"),
    ("NeuronFluxAttention[attn][0]/RMSNorm[norm_k][0]_rsqrt.3", "attn"),
    ("LayerNorm[norm2][0]_reduce.463", ""),
    ("Unknown", ""),
])
def test_attention_module(layer, module):
    assert components.attention_module(layer) == module
