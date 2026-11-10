"""Fused joint Q/K RMSNorm + RoPE (+ V gather) for HunyuanVideo 1.5 on the CUDA backend.

A HunyuanVideo 1.5 DiT block attends over the joint sequence ``[video | context]``. Each
stream has its own Q/K RMSNorm weights; video Q/K are rotated by RoPE, context Q/K are not.
``fused_joint_qk_norm_rope`` writes the joint Q, K and V that attention consumes in one Triton
launch, instead of per-stream norm passes, RoPE passes and three ``cat`` copies. Inputs are
read through their strides (on CUDA they are views of one packed QKV projection output); the
outputs are new contiguous ``[B, S0 + S1, H, D]`` tensors.

Numerics, per output token and head, in FP32 (V is copied unchanged):

1. Sum of squares over ``D/8`` lanes of eight elements. Each lane is an in-order chain of
   explicit FMAs from zero; the lanes are then added pairwise, adjacent lanes first, as a tree.
2. ``inv = rsqrt(ss / D + eps)``, the device's fast rsqrt.
3. ``n = (x * inv) * w``: two FP32 multiplies, then one rounding to BF16 (nearest-even).
4. Adjacent pairs ``(n[2i], n[2i+1])`` are rotated by the table row's ``cos[i]``/``sin[i]``
   with separate FP32 products and one add or subtract, stored as BF16. Context rows of the
   table are identity rows (cos 1, sin 0), so their normalised values pass through.

The kernel is compiled with ``enable_fp_fusion=False``: the step-1 FMAs are the only fused
operations. It takes BF16 q/k/v, weights and table with ``D == 128``, on CUDA with Triton.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU installs: the model keeps its prefix attention path
    triton = tl = None

HAS_TRITON = triton is not None


if HAS_TRITON:

    @triton.jit
    def _copy_tile(src, dst, mask):
        tl.store(dst, tl.load(src, mask=mask), mask=mask)

    @triton.jit
    def _rms_rotate_heads(
        src, weight, table_row, dst, mask, eps, HEAD_DIM: tl.constexpr, BLOCK_H: tl.constexpr
    ):
        """Steps 1-4 for every head of one token; ``src``/``dst`` are [BLOCK_H, D] pointers."""
        tl.static_assert(HEAD_DIM == 128)
        x = tl.load(src, mask=mask, other=0.0).to(tl.float32)

        # 1. Lane j is x[8j .. 8j+7]: an in-order FMA chain from zero. Selecting element e of
        #    each lane is exact (one value summed with zeros).
        lanes = tl.reshape(x, [BLOCK_H, HEAD_DIM // 8, 8])
        element = tl.arange(0, 8)[None, None, :]
        acc = tl.zeros([BLOCK_H, HEAD_DIM // 8], dtype=tl.float32)
        for e in tl.static_range(8):
            x_e = tl.sum(tl.where(element == e, lanes, 0.0), axis=2)
            acc = tl.fma(x_e, x_e, acc)
        #    Then adjacent lanes pairwise, 16 -> 8 -> 4 -> 2 -> 1.
        lo, hi = tl.split(tl.reshape(acc, [BLOCK_H, 8, 2]))
        acc = lo + hi
        lo, hi = tl.split(tl.reshape(acc, [BLOCK_H, 4, 2]))
        acc = lo + hi
        lo, hi = tl.split(tl.reshape(acc, [BLOCK_H, 2, 2]))
        acc = lo + hi
        lo, hi = tl.split(acc)
        ss = lo + hi

        # 2. D is a power of two, so ss * (1/D) equals the IEEE ss / D exactly; Triton's FP32
        #    "/" is an approximate division.
        inv = tl.rsqrt(ss * (1.0 / HEAD_DIM) + eps)

        # 3. Two FP32 multiplies, one BF16 rounding (nearest-even), back to FP32.
        w = tl.load(weight + tl.arange(0, HEAD_DIM)).to(tl.float32)
        n = (x * inv[:, None]) * w[None, :]
        n = n.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

        # 4. Rotate adjacent pairs by the row's [cos | sin]: separate products, no FMA.
        a, b = tl.split(tl.reshape(n, [BLOCK_H, HEAD_DIM // 2, 2]))
        half = tl.arange(0, HEAD_DIM // 2)
        cos = tl.load(table_row + half).to(tl.float32)[None, :]
        sin = tl.load(table_row + HEAD_DIM // 2 + half).to(tl.float32)[None, :]
        even = a * cos - b * sin
        odd = b * cos + a * sin
        out = tl.reshape(tl.join(even, odd), [BLOCK_H, HEAD_DIM])
        tl.store(dst, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)

    @triton.jit
    def _fused_joint_qk_norm_rope_kernel(
        q0,
        k0,
        v0,
        q1,
        k1,
        v1,
        wq0,
        wk0,
        wq1,
        wk1,
        table,
        q_out,
        k_out,
        v_out,
        s0,
        s1,
        heads,
        stride0_b,
        stride0_s,
        stride0_h,
        stride1_b,
        stride1_s,
        stride1_h,
        eps,
        HEAD_DIM: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """Axis 0: joint output row over all samples; axis 1: 0 = Q, 1 = K, 2 = V.

        A stream's q/k/v share one set of strides (unit last stride). Offsets are int64.
        """
        row = tl.program_id(0).to(tl.int64)
        tensor = tl.program_id(1)
        sample = row // (s0 + s1)
        token = row % (s0 + s1)
        head = tl.arange(0, BLOCK_H).to(tl.int64)[:, None]
        dim = tl.arange(0, HEAD_DIM)[None, :]
        mask = (head < heads) & (dim < HEAD_DIM)
        dst = (row * heads + head) * HEAD_DIM + dim
        table_row = table + row * HEAD_DIM
        if token < s0:
            src0 = sample * stride0_b + token * stride0_s + head * stride0_h + dim
            if tensor == 0:
                _rms_rotate_heads(
                    q0 + src0, wq0, table_row, q_out + dst, mask, eps, HEAD_DIM, BLOCK_H
                )
            elif tensor == 1:
                _rms_rotate_heads(
                    k0 + src0, wk0, table_row, k_out + dst, mask, eps, HEAD_DIM, BLOCK_H
                )
            else:
                _copy_tile(v0 + src0, v_out + dst, mask)
        else:
            src1 = sample * stride1_b + (token - s0) * stride1_s + head * stride1_h + dim
            if tensor == 0:
                _rms_rotate_heads(
                    q1 + src1, wq1, table_row, q_out + dst, mask, eps, HEAD_DIM, BLOCK_H
                )
            elif tensor == 1:
                _rms_rotate_heads(
                    k1 + src1, wk1, table_row, k_out + dst, mask, eps, HEAD_DIM, BLOCK_H
                )
            else:
                _copy_tile(v1 + src1, v_out + dst, mask)


def _new_outputs(q0, q1):
    joint = (q0.shape[0], q0.shape[1] + q1.shape[1], *q0.shape[2:])
    return tuple(torch.empty(joint, dtype=q0.dtype, device=q0.device) for _ in range(3))


def _with_shared_strides(q, k, v):
    """The kernel reads a stream's q/k/v through one set of strides with a unit last stride
    (true of views split from one packed projection); anything else is made contiguous."""
    if q.stride() == k.stride() == v.stride() and q.stride(-1) == 1:
        return q, k, v
    return q.contiguous(), k.contiguous(), v.contiguous()


def _launch_kernel(q0, k0, v0, q1, k1, v1, wq0, wk0, wq1, wk1, table, eps):
    batch, s0, heads, head_dim = q0.shape
    s1 = q1.shape[1]
    outputs = _new_outputs(q0, q1)
    if outputs[0].numel() == 0:
        return outputs
    q0, k0, v0 = _with_shared_strides(q0, k0, v0)
    q1, k1, v1 = _with_shared_strides(q1, k1, v1)
    weights = [w.contiguous() for w in (wq0, wk0, wq1, wk1)]
    with torch.cuda.device(q0.device):
        _fused_joint_qk_norm_rope_kernel[(batch * (s0 + s1), 3)](
            q0,
            k0,
            v0,
            q1,
            k1,
            v1,
            *weights,
            table.contiguous(),
            *outputs,
            s0,
            s1,
            heads,
            *q0.stride()[:3],
            *q1.stride()[:3],
            float(eps),
            HEAD_DIM=head_dim,
            BLOCK_H=triton.next_power_of_2(heads),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return outputs


@torch.library.custom_op(
    "difflet_hv15::fused_joint_qk_norm_rope", mutates_args=(), device_types="cuda"
)
def fused_joint_qk_norm_rope(
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    wq0: torch.Tensor,
    wk0: torch.Tensor,
    wq1: torch.Tensor,
    wk1: torch.Tensor,
    table: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joint ``(q, k, v)`` for attention, each a new contiguous ``[B, S0 + S1, H, D]``.

    Stream 0 (video, ``[B, S0, H, D]``) rows come first in each sample, then stream 1
    (context). Q/K are RMS-normalised with their stream's weights (``wq0, wk0`` video,
    ``wq1, wk1`` context) and rotated by ``table``'s row for that joint token
    (``[B * (S0 + S1), D]``, ``[cos | sin]``, identity rows for context); V is copied.
    Inputs are BF16 with ``D == 128``; the caller checks that before taking this path.
    """
    return _launch_kernel(q0, k0, v0, q1, k1, v1, wq0, wk0, wq1, wk1, table, eps)


@fused_joint_qk_norm_rope.register_fake
def _(q0, k0, v0, q1, k1, v1, wq0, wk0, wq1, wk1, table, eps):
    joint = (q0.shape[0], q0.shape[1] + q1.shape[1], *q0.shape[2:])
    return q0.new_empty(joint), q0.new_empty(joint), q0.new_empty(joint)


def pack_rope_table(
    cos: torch.Tensor, sin: torch.Tensor, context_rows: int, dtype: torch.dtype
) -> torch.Tensor:
    """One sample's ``[S_video + context_rows, D]`` table: rows ``[cos | sin]``, one value per
    interleaved pair, cast to ``dtype``, then ``context_rows`` identity rows (cos 1, sin 0).

    ``cos``/``sin`` are ``[S_video, D]`` tables that repeat each angle for both members of a
    pair, as Diffusers' HunyuanVideo 1.5 RoPE builds them.
    """
    half = cos.shape[-1] // 2
    video = torch.cat((cos[:, 0::2], sin[:, 0::2]), dim=-1).to(dtype)
    identity = torch.zeros((context_rows, 2 * half), dtype=dtype, device=cos.device)
    identity[:, :half] = 1
    return torch.cat((video, identity))


__all__ = ["HAS_TRITON", "fused_joint_qk_norm_rope", "pack_rope_table"]
