#!/usr/bin/env python3
"""Minimal repro for FLUX's VAE decode failing under torch_tpu.

    RuntimeError: view(): cannot create a view of shape [1, 32, 262144] from the
    input tensor of shape [1, 512, 128, 128] and strides [512, 1, 65536, 512]

Three questions, each falsifiable:
  1. Does conv2d on tpu return a channels-last tensor from a contiguous input?
  2. Does group_norm on tpu fail on a channels-last input?
  3. Does .contiguous() before group_norm fix it, and give the CPU answer?
"""

import torch
import torch_tpu  # noqa: F401

dev = torch.device("tpu")
torch.manual_seed(0)
N, C, H, W, G = 1, 512, 128, 128, 32

x_cpu = torch.randn(N, C, H, W)
conv = torch.nn.Conv2d(C, C, 3, padding=1)
norm = torch.nn.GroupNorm(G, C)

print("Q1: does conv2d on tpu return channels-last?")
x = x_cpu.to(dev)
print("    input  contiguous=%s strides=%s" % (x.is_contiguous(), x.stride()))
y = conv.to(dev)(x)
print("    output contiguous=%s strides=%s" % (y.is_contiguous(), y.stride()))
print("    channels_last=%s" % y.is_contiguous(memory_format=torch.channels_last))

print("Q2: does group_norm fail on that output?")
try:
    z = norm.to(dev)(y)
    float(z.sum().cpu())
    print("    NO - it ran")
except RuntimeError as exc:
    print("    YES - %s" % str(exc).splitlines()[0][:110])

print("Q2b: on a tensor made channels-last explicitly, conv out of the picture?")
xcl = x_cpu.to(memory_format=torch.channels_last).to(dev)
print("    strides=%s" % (xcl.stride(),))
try:
    float(norm.to(dev)(xcl).sum().cpu())
    print("    ran")
except RuntimeError as exc:
    print("    FAILED - %s" % str(exc).splitlines()[0][:80])

print("Q3: does .contiguous() fix it, and match CPU?")
try:
    z = norm.to(dev)(y.contiguous())
    ref = norm.cpu()(conv.cpu()(x_cpu))
    got = z.float().cpu()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten().double(), ref.flatten().double(), dim=0).item()
    print("    ran; 1-cos vs CPU = %.3e   max|abs| = %.3e"
          % (1 - cos, (got - ref).abs().max().item()))
except RuntimeError as exc:
    print("    still FAILED - %s" % str(exc).splitlines()[0][:80])
