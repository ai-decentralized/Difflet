"""MXFP8 post-training quantization of DiT linears (OCP microscaling, W8A8, no calibration).

Backend-neutral: the block-scale law in Trainium3's block layout (``mx``), the spec that names
what is quantized and how (``spec``), the per-model target layers (``targets``), the CPU
reference layer every device result is checked against (``fake_linear``) and the offline
quantized checkpoint (``checkpoint``). The device side lives with each backend
(``difflet/backends/neuron/ops_impl/fp8_linear.py``).

``spec`` and ``targets`` are torch-free so the CLI driver can parse flags and build cache keys
before torchrun starts a rank; the other modules import torch.
"""
