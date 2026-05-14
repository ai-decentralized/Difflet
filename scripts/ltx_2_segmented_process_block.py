#!/usr/bin/env python3
"""Worker for one LTX-2 segmented block forward in a fresh Neuron process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    if Path(sys.executable) == NEURON_PYTHON or not NEURON_PYTHON.exists():
        return
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        env = os.environ.copy()
        env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--input-tensors", required=True)
    parser.add_argument("--output-tensors", required=True)
    parser.add_argument("--block-index", type=int, required=True)
    return parser


def _dtype(name: str):
    import torch

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype in runtime config: {name!r}")


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()

    import torch

    from nova.backends.trainium.ltx_2.segmented import LTX2SegmentedTransformerApplication
    from nova.models.ltx_2.application import create_ltx_2_transformer_config

    runtime = json.loads(Path(args.runtime_config).read_text(encoding="utf-8"))
    dtype = _dtype(runtime["dtype"])
    config = create_ltx_2_transformer_config(
        model_path=runtime["model_dir"],
        world_size=int(runtime["tp_degree"]),
        tp_degree=int(runtime["tp_degree"]),
        dtype=dtype,
        height=int(runtime["height"]),
        width=int(runtime["width"]),
        num_frames=int(runtime["num_frames"]),
        text_seq_len=int(runtime["text_seq_len"]),
        audio_text_seq_len=int(runtime["audio_text_seq_len"]),
        audio_num_frames=int(runtime["audio_num_frames"]),
        frame_rate=float(runtime["frame_rate"]),
        batch_size=1,
    )
    segmented = LTX2SegmentedTransformerApplication(
        model_path=str(Path(runtime["model_dir"]) / "transformer"),
        config=config,
        block_load_mode="streaming",
    )
    block = segmented.block
    component_path = Path(runtime["compiled_model_path"]) / "transformer_block"
    block.load(str(component_path), skip_warmup=True)
    block.reload_block_weights(int(args.block_index))

    tensors = torch.load(args.input_tensors, map_location="cpu", weights_only=False)
    inputs = tuple(tensor.contiguous() for tensor in tensors["inputs"])
    with torch.no_grad():
        hidden_states, audio_hidden_states = block(*inputs)
    output_path = Path(args.output_tensors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden_states": hidden_states.detach().cpu(),
            "audio_hidden_states": audio_hidden_states.detach().cpu(),
        },
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
