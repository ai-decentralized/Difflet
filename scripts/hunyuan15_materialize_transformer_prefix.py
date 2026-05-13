#!/usr/bin/env python3
"""Materialize a HunyuanVideo 1.5 transformer snapshot with the first N blocks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Parent model dir")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--output-dir", required=True, help="Output parent dir to create")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_safetensor_shards(component_dir: Path) -> list[Path]:
    for index_name in ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"):
        index_path = component_dir / index_name
        if index_path.exists():
            index = _load_json(index_path)
            return [component_dir / name for name in sorted(set(index["weight_map"].values()))]
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        path = component_dir / name
        if path.exists():
            return [path]
    raise FileNotFoundError(f"no safetensors shard/index found under {component_dir}")


def _keep_weight(key: str, num_layers: int) -> bool:
    prefix = "transformer_blocks."
    if not key.startswith(prefix):
        return True
    rest = key[len(prefix) :]
    block_id = int(rest.split(".", 1)[0])
    return block_id < num_layers


def main() -> int:
    ensure_runtime_python()
    from safetensors.torch import load_file, save_file

    args = build_parser().parse_args()
    if args.num_layers < 0:
        raise ValueError("--num-layers must be >= 0")

    source = Path(args.source_dir).expanduser().resolve()
    source_transformer = source / args.transformer_subfolder
    output = Path(args.output_dir).expanduser().resolve()
    output_transformer = output / "transformer"
    if output.exists():
        if not args.force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    output_transformer.mkdir(parents=True)

    config = _load_json(source_transformer / "config.json")
    original_layers = int(config["num_layers"])
    if args.num_layers > original_layers:
        raise ValueError(f"--num-layers {args.num_layers} > source num_layers {original_layers}")
    config["num_layers"] = args.num_layers
    (output_transformer / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    kept: dict = {}
    for shard in _iter_safetensor_shards(source_transformer):
        print(f"[hunyuan15-prefix] scan {shard.name}", flush=True)
        state = load_file(str(shard), device="cpu")
        for key, value in state.items():
            if _keep_weight(key, args.num_layers):
                kept[key] = value
    if not kept:
        raise RuntimeError(f"no weights kept from {source_transformer}")

    weights_path = output_transformer / "diffusion_pytorch_model.safetensors"
    save_file(kept, str(weights_path), metadata={"format": "pt"})
    print(
        json.dumps(
            {
                "source_dir": str(source),
                "transformer_subfolder": args.transformer_subfolder,
                "output_dir": str(output),
                "original_layers": original_layers,
                "num_layers": args.num_layers,
                "weights": str(weights_path),
                "num_tensors": len(kept),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
