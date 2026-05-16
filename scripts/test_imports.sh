#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="python"
NEURON_VENV="/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference"
NEURON_PYTHON="${NEURON_VENV}/bin/python"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${NEURON_PYTHON}" ]]; then
    PYTHON_BIN="${NEURON_PYTHON}"
  else
    PYTHON_BIN="${DEFAULT_PYTHON}"
  fi
fi

if [[ -d "${NEURON_VENV}/bin" ]]; then
  export PATH="${NEURON_VENV}/bin:${PATH}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

exec "${PYTHON_BIN}" - <<'PY'
import importlib
import importlib.util

modules = [
    "nova",
    "nova.registry",
    "nova.pipeline.nova_pipeline",
    "nova.pipeline.compile_cache",
    "nova.pipeline.precision_schedule",
    "nova.pipeline.parallel_config",
    "nova.models.flux.application",
    "nova.models.hunyuan_video.application",
    "nova.models.hunyuan_video.entry",
    "nova.models.hunyuan_video.modeling_hunyuan_video",
    "nova.models.hunyuan_video.vae.modeling_vae",
    "nova.models.hunyuan_video.pipeline",
    "nova.backends.trainium.hunyuan_video.backbone",
    "nova.backends.trainium.hunyuan_video.backbone15",
    "nova.backends.trainium.hunyuan_video.segmented15",
    "nova.backends.trainium.hunyuan_video.vae",
    "nova.backends.trainium.hunyuan_video.vae15",
    "nova.models.qwen_image.application",
    "nova.models.qwen_image.entry",
    "nova.models.qwen_image.pipeline",
    "nova.backends.trainium.qwen_image.transformer",
    "nova.models.ltx_2.application",
    "nova.models.ltx_2.entry",
    "nova.models.ltx_2.pipeline",
    "nova.backends.trainium.ltx_2.segmented",
    "nova.backends.trainium.ltx_2.transformer",
    "nova.ops.mx",
    "nova.backends.cpu.ops_impl.mx",
    "nova.backends.trainium.ops_impl.mx",
    "nova.backends.trainium.nki_kernels.mx",
    "nova.models.wan.application",
    "nova.models.wan.entry",
    "nova.models.wan.pipeline",
    "nova.models.wan.modeling_wan",
    "nova.models.wan.umt5.modeling_umt5",
    "nova.models.wan.vae.modeling_vae",
    "nova.backends.trainium.wan.text_encoder",
    "nova.backends.trainium.wan.vae",
]

for name in modules:
    importlib.import_module(name)
    print(f"ok import {name}")

for path, name in [
    ("scripts/ltx_2_full_transformer_closure.py", "ltx_2_full_transformer_closure"),
    ("scripts/ltx_2_host_e2e_smoke.py", "ltx_2_host_e2e_smoke"),
    ("scripts/ltx_2_production_block_compile_probe.py", "ltx_2_production_block_compile_probe"),
    ("scripts/ltx_2_segmented_block_compile_probe.py", "ltx_2_segmented_block_compile_probe"),
    ("scripts/ltx_2_segmented_block_parity.py", "ltx_2_segmented_block_parity"),
    ("scripts/ltx_2_segmented_process_block.py", "ltx_2_segmented_process_block"),
    ("scripts/ltx_2_snapshot_report.py", "ltx_2_snapshot_report"),
    ("scripts/ltx_2_trajectory_parity.py", "ltx_2_trajectory_parity"),
    ("scripts/ltx_2_transformer_parity.py", "ltx_2_transformer_parity"),
    ("scripts/hv15_to_q_mx_probe.py", "hv15_to_q_mx_probe"),
    ("scripts/hv15_rowparallel_mx_probe.py", "hv15_rowparallel_mx_probe"),
    ("scripts/hv15_precision_calibration_sweep.py", "hv15_precision_calibration_sweep"),
    ("scripts/hv15_precision_schedule_frontier.py", "hv15_precision_schedule_frontier"),
    ("scripts/hv15_precision_verify.py", "hv15_precision_verify"),
    ("scripts/mx_smoke.py", "mx_smoke"),
]:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"ok import {path}")

import ast
from pathlib import Path

forbidden_modules = {
    "nova.core",
    "nova.utils.compile_env",
    "nova.utils.runtime_env",
    "nova.utils.distributed",
    "nova.utils.snapshot",
}
roots = [Path("nova"), Path("tests"), Path("scripts"), Path("examples")]
violations = []


def is_forbidden(module: str) -> bool:
    return any(module == item or module.startswith(item + ".") for item in forbidden_modules)


for root in roots:
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if is_forbidden(alias.name):
                        violations.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if is_forbidden(module):
                    violations.append((path, node.lineno, module))
                if module == "nova":
                    for alias in node.names:
                        if alias.name == "core":
                            violations.append((path, node.lineno, f"{module}.{alias.name}"))
                if module == "nova.utils":
                    for alias in node.names:
                        candidate = f"{module}.{alias.name}"
                        if candidate in forbidden_modules:
                            violations.append((path, node.lineno, candidate))

if violations:
    for path, line, module in violations:
        print(f"forbidden import {module} at {path}:{line}")
    raise SystemExit(1)

print("ok forbidden import guard")
PY
