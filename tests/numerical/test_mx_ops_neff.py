import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.neuron
@pytest.mark.numerical
@pytest.mark.parametrize("mx_dtype", ["float8_e4m3fn_x4", "float8_e5m2_x4"])
@pytest.mark.parametrize("k_tiles", [1, 2, 8])
def test_mx_smoke_trainium(tmp_path, k_tiles, mx_dtype):
    if shutil.which("neuron-ls") is None:
        pytest.skip("Neuron runtime is not available")

    metrics_path = tmp_path / f"mx_metrics_{mx_dtype}_{k_tiles}.json"
    cmd = [
        "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python",
        "scripts/mx_smoke.py",
        "--mode",
        "trainium",
        "--k-tiles",
        str(k_tiles),
        "--mx-dtype",
        mx_dtype,
        "--metrics-path",
        str(metrics_path),
    ]
    env = os.environ.copy()
    neuron_bin = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin"
    env["PATH"] = f"{neuron_bin}:{env.get('PATH', '')}"
    subprocess.run(cmd, check=True, env=env)

    metrics = json.loads(Path(metrics_path).read_text())
    assert metrics["mx_dtype"] == mx_dtype
    assert metrics["passed"] is True
    assert metrics["cosine"] >= 0.999
    assert metrics["mean_abs"] <= 0.01
    assert metrics["max_abs"] <= 0.05


@pytest.mark.neuron
@pytest.mark.numerical
def test_hv15_to_q_mx_trainium_real_weight_gate(tmp_path):
    if shutil.which("neuron-ls") is None:
        pytest.skip("Neuron runtime is not available")

    model_dir = Path("/tmp/nova_hunyuan15_real_prefix_1")
    bundle = Path(".nova-cache/hunyuan15_dit_inputs/real_320x512x61_4step.safetensors")
    if not (model_dir / "transformer" / "diffusion_pytorch_model.safetensors").exists():
        pytest.skip("HV-1.5 prefix_1 safetensors artifact is not available")
    if not bundle.exists():
        pytest.skip("HV-1.5 320p real input bundle is not available")

    metrics_path = tmp_path / "hv15_to_q_mx_metrics.json"
    cmd = [
        "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python",
        "scripts/hv15_to_q_mx_probe.py",
        "--model-dir",
        str(model_dir),
        "--bundle",
        str(bundle),
        "--mode",
        "trainium",
        "--rank",
        "0",
        "--m-slice",
        "128",
        "--metrics-out",
        str(metrics_path),
    ]
    env = os.environ.copy()
    neuron_bin = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin"
    env["PATH"] = f"{neuron_bin}:{env.get('PATH', '')}"
    env["NEURON_RT_NUM_CORES"] = "1"
    subprocess.run(cmd, check=True, env=env)

    metrics = json.loads(metrics_path.read_text())
    assert metrics["passed"] is True
    assert metrics["cosine"] >= 0.999
    assert metrics["mean_abs"] <= 0.03
    assert metrics["max_abs"] <= 0.20


@pytest.mark.neuron
@pytest.mark.numerical
@pytest.mark.parametrize("target", ["to_out.0", "ff.net.2"])
def test_hv15_rowparallel_mx_trainium_real_weight_gate(tmp_path, target):
    if shutil.which("neuron-ls") is None:
        pytest.skip("Neuron runtime is not available")

    model_dir = Path("/tmp/nova_hunyuan15_real_prefix_1")
    bundle = Path(".nova-cache/hunyuan15_dit_inputs/real_320x512x61_4step.safetensors")
    if not (model_dir / "transformer" / "diffusion_pytorch_model.safetensors").exists():
        pytest.skip("HV-1.5 prefix_1 safetensors artifact is not available")
    if not bundle.exists():
        pytest.skip("HV-1.5 320p real input bundle is not available")

    metrics_path = tmp_path / f"hv15_{target.replace('.', '_')}_mx_metrics.json"
    cmd = [
        "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python",
        "scripts/hv15_rowparallel_mx_probe.py",
        "--model-dir",
        str(model_dir),
        "--bundle",
        str(bundle),
        "--mode",
        "trainium",
        "--rank",
        "0",
        "--m-slice",
        "128",
        "--target",
        target,
        "--metrics-out",
        str(metrics_path),
    ]
    env = os.environ.copy()
    neuron_bin = "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin"
    env["PATH"] = f"{neuron_bin}:{env.get('PATH', '')}"
    env["NEURON_RT_NUM_CORES"] = "1"
    subprocess.run(cmd, check=True, env=env)

    metrics = json.loads(metrics_path.read_text())
    assert metrics["passed"] is True
    assert metrics["cosine"] >= 0.999
    assert metrics["mean_abs"] <= 0.03
    assert metrics["max_abs"] <= 0.20
