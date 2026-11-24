"""The fork of nkilib's ring attention kernel keeps its license and its one functional change:
both ``attention_cte`` calls run the softmax exp on the Scalar engine (Trainium3)."""

import re
from pathlib import Path

KERNEL = Path(__file__).resolve().parents[3] / "difflet/backends/neuron/ops_impl/ring_attention_fwd_kernel.py"


def test_ring_kernel_fork_keeps_the_apache_header_and_names_its_source():
    head = KERNEL.read_text().splitlines()[:30]
    assert any("Copyright Amazon.com, Inc. or its affiliates" in line for line in head)
    assert any("Licensed under the Apache License, Version 2.0" in line for line in head)
    assert any("Difflet: forked from nkilib" in line for line in head)


def test_both_attention_cte_calls_put_the_exp_on_the_scalar_engine():
    src = KERNEL.read_text()
    # The kernel's calls open their argument list on a new line; its docstring's pseudocode does not.
    calls = re.findall(r"= attention_cte\(\n(.*?)\n\s*\)", src, flags=re.S)
    assert len(calls) == 2
    assert all("allow_vector_exp=False" in call for call in calls)
    assert not re.search(r"^\s*from \.", src, flags=re.M)  # absolute imports: the fork lives outside nkilib


CONSTMAX = [KERNEL.parent / "ring_const_max_attention_kernel.py", KERNEL.parent / "ring_const_max_fwd_kernel.py"]


def test_const_max_ring_forks_keep_the_apache_header_and_name_their_source():
    for path in CONSTMAX:
        head = path.read_text().splitlines()[:25]
        assert any("Copyright Amazon.com, Inc. or its affiliates" in line for line in head), path.name
        assert any("Licensed under the Apache License, Version 2.0" in line for line in head), path.name
        assert any("forked from aws-neuron/vllm-omni-neuron v0.29.0.0.2.0" in line for line in head), path.name


def test_const_max_ring_forks_import_absolutely():
    for path in CONSTMAX:
        assert not re.search(r"^\s*from \.", path.read_text(), flags=re.M), path.name
