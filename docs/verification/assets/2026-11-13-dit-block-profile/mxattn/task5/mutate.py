"""Task 5 mutation sweep: each mutation applied to the worktree file, the new tests run, restored.
Run from the worktree root."""
import os
import pathlib
import subprocess
import sys

A = "difflet/backends/neuron/ops_impl/attention.py"
MX = "difflet/backends/neuron/ops_impl/mx_attention.py"
APP = "difflet/backends/neuron/core/application_base.py"
M = [
    ("dense_bhsd mx", A, "scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False, mx=False)",
     "scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False)"),
    ("cross_attention mx", A, "        tp_out=False,\n        mx=False,\n        **kwargs,",
     "        tp_out=False,\n        **kwargs,"),
    ("joint ulysses mx", A, "bound_max=bound_max, tp_q=True, tp_k=True, tp_out=False, mx=False)",
     "bound_max=bound_max, tp_q=True, tp_k=True, tp_out=False)"),
    ("no smoothing", A, "    k3 = smooth_keys(k3)\n", ""),
    ("head dim", A, "and q.shape[-1] == 128 and", "and"),
    ("qk/pv flags", A, "scale, False, True)", "scale, True, True)"),
    ("scalar exp", MX, "allow_vector_exp=True, **bounds)", "allow_vector_exp=False, **bounds)"),
    ("plain count", MX,
     '    comptime(lambda ctx: _count(ctx.get_local("key").as_python_constant()), lambda: _count(key))',
     "    _count(key)"),
    ("wan mx", "difflet/models/wan/modeling_wan.py", ", mx=not self.is_cross_attention)", ")"),
    ("ltx2 masked mx", "difflet/models/ltx_2/modeling_ltx_2.py",
     "tp_q=True, tp_k=True, tp_out=False, mx=False,", "tp_q=True, tp_k=True, tp_out=False,"),
    ("ltx2 cp mx", "difflet/backends/neuron/ltx_2/transformer.py",
     "tp_out=False, mx=is_self_attention)", "tp_out=False)"),
    ("app check", APP, "        elif self._attn_quant():", "        elif False:"),
    ("app log", APP, "            mx_attention.log_routed_calls()", "            pass"),
]
env = {**os.environ, "DIFFLET_BACKEND": "cpu", "PYTHONPATH": "."}
for name, path, a, b in M:
    p = pathlib.Path(path)
    orig = p.read_text()
    assert orig.count(a) == 1, name
    p.write_text(orig.replace(a, b))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest",
                            "tests/unit/backends/test_neuron_mx_attention.py", "-q",
                            "-p", "no:cacheprovider", "-p", "no:logging"],
                           env=env, capture_output=True, text=True, timeout=900)
        fails = [line.split("::")[1].split(" ")[0] for line in r.stdout.splitlines()
                 if line.startswith("FAILED")]
        print(f"{name}: {len(fails)} failed {fails}", flush=True)
    finally:
        p.write_text(orig)
