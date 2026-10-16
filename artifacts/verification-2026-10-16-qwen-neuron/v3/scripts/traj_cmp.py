"""Per-step parity on the real CLI trajectory (teacher-forced) and free-run divergence curves."""
import json
import sys

import torch

D = "/home/ubuntu/qwen-v3-work/diag"


def load(name):
    return torch.load(f"{D}/traj-{name}.pt", map_location="cpu")


def rel_l1(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).abs().sum() / b.abs().sum())


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(),
                                                       dim=0))


ref = load("tp4c-free")
res = {"timesteps_dit": [float(t.float().flatten()[0]) for t in ref["timesteps"]]}
fed = {}
for name in ("tp4c-fed", "tp4e-fed", "cp2u-fed"):
    t = load(name)
    assert all(torch.equal(x, y) for x, y in zip(t["inputs"], ref["inputs"])), name
    per = [rel_l1(o, r) for o, r in zip(t["outputs"], ref["outputs"])]
    cs = [cos(o, r) for o, r in zip(t["outputs"], ref["outputs"])]
    fed[name] = {"inputs_equal_to_reference": True,
                 "outputs_all_equal": all(torch.equal(o, r) for o, r in zip(t["outputs"], ref["outputs"])),
                 "noise_pred_rel_l1_per_step": [round(v, 6) for v in per],
                 "noise_pred_cos_per_step": [round(v, 7) for v in cs],
                 "rel_l1_mean": sum(per) / len(per), "rel_l1_max": max(per),
                 "rel_l1_first10_mean": sum(per[:10]) / 10, "rel_l1_last10_mean": sum(per[-10:]) / 10}
res["teacher_forced_vs_tp4_compile"] = fed
if "tp4e-fed" in fed and "cp2u-fed" in fed:
    a, b = fed["cp2u-fed"]["noise_pred_rel_l1_per_step"], fed["tp4e-fed"]["noise_pred_rel_l1_per_step"]
    res["cp2u_over_tp4e_rel_l1_ratio_per_step"] = [round(x / y, 3) if y else None for x, y in zip(a, b)]
free = {}
for name in ("tp4e-free", "cp2u-free"):
    t = load(name)
    per = [rel_l1(x, r) for x, r in zip(t["inputs"], ref["inputs"])]
    final_out = rel_l1(t["outputs"][-1], ref["outputs"][-1])
    free[name] = {"latent_rel_l1_per_step_input": [round(v, 6) for v in per],
                  "first_output_rel_l1": rel_l1(t["outputs"][0], ref["outputs"][0]),
                  "last_output_rel_l1": final_out}
res["free_run_divergence_vs_tp4_compile"] = free
print(json.dumps({k: v for k, v in res.items()}, indent=1)[:6000])
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(json.dumps(res, indent=1))
