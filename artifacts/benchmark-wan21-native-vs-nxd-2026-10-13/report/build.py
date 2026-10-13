"""Inline data.json (+ the ccop 2-step oracle comparison, when present) into index.html."""
import json
from pathlib import Path

R = Path("/home/ubuntu/wan21-bench/report")
data = json.loads((R / "data.json").read_text())
cmp_path = Path("/home/ubuntu/wan21-bench/ccop-verify/oracle/compare.json")
if cmp_path.exists():
    b = next(x for x in json.loads(cmp_path.read_text())["backends"] if x["backend"] == "neuron")
    data["ccopOracle"] = {
        "init": b["init_noise_equal_to_ref"],
        "cosine": b["final"]["vs_fp32"]["cosine"],
        "rel": b["final"]["vs_fp32"]["rel_l1"],
        "control": b["final"]["control"]["rel_l1"],
        "ratio": b["final"]["ratio_vs_control"],
        "update_cosine": b["update"]["vs_fp32"]["cosine"],
        "pass": all(b["checks"].values()),
        "describe": b["final_describe"],
    }
c50 = Path("/home/ubuntu/wan21-bench/ccop-verify/compare_50step.json")
if c50.exists():
    data["ccop50"] = json.loads(c50.read_text())
html = (R / "template.html").read_text().replace(
    "__DATA__", json.dumps(data).replace("</", "<\\/"))
(R / "index.html").write_text(html)
print("index.html", len(html))
