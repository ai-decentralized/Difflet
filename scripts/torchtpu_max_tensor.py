#!/usr/bin/env python3
"""Largest single tensor per model, from safetensors headers only (no weights read)."""

import glob
import json
import struct

BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U8": 1, "I8": 1, "F64": 8, "BOOL": 1}
MiB = 2 ** 20
MODELS = {
    "Qwen-Image": "/mnt/models/hf/hub/models--Qwen--Qwen-Image/snapshots/*",
    "Wan2.1-14B": "/mnt/models/hf/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/*",
    "FLUX.1-dev": "/mnt/models/hf/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*",
}

for name, root in MODELS.items():
    per_component = {}
    for path in glob.glob(root + "/*/*.safetensors") + glob.glob(root + "/*.safetensors"):
        component = path.split("/")[-2] if path.count("/") > root.count("/") + 1 else "root"
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            nbytes = BYTES.get(meta["dtype"], 2)
            for d in meta["shape"]:
                nbytes *= d
            cur = per_component.get(component)
            if cur is None or nbytes > cur[0]:
                per_component[component] = (nbytes, key, meta["shape"], meta["dtype"])
    print("== %s" % name)
    for comp, (nb, key, shape, dt) in sorted(per_component.items(), key=lambda kv: -kv[1][0]):
        print("   %-16s max %8.1f MiB  %s %s %s" % (comp, nb / MiB, dt, shape, key[:60]))
