import json
import sys

d = json.load(open(sys.argv[1]))
for k, v in d.items():
    print(k, type(v).__name__, len(v) if hasattr(v, "__len__") else v)
    if isinstance(v, list) and v and isinstance(v[0], dict):
        print("   keys", sorted(v[0].keys())[:60])
        print("   ex", json.dumps(v[0])[:800])
    if isinstance(v, dict):
        print("   subkeys", list(v.keys())[:30])
