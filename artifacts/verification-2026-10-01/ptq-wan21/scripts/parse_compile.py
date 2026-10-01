import json, sys
sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan")
from benchmark.parse_compile import parse_file
base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21/ab/logs"
for arm in sys.argv[1:]:
    print("==", arm)
    print(json.dumps(parse_file(f"{base}/compile_{arm}.log"), indent=1))
