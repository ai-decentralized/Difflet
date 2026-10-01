#!/usr/bin/env bash
# Replacement for chain2's serve phase: host-VAE decode of the fixed fp8 CLI latents, then the serve smoke with --host-vae.
set -uo pipefail
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
export PYTHONPATH=$PWD
AB=$PWD/artifacts/verification-2026-10-01/ptq-wan21/ab-fixed
echo "=== host decode (fixed fp8 latents) $(date -u +%FT%TZ)"
python - <<'EOF' 2>&1 | grep -vE 'Warning|warn' | tail -3
import sys, time
sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan")
from difflet.cli.orchestrators.wan import _decode_latents_host
base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21"
for d in ("ab-fixed", "ab-wo-fixed"):
    t0 = time.perf_counter()
    _decode_latents_host(f"{base}/{d}/work_fp8_run1/latents.pt", "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                         f"{base}/{d}/fp8_run1_hostvae.mp4", revision="38ec498cb3208fb688890f8cc7e94ede2cbd7f68")
    print(f"[hostdecode] {d}: {time.perf_counter()-t0:.1f}s", flush=True)
EOF
for d in ab-fixed ab-wo-fixed; do
  python scripts/ptq_compare_outputs.py --reference "$PWD/artifacts/verification-2026-10-01/ptq-wan21/ab/bf16_run1_hostvae.mp4" \
    --test "$PWD/artifacts/verification-2026-10-01/ptq-wan21/$d/fp8_run1_hostvae.mp4" \
    --out "$PWD/artifacts/verification-2026-10-01/ptq-wan21/$d/compare_hostvae_fp8_vs_bf16.json" 2>&1 | grep -vE 'Warning|warn' | tail -2
done
echo "HOSTDECODE_DONE"
bash /home/ubuntu/.claude/jobs/b5f130d0/tmp/gate_idle.sh | tee "$PWD/artifacts/verification-2026-10-01/ptq-wan21/gate_before_serve.txt" | tail -1
bash /home/ubuntu/.claude/jobs/b5f130d0/tmp/job_serve.sh
echo "CHAIN_DONE"
echo 0 > /home/ubuntu/.claude/jobs/b5f130d0/tmp/logs/chain2.done   # release the queued probe/full-forward jobs
