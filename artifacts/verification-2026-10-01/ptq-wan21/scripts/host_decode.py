"""Decode the saved DiT latents of both arms with the diffusers Wan VAE on the host (CPU)."""
import sys, time
sys.path.insert(0, "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan")
from difflet.cli.orchestrators.wan import _decode_latents_host

base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21/ab"
for arm in ("bf16", "fp8"):
    t0 = time.perf_counter()
    _decode_latents_host(f"{base}/work_{arm}_run1/latents.pt", "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                         f"{base}/{arm}_run1_hostvae.mp4", revision="38ec498cb3208fb688890f8cc7e94ede2cbd7f68")
    print(f"[hostdecode] {arm}: {time.perf_counter() - t0:.1f}s -> {base}/{arm}_run1_hostvae.mp4", flush=True)
