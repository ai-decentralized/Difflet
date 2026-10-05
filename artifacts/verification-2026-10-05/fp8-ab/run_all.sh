#!/usr/bin/env bash
# Serialise every device job of the 2026-10-05 session: the full A/B first, then the pad512/256
# screen. Two compiles must never overlap on this host (shared device, and the vendor
# ModelBuilder rmtree's /tmp/nxd_model scratch at trace start).
set -u
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm -rf /tmp/nxd_model
bash "${D}/job_ab.sh"
bash "${D}/../fp8-screen/job_screen2.sh" >> "${D}/../fp8-screen/job_screen2.out" 2>&1
echo "=== $(date -u +%T) ALL_DONE" >> "${D}/job_ab.out"
