#!/usr/bin/env bash
# Stop every launched job (each launch.sh job is its own setsid process group) and any stragglers.
L=${PTQ_JOBS:-/home/ubuntu/ptq-jobs}/logs
for name in verify_wan22 hv_probe verify_flux2 verify_ltx2b verify_qwen2 verify_hv2; do
  if [ -f "$L/$name.pid" ]; then
    pid=$(cat "$L/$name.pid")
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null && echo "stopped $name (pgid $pid)" || echo "could not signal $name ($pid)"
    else
      echo "$name not running"
    fi
  fi
done
sleep 5
for pat in 'job_verify_queue.sh' 'job_hv_probe.sh' 'ptq_model_verify.sh' 'benchmark.bench' 'benchmark.cold_warm_e2e' 'bin/difflet compile' 'bin/difflet generate' 'bin/neuronx-cc' 'difflet.cli.main'; do
  for p in $(pgrep -f "$pat"); do kill -TERM "$p" 2>/dev/null && echo "killed straggler $p ($pat)"; done
done
sleep 5
for pat in 'bin/neuronx-cc' 'bin/difflet' 'benchmark.bench' 'benchmark.cold_warm_e2e'; do
  for p in $(pgrep -f "$pat"); do kill -KILL "$p" 2>/dev/null && echo "force-killed $p ($pat)"; done
done
echo "--- remaining difflet/neuron processes:"
pgrep -af 'difflet|neuronx-cc|benchmark\.' | grep -v stop_all | cut -c1-120 || true
echo "--- neuron device usage:"
neuron-ls 2>/dev/null | tail -8 || true
