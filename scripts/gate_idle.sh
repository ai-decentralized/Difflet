#!/usr/bin/env bash
# Idle gate before any device/metrics phase: exits 0 only when nothing else is loading the host.
# Expected processes: this session's claude processes and (optionally) the jobs named in $ALLOW.
ALLOW="${ALLOW:-}"  # regex of allowed foreign processes (e.g. a CPU job of ours)
fail=0
echo "--- $(date -u +%FT%TZ) load: $(cut -d' ' -f1-3 /proc/loadavg)"
# 1. Neuron devices: any attached process?
if neuron-ls -j 2>/dev/null | grep -q '"pid"'; then
  echo "BUSY: a process holds a NeuronCore"; neuron-ls; fail=1
else
  echo "neuron: no process attached"
fi
# 2. CPU-heavy foreign processes (>20% CPU), excluding claude's own harness and allowed job names.
foreign=$(ps -eo pid,pcpu,etimes,args --sort=-pcpu | awk 'NR>1 && $2>20' | grep -vE 'claude|gate_idle' | { [ -n "$ALLOW" ] && grep -vE "$ALLOW" || cat; })
if [ -n "$foreign" ]; then echo "BUSY: foreign CPU-heavy processes:"; echo "$foreign"; fail=1; else echo "cpu: no foreign heavy processes"; fi
# 3. Any python outside this job's venv/jobs (other sessions' work)
other_py=$(ps -eo pid,args | grep -E '[p]ython' | grep -vE 'networkd-dispatcher|unattended-upgrade|verify-ptq-fp8-wan|/home/ubuntu/.claude/jobs/b5f130d0|\.venvs/hf' )
if [ -n "$other_py" ]; then echo "BUSY: python processes from elsewhere:"; echo "$other_py"; fail=1; else echo "python: none from other sessions"; fi
# 4. 1-minute load below the core count
one=$(cut -d' ' -f1 /proc/loadavg); cores=$(nproc)
awk -v l="$one" -v c="$cores" 'BEGIN{ if (l+0 > c*0.5) { print "WARN: load " l " on " c " cores"; } else print "load ok (" l ")" }'
[ $fail -eq 0 ] && echo "GATE: IDLE" || echo "GATE: BUSY"
exit $fail
