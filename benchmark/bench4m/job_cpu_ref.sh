#!/usr/bin/env bash
# One CPU fp32 reference (or smoke) run, meant to run as a systemd --user unit:
#   systemd-run --user --unit=<unit> --collect -- bash job_cpu_ref.sh <model> <name> -- <script args>
# <model>: qwen_image | ltx_2 | wan2_1 | hunyuan_video.  Output and log: B/<model>/cpu_ref/<name>/ (run.log, rc, reference.json, ...).
# One reference at a time: refuses to start while another cpu_ref python process is alive.
set -u
# Host paths are settings (2026-11-03, moved into the repo): BENCH4M_DIR is the data root, the
# AoT venv's python runs the reference, and the reference scripts sit next to this file. Only
# cpu_ref_qwen.py was moved into the repo.
B=${BENCH4M_DIR:?set BENCH4M_DIR (see benchmark/bench4m/README.md)}
PY=${BENCH4M_AOT_VENV:?set BENCH4M_AOT_VENV}/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
model=$1; name=$2; shift 2
[ "${1:-}" = "--" ] && shift
case "$model" in
  qwen_image) script=$HERE/cpu_ref_qwen.py ;;
  ltx_2) script=$HERE/cpu_ref_ltx2.py ;;
  wan2_1) script=$HERE/cpu_ref_wan.py ;;
  hunyuan_video) script=$HERE/cpu_ref_hv.py ;;
  *) echo "unknown model $model" >&2; exit 2 ;;
esac
out=$B/$model/cpu_ref/$name
if [ -e "$out/rc" ]; then echo "$out already has an rc; refusing to overwrite" >&2; exit 2; fi
mkdir -p "$out"
log=$out/run.log
others=$(pgrep -f "/cpu_ref_(qwen|ltx2|wan|hv)\.py" | grep -v "^$$\$" || true)
if [ -n "$others" ]; then
  echo "another CPU reference is running (pids $others); one at a time" | tee -a "$log"; echo 9 > "$out/rc"; exit 9
fi
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
{
  echo "date: $(date -u +%FT%TZ)"
  echo "unit: ${INVOCATION_ID:-none} host: $(hostname)"
  echo "python: $PY"
  echo "script: $script sha256 $(sha256sum "$script" | cut -c1-64)"
  echo "free_disk_gb: $free_gb"
  echo "meminfo: $(grep -E '^(MemTotal|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
  echo "neuron_users: $(fuser /dev/neuron* 2>&1 | tr -s ' ' | tr '\n' ' ')"
  echo "env: HF_HUB_OFFLINE=1 USE_TORCH_XLA=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1"
  echo "cmd: $PY $script $* --out-dir $out"
} >> "$log"
if [ "$free_gb" -lt 20 ]; then echo "free disk below 20 GB" | tee -a "$log"; echo 8 > "$out/rc"; exit 8; fi
# The full-depth LTX-2 smoke peaked at 117.3 GiB VmRSS (124.8 GiB host, no swap): refuse to
# start unless nearly all of the host's memory is available (page cache counts as available).
avail_gib=$(( $(grep '^MemAvailable:' /proc/meminfo | tr -dc 0-9) / 1048576 ))
echo "mem_available_gib: $avail_gib" >> "$log"
if [ "$avail_gib" -lt 112 ]; then echo "MemAvailable ${avail_gib} GiB < 112 GiB" | tee -a "$log"; echo 7 > "$out/rc"; exit 7; fi
export HF_HUB_OFFLINE=1 USE_TORCH_XLA=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /tmp
/usr/bin/time -v "$PY" "$script" "$@" --out-dir "$out" >> "$log" 2>&1
rc=$?
echo "rc: $rc  date: $(date -u +%FT%TZ)" >> "$log"
echo "$rc" > "$out/rc"
exit "$rc"
