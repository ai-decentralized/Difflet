# Shared environment for campaign jobs.
export PATH=/home/ubuntu/venvs/difflet-neuron/bin:$PATH
export HF_HUB_OFFLINE=1
export CAMPAIGN=/home/ubuntu/campaign/bf16-reduce
export REPO=/home/ubuntu/Difflet/.claude/worktrees/bf16-tp-reduce-qwen-ltx2
export PROMPT="a cinematic shot of a red fox running through a snowy forest"
export QWEN_REV=75e0b4be04f60ec59a75f475837eced720f823b6
export LTX_REV=47da56e2ad66ce4125a9922b4a8826bf407f9d0a
export FLUX_REV=3de623fc3c33e44ffbe2bad470d0f45bccf2eb21
export WAN22_REV=5be7df9619b54f4e2667b2755bc6a756675b5cd7
export WAN21_REV=38ec498cb3208fb688890f8cc7e94ede2cbd7f68
export HV_REV=e8c2aaa66fe3742a32c11a6766aecbf07c56e773

# timed <label> <logfile> <cmd...>: run a command, append "<label> seconds=<s> rc=<rc>" to timings.
timed() {
  local label=$1 log=$2
  shift 2
  local start end rc
  start=$(date +%s.%N)
  "$@" > "$log" 2>&1
  rc=$?
  end=$(date +%s.%N)
  echo "$label seconds=$(echo "$end - $start" | bc) rc=$rc" | tee -a "$CAMPAIGN/timings.txt"
  return $rc
}
