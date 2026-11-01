# Wan 2.2 dual-expert (AOT) device campaign: shared environment. Source, don't run.
export WT=/home/ubuntu/Difflet/.claude/worktrees/wan-aot-parity-plan
export VENV=/home/ubuntu/venvs/difflet-neuron
export PATH=$VENV/bin:$PATH
export DIFFLET_BACKEND=trainium
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export C=/home/ubuntu/campaigns/wan-dual-2026-10-31
export R=$C/root
export DIFFLET_SHARED_WEIGHTS_DIR=$R/_shared_weights
export NEURON_COMPILE_CACHE_URL=$R/neuron-compile-cache
export MODEL=Wan-AI/Wan2.2-T2V-A14B-Diffusers
export REV=5be7df9619b54f4e2667b2755bc6a756675b5cd7
export PROMPT='a cinematic shot of a red fox running through a snowy forest'
export SHAPE="--height 480 --width 832 --num-frames 9"
unset NEURON_RT_VIRTUAL_CORE_SIZE
# SRC selects the code under test: the branch worktree by default, main for baselines.
export SRC=${SRC:-$WT}
export PYTHONPATH=$SRC
mkdir -p "$R" "$C/logs" "$C/out"
cd "$SRC"
