#!/usr/bin/env bash
set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=0
/home/ubuntu/.local/bin/hf download Wan-AI/Wan2.1-T2V-14B-Diffusers --max-workers 8
rc=$?
echo "DOWNLOAD_DONE rc=$rc"
du -sh /home/ubuntu/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers
