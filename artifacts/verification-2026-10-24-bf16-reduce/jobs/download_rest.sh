#!/usr/bin/env bash
# Download Flux, Wan 2.2, HunyuanVideo, Wan 2.1 at pinned revisions (allow-listed).
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
unset HF_HUB_OFFLINE
cd /home/ubuntu
mkdir -p "$CAMPAIGN/downloads"
difflet download --model-id black-forest-labs/FLUX.1-dev --revision "$FLUX_REV" > "$CAMPAIGN/downloads/flux.log" 2>&1; echo "flux rc=$?"
difflet download --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision "$WAN22_REV" > "$CAMPAIGN/downloads/wan22.log" 2>&1; echo "wan22 rc=$?"
difflet download --model-id hunyuanvideo-community/HunyuanVideo --revision "$HV_REV" > "$CAMPAIGN/downloads/hv.log" 2>&1; echo "hv rc=$?"
difflet download --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision "$WAN21_REV" > "$CAMPAIGN/downloads/wan21.log" 2>&1; echo "wan21 rc=$?"
df -h / | tail -1
