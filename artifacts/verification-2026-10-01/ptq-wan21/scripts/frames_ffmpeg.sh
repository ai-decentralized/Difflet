#!/usr/bin/env bash
# Decode frames 0/4/8 of a video with ffmpeg (independent of the imageio/pyav reader) into a 3x1 strip.
cd /home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan
source .venv/bin/activate
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)
IN="$1"; OUT="$2"
"$FF" -v error -i "$IN" -vf "select='eq(n\,0)+eq(n\,4)+eq(n\,8)',scale=416:240,tile=3x1" -vsync 0 -frames:v 1 -y "$OUT"
"$FF" -i "$IN" 2>&1 | grep -E 'Stream|Duration' | cut -c1-160
ls -la "$OUT"
