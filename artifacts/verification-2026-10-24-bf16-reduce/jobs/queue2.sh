#!/usr/bin/env bash
# Device queue 2: extra LTX-2 fp32 warm run, rank-0 profiles of the four DiT
# NEFFs, then Task 11 (Flux, Wan 2.2, HunyuanVideo: compile fp32+bf16, quick
# measurement). Steps with a marker are skipped on restart.
source /home/ubuntu/campaign/bf16-reduce/jobs/env.sh
cd /home/ubuntu
C=/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47
mark() { touch "$CAMPAIGN/runs/q2_$1.done"; echo "queue2: $1 done"; }
done_() { [ -e "$CAMPAIGN/runs/q2_$1.done" ]; }

done_ ltx2_extra || { bash "$CAMPAIGN/jobs/measure.sh" ltx2 fp32 extra warm4 && mark ltx2_extra; }

for p in qwen_fp32:MODULE_e75a2dee8db307ad6163+15b553c6 qwen_bf16:MODULE_2705a21040bba6dacbfb+96a32ed9 \
         ltx2_fp32:MODULE_8521223016edca772e7f+9de06980 ltx2_bf16:MODULE_20da57192ef0a91533f2+9de06980; do
  label=${p%%:*} mod=${p#*:}
  done_ "profile_$label" || { bash "$CAMPAIGN/jobs/profile.sh" "$label" "$C/$mod/model.neff"; mark "profile_$label"; }
done

for m in flux:black-forest-labs/FLUX.1-dev:$FLUX_REV:"--height 1024 --width 1024" \
         wan22:Wan-AI/Wan2.2-T2V-A14B-Diffusers:$WAN22_REV:"--height 480 --width 832 --num-frames 9" \
         hv:hunyuanvideo-community/HunyuanVideo:$HV_REV:"--height 320 --width 512 --num-frames 61"; do
  IFS=: read -r key id rev shape <<< "$m"
  mkdir -p "$CAMPAIGN/runs/$key"
  for dt in fp32 bf16; do
    done_ "compile_${key}_$dt" || {
      # shellcheck disable=SC2086
      timed "${key}_compile_$dt" "$CAMPAIGN/runs/$key/compile_$dt.log" \
        difflet compile --model-id "$id" --revision "$rev" --tp-degree 4 --cp-degree 1 $shape \
          --tp-reduce-dtype "$dt" && mark "compile_${key}_$dt"; } || { echo "queue2: compile $key $dt FAILED"; continue; }
    done_ "measure_${key}_$dt" || { bash "$CAMPAIGN/jobs/measure.sh" "$key" "$dt" quick && mark "measure_${key}_$dt"; } \
      || echo "queue2: measure $key $dt FAILED"
  done
done
touch "$CAMPAIGN/runs/queue2.done"
echo "queue2: all done"
