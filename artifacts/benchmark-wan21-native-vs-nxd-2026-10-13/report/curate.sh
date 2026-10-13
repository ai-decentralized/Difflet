#!/bin/bash
# Curate the campaign's evidence into the repo: logs (big ones gzipped), probe JSONs, summaries,
# quality/oracle JSON, selected sheets, mp4s, profiler summaries. No caches, NEFFs, traces, or
# latent tensors.
set -eu
S=/home/ubuntu/wan21-bench
D=/home/ubuntu/Difflet/.claude/worktrees/wan21-bench/artifacts/benchmark-wan21-native-vs-nxd-2026-10-13
mkdir -p $D
cp $S/runs.jsonl $S/summary.md $S/job.sh $D/
for b in neuron trainium; do
  for r in $S/$b/*/; do
    n=$(basename $r); [ "$n" = cache ] && continue
    mkdir -p $D/$b/$n
    [ -f $r/out.mp4 ] && cp $r/out.mp4 $D/$b/$n/
    [ -d $r/probe ] && mkdir -p $D/$b/$n/probe && cp $r/probe/*.json $D/$b/$n/probe/
    if [ $(stat -c %s $r/log.txt) -gt 1000000 ]; then gzip -9c $r/log.txt > $D/$b/$n/log.txt.gz; else cp $r/log.txt $D/$b/$n/; fi
  done
done
# quality
Q=$D/quality; mkdir -p $Q/sheets $Q/blind $Q/ref $Q/oracle $Q/device-2step
cp $S/quality/metrics.json $S/quality/run.log $S/quality/blind_key.json $Q/
cp $S/quality/blind/*.png $Q/blind/
cp $S/quality/ref/*.mp4 $Q/ref/
for f in neuron_warm1-g5 trainium_warm1-g5 neuron_warm1-g1 trainium_warm1-g1 \
         cpuvae_neuron_warm1-g5 cpuvae_trainium_warm1-g5 cpuvae_neuron_warm1-g1 cpuvae_trainium_warm1-g1; do
  cp $S/report/img/$f.jpg $Q/sheets/ 2>/dev/null || \
    /home/ubuntu/workspace/native_venv/bin/python -c "from PIL import Image; Image.open('$S/quality/sheets/$f.png').convert('RGB').save('$Q/sheets/$f.jpg', quality=82, optimize=True)"
done
cp $S/quality/oracle/compare.json $S/quality/oracle/ref.log $S/quality/oracle/run_ref.sh $Q/oracle/ 2>/dev/null || true
for b in neuron trainium; do
  mkdir -p $Q/device-2step/$b/probe
  cp $S/quality/device-2step/$b/log.txt $S/quality/device-2step/$b/run.sh $Q/device-2step/$b/ 2>/dev/null || true
  cp $S/quality/device-2step/$b/probe/*.json $Q/device-2step/$b/probe/
done
# profile (summaries and scripts only)
P=$D/profile; mkdir -p $P/analysis $P/prof $P/probe2
cp $S/profile/results.json $S/profile/*.py $S/profile/*.sh $P/
cp $S/profile/probe2/*.py $P/probe2/ 2>/dev/null || true
find $S/profile/analysis -maxdepth 1 -name "*.txt" -exec cp {} $P/analysis/ \;
# The *_sys.json runtime system traces (34-123 MB each) stay on the host.
find $S/profile/prof -maxdepth 1 \( -name "*.txt" -o -name "*summary*.json" \) -exec cp {} $P/prof/ \;
# verifier re-measurements (numbers + scripts)
V=$D/verify; mkdir -p $V
cp $S/verify_meas/*.sh $S/verify_meas/*.txt $V/ 2>/dev/null || true
for r in base1 base2 ccop1 ccop2 omp3 omp_default; do
  [ -d $S/verify_meas/$r ] || continue
  mkdir -p $V/$r && find $S/verify_meas/$r -name "probe-*.json" -exec cp {} $V/$r/ \;
done
# VCS-unset NxD check
mkdir -p $D/vae-vcs-test/probe
cp $S/vae-vcs-test/run_novcs.sh $S/vae-vcs-test/novcs/log.txt $D/vae-vcs-test/
cp $S/vae-vcs-test/novcs/probe/*.json $D/vae-vcs-test/probe/
sha256sum $S/vae-vcs-test/novcs/out.mp4 $S/trainium/warm1-g5/out.mp4 > $D/vae-vcs-test/mp4_sha256.txt
# ccop fix verification
if [ -d $S/ccop-verify ]; then
  C=$D/ccop-verify; mkdir -p $C
  cp $S/ccop-verify/run.sh $S/ccop-verify/walls.txt $C/ 2>/dev/null || true
  for r in $S/ccop-verify/*-g*/; do
    n=$(basename $r); case $n in aborted*) continue;; esac
    mkdir -p $C/$n/probe; cp $r/log.txt $C/$n/; cp $r/probe/*.json $C/$n/probe/ 2>/dev/null || true
    [ -f $r/out.mp4 ] && [ $n = warm1-g5 ] && cp $r/out.mp4 $C/$n/
  done
  [ -d $S/ccop-verify/oracle ] && mkdir -p $C/oracle && cp $S/ccop-verify/oracle/*.json $C/oracle/
fi
du -sh $D
