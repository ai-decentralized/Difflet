#!/usr/bin/env bash
# job_verify_queue.sh <slug> <wait-for-name> : wait for logs/<wait-for-name>.done, then run the verifier for <slug>.
SLUG=${1:?}; AFTER=${2:?}
until [ -f ${PTQ_JOBS:-/home/ubuntu/ptq-jobs}/logs/$AFTER.done ]; do sleep 30; done
exec bash ${PTQ_JOBS:-/home/ubuntu/ptq-jobs}/run_verify.sh "$SLUG"
