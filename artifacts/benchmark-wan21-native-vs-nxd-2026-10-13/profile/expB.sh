#!/bin/bash
export NEURON_CC_FLAGS="--tensorizer-options=--enable-ccop-compute-overlap"
export CACHE_OVERRIDE=/home/ubuntu/wan21-bench/profile/cache_ccop
export INSPECT=1
export EXTRA_ARGS="--host-text-encoder --host-vae"
/home/ubuntu/wan21-bench/profile/run.sh neuron expB-ccop 3 1.0
