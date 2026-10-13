#!/bin/bash
export DIFFLET_PROBE2_BLOCK_GROUP=2
export CACHE_OVERRIDE=/home/ubuntu/wan21-bench/profile/cache_pair
export INSPECT=1
/home/ubuntu/wan21-bench/profile/run.sh neuron expD-pair 3 1.0
