#!/bin/bash
export DIFFLET_PROBE2_WLO=1
export CACHE_OVERRIDE=/home/ubuntu/wan21-bench/profile/cache_wlo
export INSPECT=1
/home/ubuntu/wan21-bench/profile/run.sh neuron expC-wlo 3 1.0
