#!/bin/bash
export NEURON_COMPILER_OPT_LEVEL=-O1
export CACHE_OVERRIDE=/home/ubuntu/wan21-bench/profile/cache_O1
export INSPECT=1
/home/ubuntu/wan21-bench/profile/run.sh neuron expA-O1 3 1.0
