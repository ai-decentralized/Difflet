#!/bin/bash
D=/home/ubuntu/qwen-v3-work/diag
F=$D/traj-tp4c-free.pt
S=/home/ubuntu/qwen-v3-work/diag.sh
bash $S $D/b-tp4c-fed compile 4 1 gather_kv $D/traj-tp4c-fed.pt $F > $D/b-tp4c-fed.log 2>&1
bash $S $D/c-tp4e-fed eager 4 1 gather_kv $D/traj-tp4e-fed.pt $F > $D/c-tp4e-fed.log 2>&1
bash $S $D/d-cp2u-fed compile 2 2 ulysses $D/traj-cp2u-fed.pt $F > $D/d-cp2u-fed.log 2>&1
bash $S $D/e-tp4e-free eager 4 1 gather_kv $D/traj-tp4e-free.pt > $D/e-tp4e-free.log 2>&1
bash $S $D/f-cp2u-free compile 2 2 ulysses $D/traj-cp2u-free.pt > $D/f-cp2u-free.log 2>&1
echo BATCH DONE
