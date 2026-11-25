#!/bin/bash
# Quick A/B on mild-cold scenarios x 10 days.  usage: tools/quick_mild.sh <tag> '<json kwargs>' [policy]
TAG=$1; KW=${2:-'{}'}; POL=${3:-sb_policy:SBPolicy}
cd /home/user/smart-buildings
~/sbvenv/bin/python tools/run_batch.py --policy $POL --tag qm_$TAG --steps 2880 --procs 4 --kwargs "$KW" \
  --only building_train_8/atlanta_1 building_train_7/atlanta_8 building_train_4/denver_2 building_train_5/phoenix_1 | tail -5
