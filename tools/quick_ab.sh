#!/bin/bash
# Quick A/B on four small scenarios x 10 days.  usage: tools/quick_ab.sh <tag> '<json kwargs>' [policy]
TAG=$1; KW=${2:-'{}'}; POL=${3:-sb_policy:SBPolicy}
cd /home/user/smart-buildings
~/sbvenv/bin/python tools/run_batch.py --policy $POL --tag q_$TAG --steps 2880 --procs 4 --kwargs "$KW" \
  --only building_train_8/fairbanks_8 building_train_8/phoenix_4 building_train_7/denver_5 building_train_7/atlanta_8 | tail -5
