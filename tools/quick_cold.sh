#!/bin/bash
# Quick A/B on two cold scenarios x 10 days. usage: tools/quick_cold.sh <tag> '<json kwargs>'
TAG=$1; KW=${2:-'{}'}
cd /home/user/smart-buildings
~/sbvenv/bin/python tools/run_batch.py --policy sb_policy:SBPolicy --tag qc_$TAG --steps 2880 --procs 4 --kwargs "$KW" \
  --only building_train_8/fairbanks_8 building_train_7/fairbanks_7 building_train_8/atlanta_1 building_train_7/atlanta_8 | tail -5
