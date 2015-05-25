#!/bin/bash

for runseed in {1..5}
do
  python pysrc/experiments/stdrwexp.py $runseed StdRWFreqReward results/wislstd-experiments/stdrw-freq-reward-states/oislstd/ &

  python pysrc/experiments/stdrwexp.py $runseed StdRWFreqReward results/wislstd-experiments/stdrw-freq-reward-states/wislstd/ &

done