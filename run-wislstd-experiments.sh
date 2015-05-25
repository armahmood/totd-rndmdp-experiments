#!/bin/bash

for runseed in {1..5}
do
  python pysrc/experiments/stdrwexp.py $runseed StdRWSparseReward results/wislstd-experiments/stdrw-sparse-reward-states/oislstd/ &

  python pysrc/experiments/stdrwexp.py $runseed StdRWSparseReward results/wislstd-experiments/stdrw-sparse-reward-states/wislstd/ &

done