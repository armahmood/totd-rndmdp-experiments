#!/bin/bash

for runseed in {1..10}
do
  python pysrc/experiments/stdrwexp.py $runseed StdRWSparseReward results/wislstdexperiments/oislstd/ &

  python pysrc/experiments/stdrwexp.py $runseed StdRWSparseReward results/wislstdexperiments/wislstd/ &

done