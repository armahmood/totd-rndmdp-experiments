#!/bin/bash


for runseed in {1..5}
do
#for alg in gtd togtd oislstd wislstd olstd2 wtd wgtd wtogtd
for alg in togtd
do
python pysrc/experiments/stdrwexp.py $runseed StdRWSparseReward results/usage-experiments/stdrw-sparse-reward-11-states-test/$alg/ &


done

done