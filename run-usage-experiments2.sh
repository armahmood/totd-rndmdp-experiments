#!/bin/bash


for runseed in {1..1}
do
for alg in gtd togtd #oislstd wislstd olstd2 wtd wgtd wtogtd
do
python pysrc/experiments/stdrwexp2.py $runseed StdRWSparseReward2 results/usage-experiments/stdrw-sparse-reward-11-states/$alg/ &


done

done