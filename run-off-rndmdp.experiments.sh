#!/bin/bash


for runseed in {1..1}
do
for alg in gtd togtd oislstd wislstd olstd2 wtd wgtd wtogtd
do
python pysrc/experiments/offrndmdpexp.py 1000 $runseed results/offpolicy-rndmdp-experiments/state-10-bpol-skewed-tpol-skewed-ftype-binary/ $alg &


done

done