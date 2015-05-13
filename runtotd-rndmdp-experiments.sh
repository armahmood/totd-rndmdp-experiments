#!/bin/bash

problist='{small, large, deterministic}'
featurelist='{tabular, binary, normal}'
algorithmlist='{td, tdr, totd}'

if [ $# != 3 ] 	|| [ `eval echo $problist 		| grep -c $1`  == 0 ] \
				|| [ `eval echo $featurelist 	| grep -c $2`  == 0 ] \
				|| [ `eval echo $algorithmlist	| grep -c $3`  == 0 ] 
then
	echo Three arguments required.
	echo Choose argument 1 from the following list: $problist
	echo Choose argument 2 from the following list: $featurelist
	echo Choose argument 3 from the following list: $algorithmlist
exit 0
else
	echo Running $1 random mdp experiment with $2 features using $3 algorithm
	for runseed in {1..10}
	do
		python pysrc/experiments/rndmdpexp.py 1000 $2 $runseed results/totd-rndmdp-experiments/$1/$3/ &
	done
fi
