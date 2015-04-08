#!/bin/bash

python pysrc/experiments/stdrwexp.py 1 StdRWSparseReward results/wislstdexperiments/oislstd/ &

python pysrc/experiments/stdrwexp.py 1 StdRWSparseReward results/wislstdexperiments/wislstd/ &

