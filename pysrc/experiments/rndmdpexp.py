'''
Created on Sep 30, 2014

@author: A. Rupam Mahmood

This module instantiates an algorithm on a random MDP problem 
(see van Seijen, Sutton, Mahmood, Pilarski 2015, ewrl) and runs an experiment.

'''

import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
from pysrc.problems import mdp
from pysrc.problems import randommdp
from pysrc.algorithms.tdprediction.onpolicy import td, tdr, totd
import copy
import pickle

def runoneconfig(config, prob, alg, perf):
  prob.initTrajectory(config['runseed'])
  for t in range(config['T']):
    probstep          = prob.step()
    probstep['l']     = config['lmbda']
    probstep['lnext'] = config['lmbda']
    alg.step(probstep)
    perf.calcMSPVE(alg, t)

def main():
  parser          = argparse.ArgumentParser()
  parser.add_argument("mdpseed", help="used as a seed to generate a random MDP", type=int)
  parser.add_argument("ftype", help="Type of feature representations: tabular/binary/normal")
  parser.add_argument("runseed", help="used as a seed of an independent run", type=int)
  parser.add_argument("path", help="location of the config file")
  args = parser.parse_args()
  configpathname  = args.path + "config.pkl"
  cf              = open(configpathname, 'rb')
  configs         = pickle.load(cf)  
    
  filepathname  = args.path   +\
                  "mdpseed_"  + str(args.mdpseed)   + "_"\
                  "ftype_"    + str(args.ftype) + "_"\
                  "runseed_"  + str(args.runseed)   +\
                  ".dat"
  f             = open(filepathname, 'wb')
  
  algs  = {
           'td':td.TD,
           'tdr':tdr.TDR,
           'totd':totd.TOTD,
           }
  algname               = configs[0]['algname']
  probconfig            = copy.copy(configs[0])
  probconfig['mdpseed'] = args.mdpseed
  probconfig['ftype']   = args.ftype
  prob                  = randommdp.RandomMDP(probconfig)

  perf      = mdp.PerformanceMeasure(probconfig, prob)
  print("Running algorithm " + algname + ", runseed: " + str(args.runseed) )
  for config in configs:
    config['ftype']       = args.ftype
    config['nf']          = prob.nf
    alg                   = algs[algname](config)
    config['runseed']     = args.runseed
    runoneconfig(config, prob, alg, perf)
    config['NMSPVE']      = perf.getNormMSPVE()
    pickle.dump(config, f, -1)

if __name__ == '__main__':
    main()
    
