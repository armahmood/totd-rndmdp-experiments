'''
Created on Sep 30, 2014

@author: A. Rupam Mahmood

This module instantiates an algorithm on an off-policy random MDP problem 
 and runs an experiment.

'''

import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
from pysrc.problems import mdp
from pysrc.problems import offrandommdp
from pysrc.algorithms.tdprediction.offpolicy import oislstd
from pysrc.algorithms.tdprediction.offpolicy import olstd2
from pysrc.algorithms.tdprediction.offpolicy import wislstd
from pysrc.algorithms.tdprediction.offpolicy import gtd
from pysrc.algorithms.tdprediction.offpolicy import togtd
from pysrc.algorithms.tdprediction.offpolicy import wtd
from pysrc.algorithms.tdprediction.offpolicy import wgtd
from pysrc.algorithms.tdprediction.offpolicy import wtogtd
import copy
import pickle

def runoneconfig(config, prob, alg, perf):
  prob.initTrajectory(config['runseed'])
  for t in range(config['T']):
    probstep          = prob.step()
    s                 = probstep['s']
    a                 = probstep['act']
    probstep['l']     = config['lmbda']
    probstep['lnext'] = config['lmbda']
    probstep['rho']   = prob.getRho(s, a)
    alg.step(probstep)
    perf.calcMSPVE(alg, t)

def main():
  parser          = argparse.ArgumentParser()
  parser.add_argument("mdpseed", help="used as a seed to generate a random MDP", type=int)
  parser.add_argument("runseed", help="used as a seed of an independent run", type=int)
  parser.add_argument("path", help="location of the config file")
  parser.add_argument("algname", help="name of the algorithm")
  args = parser.parse_args()
  configprobpathname  = args.path + "configprob.pkl"
  cf              = open(configprobpathname, 'rb')
  configprob      = pickle.load(cf)  

  configalgpathname  = args.path + args.algname + "/configalg.pkl"
  cf              = open(configalgpathname, 'rb')
  configsalg      = pickle.load(cf)
    
  filepathname  = args.path + args.algname   +\
                  "/mdpseed_"  + str(args.mdpseed)   + "_"\
                  "runseed_"  + str(args.runseed)   +\
                  ".dat"
  f             = open(filepathname, 'wb')
  
  algs  = {
           'gtd':gtd.GTD, \
           'togtd':togtd.TOGTD,\
           'oislstd':oislstd.OISLSTD,\
           'olstd2':olstd2.OLSTD2,\
           'wislstd':wislstd.WISLSTD,\
           'wtd':wtd.WTD, \
           'wgtd':wgtd.WGTD, \
           'wtogtd':wtogtd.WTOGTD,         
           }
  configprob['mdpseed'] = args.mdpseed
  prob                  = offrandommdp.OffRandomMDP(configprob)

  print("Running algorithm " + args.algname + ", runseed: " + str(args.runseed) )
  for config in configsalg:
    perf      = mdp.PerformanceMeasure(configprob, prob)
    config.update(configprob)
    alg                   = algs[args.algname](config)
    config['runseed']     = args.runseed
    runoneconfig(config, prob, alg, perf)
    config['error']      = perf.getNormMSPVE()
    pickle.dump(config, f, -1)

if __name__ == '__main__':
    main()
    
