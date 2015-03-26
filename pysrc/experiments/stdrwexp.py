'''
Created on Mar 24, 2015

@author: A. Rupam Mahmood

This module instantiates an algorithm and the standard random walk 
problem (see Mahmood, van Hasselt & Sutton 2014, nips) and runs an experiment.

'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd()))) 
import numpy as np
import argparse
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
import pysrc.algorithms.gtd as gtd 
import pysrc.algorithms.wistd as wistd
import pysrc.algorithms.wislstd as wislstd
import cPickle as pickle

def runoneconfig(config, prob, alg, perf):
  prob.setrunseed(config['runseed'])
  for ep in range(config['neps']):
    prob.initepisode()
    alg.initepisode()
    while not prob.isterminal():
      probstep          = prob.step()
      s                 = probstep['s']
      a                 = probstep['act']
      probstep['g']     = config['gamma']
      probstep['gnext'] = config['gamma']
      probstep['l']     = config['lambda']
      probstep['lnext'] = config['lambda']
      probstep['rho']   = prob.getRho(s,a)
      alg.step(probstep)
      perf.calcMse(alg.th, ep)
      
def main():
  parser          = argparse.ArgumentParser()
  parser.add_argument("run", help="used as a seed of an independent run", type=int)
  parser.add_argument("path", help="location of the config file")
  args = parser.parse_args()
  configpathname  = args.path + "config.pkl"
  cf              = open(configpathname, 'rb')
  configs         = pickle.load(cf)  
    
  filepathname  = args.path + "run_"\
                  +str(args.run) + ".dat"
  f             = open(filepathname, 'wb')
  algs  = {
           'gtd':gtd.GTD, \
           'wislstd':wislstd.WISLSTD,\
           'wistd':wistd.WISTD, \
           }
  algname   = configs[0]['algname']
  rw1prob   = StdRWSparseReward(configs[0])
  perf      = PerformanceMeasure(configs[0], rw1prob)
  for config in configs:
    config['runseed'] = args.run
    gtdalg            = algs[algname](config)
    runoneconfig(config, rw1prob, gtdalg, perf)
    config['mse']     = perf.mse
    pickle.dump(config, f, -1)

if __name__ == '__main__':
    main()   
    

