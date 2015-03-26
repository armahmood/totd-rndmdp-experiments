'''
Created on Mar 24, 2015

@author: A. Rupam Mahmood

This module instantiates an algorithm and the standard random walk 
problem (see Mahmood, van Hasselt & Sutton 2014, nips) and runs an experiment.

'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd()))) 
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
  # parse command-line arguments
  parser          = argparse.ArgumentParser()
  parser.add_argument("run", help="used as a seed of an independent run", type=int)
  parser.add_argument("path", help="location of the config file")
  args = parser.parse_args()

  # load the config file
  configpathname  = args.path + "config.pkl"
  cf              = open(configpathname, 'rb')
  configs         = pickle.load(cf)  
    
  # prepare the file where data output is stored
  filepathname  = args.path + "run_"\
                  +str(args.run) + ".dat"
  f             = open(filepathname, 'wb')
  
  # list of algorithms allowed with this experiment
  algs  = {
           'gtd':gtd.GTD, \
           'wislstd':wislstd.WISLSTD,\
           'wistd':wistd.WISTD, \
           }
  # choose the algorithm mentioned by the command line argument
  algname   = configs[0]['algname']
  # choose the problem 
  rw1prob   = StdRWSparseReward(configs[0])
  # initialize ther performance measurer
  perf      = PerformanceMeasure(configs[0], rw1prob)
  # one-by-one each configuration is run 
  for config in configs:
    # for each configuration, a new algorithm is initialized
    alg            = algs[algname](config)
    config['runseed'] = args.run
    runoneconfig(config, rw1prob, alg, perf)
    config['mse']     = perf.mse
    # performance is dumper together with the configuration
    # in the same fie 
    pickle.dump(config, f, -1)

if __name__ == '__main__':
    main()   
    

