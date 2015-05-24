'''
Created on Mar 24, 2015

@author: A. Rupam Mahmood

This module instantiates an algorithm and the standard random walk 
problem (see Mahmood, van Hasselt & Sutton 2014, nips) and runs an experiment.

'''

import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.tdprediction.offpolicy import oislstd
from pysrc.algorithms.tdprediction.offpolicy import olstd2
from pysrc.algorithms.tdprediction.offpolicy import wislstd
from pysrc.algorithms.tdprediction.offpolicy import gtd
from pysrc.algorithms.tdprediction.offpolicy import togtd
from pysrc.algorithms.tdprediction.offpolicy import wtd
from pysrc.algorithms.tdprediction.offpolicy import wgtd
from pysrc.algorithms.tdprediction.offpolicy import wtogtd
import cPickle as pickle

def runoneconfig(config, rwprob1, alg, perf):
  rwprob1.setrunseed(config['runseed'])
  for ep in range(config['N']):
    rwprob1.initepisode()
    alg.initepisode()
    while not rwprob1.isterminal():
      probstep          = rwprob1.step()
      s                 = probstep['s']
      a                 = probstep['act']
      probstep['l']     = config['lambda']
      probstep['lnext'] = config['lambda']
      probstep['rho']   = rwprob1.getRho(s,a)
      alg.step(probstep)
    perf.calcMse(alg, ep)
      
def main():
  parser          = argparse.ArgumentParser()
  parser.add_argument("run", help="used as a seed of an independent run", type=int)
  parser.add_argument("probname", help="name of the problem to run experiment on")
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
           'togtd':togtd.TOGTD,\
           'oislstd':oislstd.OISLSTD,\
           'olstd2':olstd2.OLSTD2,\
           'wislstd':wislstd.WISLSTD,\
           'wtd':wtd.WTD, \
           'wgtd':wgtd.WGTD, \
           'wtogtd':wtogtd.WTOGTD,         
           }
  probs = {
           'StdRWSparseReward'  : StdRWSparseReward,
           'StdRWFreqReward'    : StdRWFreqReward,
           }
  algname   = configs[0]['algname']
  rwprob   = probs[args.probname](configs[0])
  perf      = PerformanceMeasure(configs[0], rwprob)
  print("Running algorithm " + algname + " on problem " + args.probname + ", runseed: " + str(args.run) )
  for config in configs:
    alg            = algs[algname](config)
    config['runseed'] = args.run
    runoneconfig(config, rwprob, alg, perf)
    config['error']     = perf.MSPVE
    pickle.dump(config, f, -1)

if __name__ == '__main__':
    main()   
    

