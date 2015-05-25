'''
Created on May 25, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.experiments import stdrwexp
from pysrc.algorithms.tdprediction.offpolicy.wislstd import WISLSTD
from pysrc.algorithms.tdprediction.offpolicy.oislstd import OISLSTD
from pysrc.algorithms.tdprediction.offpolicy.olstd2 import OLSTD2
import matplotlib.pyplot as ppl

def main():
  ns          = 23
  ftype       = 'binary'
  runseeds    = 4
  N           = 200
  configs     = [
                   {
                    'runseed'  : runseed,
                   'ns'        : ns,
                   'gamma'     : 1.0,
                   'N'         : N,
                   'ftype'     : ftype,
                   'ns'        : ns,
                   'nf'        : int(np.ceil(np.log(ns-1)/np.log(2))),
                   'inits'     : (ns-1)/2,
                   'mright'    : 0.5,
                   'pright'    : 0.99,
                   'inita'     : 1.,
                   'lambda'    : 0.9
                   }
                   for runseed in range(runseeds) 
                ]
  perf1mean         = np.zeros((runseeds, N))
  perf2mean         = np.zeros((runseeds, N))
  perf3mean         = np.zeros((runseeds, N))
  for config in configs:
    prob            = StdRWFreqReward(config)
    alg1            = WISLSTD(config)
    alg2            = OISLSTD(config)
    alg3            = OLSTD2(config)
    perf1           = PerformanceMeasure(config, prob) 
    perf2           = PerformanceMeasure(config, prob) 
    perf3           = PerformanceMeasure(config, prob) 
    stdrwexp.runoneconfig(config, prob, alg1, perf1)                
    stdrwexp.runoneconfig(config, prob, alg2, perf2)   
    stdrwexp.runoneconfig(config, prob, alg3, perf3)   
    perf1mean[config['runseed']] = perf1.MSPVE
    perf2mean[config['runseed']] = perf2.MSPVE         
    perf3mean[config['runseed']] = perf3.MSPVE         
  ppl.plot(np.mean(perf1mean, 0), label='wis')
  ppl.plot(np.mean(perf2mean, 0), label='ois')
  ppl.plot(np.mean(perf3mean, 0), label='olstd2')
  #ppl.ylim([None, 10])
  ppl.yscale('log')
  ppl.legend()
                   
if __name__ == '__main__':
    main()
    ppl.show()
    