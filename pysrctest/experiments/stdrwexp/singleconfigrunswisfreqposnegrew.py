'''
Created on May 25, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.experiments import stdrwexp
from pysrc.algorithms.tdprediction.offpolicy.wislstd import WISLSTD
from pysrc.algorithms.tdprediction.offpolicy.wtd import WTD
import matplotlib.pyplot as ppl

def main():
  ns          = 13
  ftype       = 'binary'
  runseeds    = 4
  N           = 100
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
                   'mright'    : 0.25,
                   'pright'    : 0.75,
                   'inita'     : 1.,
                   'eta'       : 0.001,
                   'initd'       : 100,
                   'lambda'    : 0.6,
                   }
                   for runseed in range(runseeds) 
                ]
  perf1mean         = np.zeros((runseeds, N))
  perf2mean         = np.zeros((runseeds, N))
  for config in configs:
    prob            = StdRWFreqReward(config)
    alg1            = WISLSTD(config)
    alg2            = WTD(config)
    perf1           = PerformanceMeasure(config, prob) 
    perf2           = PerformanceMeasure(config, prob) 
    stdrwexp.runoneconfig(config, prob, alg1, perf1)                
    stdrwexp.runoneconfig(config, prob, alg2, perf2)   
    perf1mean[config['runseed']] = perf1.MSPVE
    perf2mean[config['runseed']] = perf2.MSPVE         
  ppl.plot(np.mean(perf1mean, 0), label='wis')
  ppl.plot(np.mean(perf2mean, 0), label='wtd')
  ppl.ylim([0.01, 1000])
  ppl.yscale('log')
  ppl.legend()
                   
if __name__ == '__main__':
    main()
    ppl.show()
    