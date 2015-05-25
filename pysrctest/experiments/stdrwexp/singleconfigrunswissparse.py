'''
Created on May 25, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward2
from pysrc.problems.mdp import PerformanceMeasure
from pysrc.experiments import stdrwexp2
from pysrc.algorithms.tdprediction.offpolicy.wislstd import WISLSTD
from pysrc.algorithms.tdprediction.offpolicy.oislstd import OISLSTD
from pysrc.algorithms.tdprediction.offpolicy.olstd2 import OLSTD2
import matplotlib.pyplot as ppl

def main():
  ns          = 13
  ftype       = 'binary'
  runseeds    = 4
  N           = 200
  gamma   = 1.0
  gm      = np.ones(ns)*gamma
  gm[0]   = gm[ns-1] = 0
  Gamma   = np.diag(gm)
  nzG              = np.diag(Gamma)!=0.0
  configs     = [
                   {
                    'offpolicy': True,
                    'mdpseed'  : 1000,
                    'runseed'  : runseed,
                   'ns'        : ns,
                   'na'        : 2,
                   'Gamma'     : Gamma,
                   'T'         : N,
                   'N'         : N,
                   'Rstd'      : 0.0,
                   'ftype'     : ftype,
                   'ns'        : ns,
                   'nf'        : int(np.ceil(np.log(np.sum(nzG)-1)/np.log(2))),
                   'initsdist'  : 'statemiddle',
                   'behavRight'    : 0.5,
                   'targtRight'    : 0.99,
                   'inita'     : 1.,
                   'lmbda'    : 0.9
                   }
                   for runseed in range(runseeds) 
                ]
  perf1mean         = np.zeros((runseeds, N))
  perf2mean         = np.zeros((runseeds, N))
  perf3mean         = np.zeros((runseeds, N))
  prob            = StdRWSparseReward2(configs[0])
  for config in configs:
    alg1            = WISLSTD(config)
    alg2            = OISLSTD(config)
    alg3            = OLSTD2(config)
    perf1           = PerformanceMeasure(config, prob) 
    perf2           = PerformanceMeasure(config, prob) 
    perf3           = PerformanceMeasure(config, prob) 
    stdrwexp2.runoneconfig(config, prob, alg1, perf1)                
    stdrwexp2.runoneconfig(config, prob, alg2, perf2)   
    stdrwexp2.runoneconfig(config, prob, alg3, perf3)   
    perf1mean[config['runseed']] = perf1.MSPVE
    perf2mean[config['runseed']] = perf2.MSPVE         
    perf3mean[config['runseed']] = perf3.MSPVE         
  ppl.plot(np.mean(perf1mean, 0), label='wis')
  ppl.plot(np.mean(perf2mean, 0), label='ois')
  ppl.plot(np.mean(perf3mean, 0), label='olstd2')
  #ppl.ylim([None, 0.2])
  ppl.yscale('log')
  ppl.legend()
                   
if __name__ == '__main__':
    main()
    ppl.show()
    