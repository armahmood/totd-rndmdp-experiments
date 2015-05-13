'''
Created on Mar 31, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.mcprediction.dais import DAIS
from pysrc.problems.stdrw import PerformanceMeasure
import pysrc.experiments.stdrwexp as stdrwexp

class Test(unittest.TestCase):
  
  def testStateDependentGamma(self):
    T         = 100
    gammas    = np.random.rand(T)
    gsum      = 0.0
    for t in range(T-1):
      gsum   += (1-gammas[t])*np.prod(gammas[:t])
    gsum     += np.prod(gammas[:T])
    assert(abs(gsum-1.)<10**-10)
    
  def testDAIS(self):
    n       = 3
    neps    = 10
    rewards         = np.array([1., 1., 1.])
    rhos            = np.array([0., 1., 1.])
    gammas          = np.array([1, 1, 0])
    Phi             = np.zeros((n+2, n))
    Phi[1:4,:]      = np.eye(n)
    params          = {}
    params['nf']    = n
    params['ftype'] = 'tabular'
    dais             = DAIS(params)
    for ep in range(neps):
      dais.initepisode()
      for t in range(n):
        params['R']       = rewards[t]
        params['phi']     = Phi[t+1]
        params['phinext'] = Phi[t+2]
        params['gnext']   = gammas[t]
        params['rho']     = rhos[t]
        dais.step(params)
    print(dais.V)
    assert((dais.V==np.array([0, 2, 1])).all())
    assert((dais.nvisits==neps*np.ones(n)).all())

  def testDAISZeroGamma(self):
    n       = 3
    neps    = 10
    rewards         = np.array([0., 0., 1.])
    rhos            = np.array([1., 1., 0.5])
    gammas          = np.array([0, 0, 0])
    Phi             = np.zeros((n+2, n))
    Phi[1:4,:]      = np.eye(n)
    params          = {}
    params['nf']    = n
    params['ftype'] = 'tabular'
    dais             = DAIS(params)
    for ep in range(neps):
      dais.initepisode()
      for t in range(n):
        params['R']       = rewards[t]
        params['phi']     = Phi[t+1]
        params['phinext'] = Phi[t+2]
        params['gnext']   = gammas[t]
        params['rho']     = rhos[t]
        dais.step(params)
    assert((dais.V==np.array([0, 0, 0.5])).all())
    assert((dais.nvisits==neps*np.ones(n)).all())
      

  def testDAISonsparsereward(self):
    ns = 7
    config = {
              'N'      : 900,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : ns-2,
              'gamma'     : 0.9,
              'lambda'    : 1.0,
              }
    alg         = DAIS(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.V
    assert (abs(perf.thstarMSE.T[0] - alg.V) < 0.15).all()

  def testDAISonfreqreward(self):
    ns = 7
    config = {
              'N'      : 500,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.99,
              'runseed'   : 1,
              'nf'        : ns-2,
              'gamma'     : 0.1,
              'lambda'    : 1.0,
              }
    alg         = DAIS(config)
    rwprob      = StdRWFreqReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.V
    assert (abs(perf.thstarMSE.T[0] - alg.V) < 0.05).all()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDAIS']
    unittest.main()