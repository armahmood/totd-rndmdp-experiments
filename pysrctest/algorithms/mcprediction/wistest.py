'''
Created on Mar 31, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.mcprediction.wis import WIS
from pysrc.problems.stdrw import PerformanceMeasure
import pysrc.experiments.stdrwexp as stdrwexp

class Test(unittest.TestCase):

  def testWISonpolicy(self):
    n       = 3
    neps    = 10
    rewards         = np.array([1., 1., 1.])
    rhos            = np.array([1., 1., 1.])
    gammas          = np.array([1, 1, 0])
    Phi             = np.zeros((n+2, n))
    Phi[1:4,:]      = np.eye(n)
    params          = {}
    params['nf']    = n
    params['ftype'] = 'tabular'
    wis             = WIS(params)
    for ep in range(neps):
      wis.initepisode()
      for t in range(n):
        params['R']       = rewards[t]
        params['phi']     = Phi[t+1]
        params['phinext'] = Phi[t+2]
        params['gnext']   = gammas[t]
        params['rho']     = rhos[t]
        wis.step(params)
    assert((wis.V==np.array([3, 2, 1])).all())
    assert((wis.U==neps*np.ones(n)).all())
      
  def testWIS(self):
    n       = 3
    neps    = 1
    rewards         = np.array([1., 1., 1.])
    rhos            = np.array([0., 1., 1.])
    gammas          = np.array([1, 1, 0])
    Phi             = np.zeros((n+2, n))
    Phi[1:4,:]      = np.eye(n)
    params          = {}
    params['nf']    = n
    params['ftype'] = 'tabular'
    wis             = WIS(params)
    for ep in range(neps):
      wis.initepisode()
      for t in range(n):
        params['R']       = rewards[t]
        params['phi']     = Phi[t+1]
        params['phinext'] = Phi[t+2]
        params['gnext']   = gammas[t]
        params['rho']     = rhos[t]
        wis.step(params)
    assert((wis.V==np.array([0, 2, 1])).all())
    assert((wis.U==rhos).all())

  def testWISZeroGamma(self):
    n       = 3
    neps    = 1
    rewards         = np.array([0., 0., 1.])
    rhos            = np.array([0., 0.5, 1.])
    gammas          = np.array([0, 0, 0])
    Phi             = np.zeros((n+2, n))
    Phi[1:4,:]      = np.eye(n)
    params          = {}
    params['nf']    = n
    params['ftype'] = 'tabular'
    wis             = WIS(params)
    for ep in range(neps):
      wis.initepisode()
      for t in range(n):
        params['R']       = rewards[t]
        params['phi']     = Phi[t+1]
        params['phinext'] = Phi[t+2]
        params['gnext']   = gammas[t]
        params['rho']     = rhos[t]
        wis.step(params)
    assert((wis.V==np.array([0, 0, 1])).all())
    assert((wis.U==rhos).all())

  def testWOISonsparsereward(self):
    ns = 7
    config = {
              'neps'      : 500,
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
    alg         = WIS(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.V
    assert (abs(perf.thstarMSE.T[0] - alg.V) < 0.05).all()

  def testWISonfreqreward(self):
    ns = 7
    config = {
              'neps'      : 500,
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
    alg         = WIS(config)
    rwprob      = StdRWFreqReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.V
    assert (abs(perf.thstarMSE.T[0] - alg.V) < 0.1).all()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testOIS']
    unittest.main()