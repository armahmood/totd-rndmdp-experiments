'''
Created on Mar 31, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.algorithms.mcprediction.ois import OIS
from pysrc.problems.stdrw import PerformanceMeasure
import pysrc.experiments.stdrwexp as stdrwexp

class Test(unittest.TestCase):

  def testOIS(self):
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
    ois             = OIS(params)
    for ep in range(neps):
      ois.initepisode()
      for t in range(n):
        params['R']       = rewards[t]
        params['phi']     = Phi[t+1]
        params['phinext'] = Phi[t+2]
        params['gnext']   = gammas[t]
        params['rho']     = rhos[t]
        ois.step(params)
    assert((ois.V==np.array([0, 2, 1])).all())
    assert((ois.nvisits==neps*np.ones(n)).all())
      

  def testOISonsparsereward(self):
    ns = 7
    config = {
              'neps'      : 3000,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : ns-2,
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'inita'     : 0.01,
              }
    alg         = OIS(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstar.T[0]
    print alg.V
    assert (abs(perf.thstar.T[0] - alg.V) < 0.05).all()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testOIS']
    unittest.main()