'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward

class Test(unittest.TestCase):

  def testgetpol(self):
    pol       = StdRWSparseReward.getpol(5, 0.1)
    poltemp   = np.array([[0,0], [0.9, 0.1], [0.9,0.1], [0.9,0.1], [0,0]])
    assert((pol == poltemp).all())
    
  def teststdrwsparsereward(self):
    ns = 13
    config = {
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : ns,
              'gamma'     : 0.9
              }
    rwprob      = StdRWSparseReward(config)
    probvisit   = np.zeros(ns)
    rwprob.setrunseed(config['runseed'])
    for ep in range(2000):
      rwprob.initepisode()
      while not rwprob.isterminal():
        rets                = rwprob.step()
        probvisit[rets['s']]  += 1
    probvisit /= np.sum(probvisit)
    Psa = StdRWSparseReward.getPsa(ns)
    initstateprob = StdRWSparseReward.getinitstateprob(
      ns, 
      config['inits'])
    mpol = StdRWSparseReward.getpol(ns, 
      config['mright'])
    Pm = StdRWSparseReward.getP(ns, mpol, Psa)
    Dm = StdRWSparseReward.getD(ns, Pm, initstateprob)
    assert((np.abs(probvisit - np.diag(Dm))<0.01).all())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()



