'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwfreqposnegreward import StdRWFreqPosNegReward

class Test(unittest.TestCase):

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
    rwprob      = StdRWFreqPosNegReward(config)
    probvisit   = np.zeros(ns)
    rwprob.setrunseed(config['runseed'])
    for ep in range(500):
      rwprob.initepisode()
      while not rwprob.isterminal():
        rets                = rwprob.step()
        assert(
               (rets['act']==0 and rets['R']==-1.0)\
               or (rets['act']==1 and rets['R']==1.0))
        probvisit[rets['s']]  += 1
    probvisit /= np.sum(probvisit)
    Psa = StdRWFreqPosNegReward.getPsa(ns)
    initstateprob = StdRWFreqPosNegReward.getinitstateprob(
      ns, 
      config['inits'])
    mpol = StdRWFreqPosNegReward.getpol(ns, 
      config['mright'])
    Pm = StdRWFreqPosNegReward.getP(ns, mpol, Psa)
    Dm = StdRWFreqPosNegReward.getD(ns, Pm, initstateprob)
    print(probvisit)
    assert((np.abs(probvisit - np.diag(Dm))<0.01).all())

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()



