'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwfreqreward import StdRWFreqReward2

class Test(unittest.TestCase):

  def teststdrwsparsereward2(self):
    ns      = 13
    gamma   = 0.9
    gm      = np.ones(ns)*gamma
    gm[0]   = gm[ns-1] = 0
    Gamma   = np.diag(gm)
    initdist= np.zeros(ns)
    initdist[(ns-1)/2] = 1.
    config = {
              'ftype'     : 'tabular',
              'ns'        : ns,
              'na'        : 2,
              'behavRight': 0.5,
              'targtRight': 0.9,
              'runseed'   : 1,
              'nf'        : ns,
              'Gamma'     : Gamma,
              'initsdist' : initdist,
              'Rstd'      : 0.0,
              'mdpseed'   : 1000
              }
    
    rwprob      = StdRWFreqReward2(config)
    probvisit   = np.zeros(ns)
    rwprob.initTrajectory(config['runseed'])
    ep = 0
    while ep < 2000:
      rets                = rwprob.step()
      assert(rets['g']==0.0 or 
             (rets['act']==0 and rets['R']==0.0)\
             or (rets['act']==1 and rets['R']==1.0))
      probvisit[rets['s']]  += 1
      if rwprob.isTerminal(): ep += 1
    probvisit /= np.sum(probvisit)
    print(probvisit)
    print(rwprob.dsb)
    assert((np.abs(rwprob.dsb - probvisit)<0.01).all())
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()



