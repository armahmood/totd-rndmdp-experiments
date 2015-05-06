'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import pylab as pl
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.tdprediction.psdtd import PSDTD
import pysrc.experiments.stdrwexp as stdrwexp
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.problems import mdp
from pysrc.problems.simpletwostate import SimpleTwoState

class Test(unittest.TestCase):

  def testEtdOnSimpleTwoStateFuncApprox2(self):
    lmbda   = 0.
    config  = \
    {
      'nf'        : 1,
      'ftype'     : None,
      'Rstd'      : 0.0,
      'initsdist' : 'steadystate',
      'Gamma'     : 0.9*np.eye(2),
      'mdpseed'   : 1000,
      'lambda'    : lmbda,
      'alpha'     : 0.05,
    }
    T         = 1000
    nruns     = 30
    prob      = SimpleTwoState(config)
    prob.Phi  = np.array([[1], [1]])
    alg       = PSDTD(config)
    ''' Test fixed points '''
    
    # target on-policy fixed point
    thstar0 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, mdp.steadystateprob(prob.Psst),\
                      prob.Gamma, 1.)
    print thstar0

    # off-policy fixed point
    thstar1 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, prob.dsb,\
                      prob.Gamma, config['lambda'])
    print(thstar1)
    
    # emphatic fixed point
    ImPG    = np.eye(prob.ns) - np.dot(prob.Psst, prob.Gamma)
    ImPLG   = np.eye(prob.ns) - np.dot(lmbda*prob.Psst, prob.Gamma)
    ImPL    = np.dot( pl.inv(ImPLG), ImPG )
    m       = np.dot(prob.dsb.T, pl.inv(ImPL))
    thstar2 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, m,\
                      prob.Gamma, lmbda)
    print(thstar2)
    
    ths   = np.zeros((nruns, config['nf']))
    for runseed in range(nruns): 
      prob.initTrajectory(runseed)
      for t in range(T):
        probstep  = prob.step()
        s                 = probstep['s']
        a                 = probstep['act']
        probstep['l']     = config['lambda']
        probstep['lnext'] = config['lambda']
        probstep['rho']   = prob.getRho(s,a)
        probstep['I']     = 1 if t==0 else 0
        alg.step(probstep)
      ths[runseed]  = alg.th
      meanth        = np.mean(ths, 0)
    print(meanth)
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()