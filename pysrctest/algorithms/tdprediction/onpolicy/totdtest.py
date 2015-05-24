'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np 
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.tdprediction.onpolicy.totd import TOTD
import pysrc.experiments.stdrwexp as stdrwexp
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.problems import mdp
from pysrc.problems.simpletwostate import SimpleTwoState

class Test(unittest.TestCase):

  def testtdronsparserewardtabular(self):
    ns = 13
    config = {
              'N'      : 200,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.5,
              'runseed'   : 1,
              'nf'        : ns-2,
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'alpha'     : 0.02,
              'beta'      : 0.0
              }
    alg         = TOTD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.th
    assert (abs(perf.thstarMSE.T[0] - alg.th) < 0.05).all()

  def testtdronsparserewardbinary(self):
    ns = 13
    config = {
              'N'      : 200,
              'ftype'     : 'binary',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.5,
              'runseed'   : 1,
              'nf'        : int(np.ceil(np.log(ns-1)/np.log(2))),
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'alpha'     : 0.005,
              'beta'      : 0.0
              }
    alg         = TOTD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSPBE.T[0]
    print alg.th
    assert (abs(perf.thstarMSPBE.T[0] - alg.th) < 0.05).all()

  def testtdronfreqrewardtabular(self):
    ns = 5
    config = {
              'N'      : 200,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.5,
              'runseed'   : 1,
              'nf'        : ns-2,
              'gamma'     : 0.9,
              'lambda'    : 0.1,
              'alpha'     : 0.02,
              }
    alg         = TOTD(config)
    rwprob      = StdRWFreqReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.th
    assert (abs(perf.thstarMSE.T[0] - alg.th) < 0.15).all()

  def testOnSimpleTwoStateFuncApprox(self):
    config = \
    {
      'nf'        : 1,
      'ftype'     : None,
      'Rstd'      : 0.0,
      'initsdist' : 'steadystate',
      'Gamma'     : 0.9*np.eye(2),
      'mdpseed'   : 1000,
      'lambda'    : 0.0,
      'alpha'     : 0.005,
    }
    T         = 10000
    rwprob1      = SimpleTwoState(config)
    rwprob1.Phi  = np.array([[1], [1]])
    alg       = TOTD(config)
    ''' Test fixed points '''
    
    # off-policy fixed point
    thstar3 = mdp.MDP.getFixedPoint(rwprob1.Psst, rwprob1.exprt,\
                      rwprob1.Phi, rwprob1.dsb,\
                      rwprob1.Gamma, config['lambda'])
    print(thstar3)
    
    runseed = 0
    rwprob1.initTrajectory(runseed)
    for t in range(T):
      probstep  = rwprob1.step()
      s                 = probstep['s']
      a                 = probstep['act']
      probstep['l']     = config['lambda']
      probstep['lnext'] = config['lambda']
      probstep['rho']   = rwprob1.getRho(s,a)
      alg.step(probstep)
    print(alg.th)
    assert((abs(thstar3-alg.th)<0.06).all())
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()