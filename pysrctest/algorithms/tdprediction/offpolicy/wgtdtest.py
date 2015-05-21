'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.tdprediction.offpolicy.wgtd import WGTD
import pysrc.experiments.stdrwexp as stdrwexp
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.problems import mdp
from pysrc.problems.simpletwostate import SimpleTwoState

class Test(unittest.TestCase):

  def testonsparserewardtabular(self):
    ns = 13
    config = {
              'N'      : 200,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : ns-2,
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'eta'       : 0.01,
              'initd'     : 0.01,
              'beta'      : 0.0
              }
    alg         = WGTD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print "tabular"
    print perf.thstarMSE.T[0]
    print alg.th
    assert (abs(perf.thstarMSE.T[0] - alg.th) < 0.05).all()

  def testonsparserewardbinary(self):
    ns = 13
    config = {
              'N'      : 200,
              'ftype'     : 'binary',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : int(np.ceil(np.log(ns-1)/np.log(2))),
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'eta'       : 0.01,
              'initd'     : 0.01,
              'beta'      : 0.0
              }
    alg         = WGTD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print "binary"
    print perf.thstarMSPBE.T[0]
    print alg.th
    assert (abs(perf.thstarMSPBE.T[0] - alg.th) < 0.05).all()

  def testonfreqrewardtabular(self):
    ns = 7
    config = {
              'N'      : 500,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : ns-2,
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'eta'       : 0.00,
              'initd'     : 0.00,
              'beta'      : 0.0
              }
    alg         = WGTD(config)
    rwprob      = StdRWFreqReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.th
    assert (abs(perf.thstarMSE.T[0] - alg.th) < 0.1).all()

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
      'eta'       : 0.005,
      'initd'     : 0.0,
      'beta'      : 0.0
    }
    T         = 5000
    prob      = SimpleTwoState(config)
    prob.Phi  = np.array([[1], [1]])
    alg       = WGTD(config)
    ''' Test fixed points '''
    
    # off-policy fixed point
    thstar3 = mdp.MDP.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, prob.dsb,\
                      prob.Gamma, config['lambda'])
    print(thstar3)
    
    runseed = 0
    prob.initTrajectory(runseed)
    for t in range(T):
      probstep  = prob.step()
      s                 = probstep['s']
      a                 = probstep['act']
      probstep['l']     = config['lambda']
      probstep['lnext'] = config['lambda']
      probstep['rho']   = prob.getRho(s,a)
      alg.step(probstep)
    print(alg.th)
    assert((abs(thstar3-alg.th)<0.06).all())
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()