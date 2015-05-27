'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward2
from pysrc.problems.stdrwfreqreward import StdRWFreqReward2
from pysrc.algorithms.tdprediction.offpolicy.wgtd import WGTD
import pysrc.experiments.stdrwexp2 as stdrwexp2
from pysrc.problems.mdp import PerformanceMeasure
from pysrc.problems import mdp
from pysrc.problems.simpletwostate import SimpleTwoState

class Test(unittest.TestCase):

  def testonsparserewardtabular(self):
    ns = 13
    gamma = 0.9
    gm = np.ones(ns) * gamma
    gm[0] = gm[ns - 1] = 0
    Gamma = np.diag(gm)
    nzG              = np.diag(Gamma)!=0.0
    config = {
              'offpolicy' : True,
              'mdpseed'   : 1000,
              'Gamma'     : Gamma,
              'initsdist' : 'statemiddle',
              'Rstd'      : 0.0,
              'T'         : 200,
              'N'      : 200,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'na'        : 2,
              'behavRight': 0.5,
              'targtRight': 0.9,
              'runseed'   : 1,
              'nf'        : np.sum(nzG),
              'lmbda'    : 0.5,
              'eta'       : 0.05,
              'initd'     : 0.0,
              'beta'      : 0.0
              }
    alg         = WGTD(config)
    rwprob      = StdRWSparseReward2(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp2.runoneconfig(config, rwprob, alg, perf)
    print "tabular"
    print perf.thstar.T
    print alg.th
    assert (abs(perf.thstar.T - alg.th) < 0.05).all()

  def testonsparserewardbinary(self):
    ns = 13
    gamma = 0.9
    gm = np.ones(ns) * gamma
    gm[0] = gm[ns - 1] = 0
    Gamma = np.diag(gm)
    nzG              = np.diag(Gamma)!=0.0
    config = {
              'offpolicy' : True,
              'mdpseed'   : 1000,
              'Gamma'     : Gamma,
              'initsdist' : 'statemiddle',
              'Rstd'      : 0.0,
              'T'         : 200,
              'N'      : 200,
              'ftype'     : 'binary',
              'ns'        : ns,
              'na'        : 2,
              'behavRight': 0.5,
              'targtRight': 0.9,
              'runseed'   : 1,
              'nf'        : int(np.ceil(np.log(np.sum(nzG)+1)/np.log(2))),
              'lmbda'    : 0.5,
              'eta'       : 0.01,
              'initd'     : 0.01,
              'beta'      : 0.0
              }
    alg         = WGTD(config)
    rwprob      = StdRWSparseReward2(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp2.runoneconfig(config, rwprob, alg, perf)
    thstarMSPBE = perf.getThstarMSPBE(config['lmbda'])
    print "binary"
    print thstarMSPBE
    print alg.th
    assert (abs(thstarMSPBE - alg.th) < 0.05).all()

  def testonfreqrewardtabular(self):
    ns = 7
    gamma = 0.9
    gm = np.ones(ns) * gamma
    gm[0] = gm[ns - 1] = 0
    Gamma = np.diag(gm)
    nzG              = np.diag(Gamma)!=0.0
    config = {
              'offpolicy' : True,
              'mdpseed'   : 1000,
              'Gamma'     : Gamma,
              'initsdist' : 'statemiddle',
              'Rstd'      : 0.0,
              'T'         : 500,
              'N'      : 500,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'na'        : 2,
              'behavRight': 0.5,
              'targtRight': 0.9,
              'runseed'   : 1,
              'nf'        : np.sum(nzG),
              'lmbda'    : 0.5,
              'eta'       : 0.005,
              'initd'     : 0.00,
              'beta'      : 0.0
              }
    alg         = WGTD(config)
    rwprob      = StdRWFreqReward2(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp2.runoneconfig(config, rwprob, alg, perf)
    print perf.thstar.T
    print alg.th
    assert (abs(perf.thstar.T - alg.th) < 0.05).all()

  def testOnSimpleTwoStateFuncApprox(self):
    config = \
    {
      'offpolicy' : True,
      'nf'        : 1,
      'ftype'     : None,
      'Rstd'      : 0.0,
      'initsdist' : 'steadystate',
      'Gamma'     : 0.9*np.eye(2),
      'mdpseed'   : 1000,
      'lmbda'    : 0.0,
      'eta'       : 0.02,
      'initd'     : 0.0,
      'beta'      : 0.0
    }
    T         = 1000
    rwprob1      = SimpleTwoState(config)
    rwprob1.Phi  = np.array([[1], [1]])
    alg       = WGTD(config)
    ''' Test fixed points '''
    
    # off-policy fixed point
    thstar3 = mdp.MDP.getFixedPoint(rwprob1.Psst, rwprob1.exprt,\
                      rwprob1.Phi, rwprob1.dsb,\
                      rwprob1.Gamma, config['lmbda'])
    print(thstar3)
    
    runseed = 0
    rwprob1.initTrajectory(runseed)
    for t in range(T):
      probstep  = rwprob1.step()
      s                 = probstep['s']
      a                 = probstep['act']
      probstep['l']     = config['lmbda']
      probstep['lnext'] = config['lmbda']
      probstep['rho']   = rwprob1.getRho(s,a)
      alg.step(probstep)
    print(alg.th)
    assert((abs(thstar3-alg.th)<0.06).all())
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()