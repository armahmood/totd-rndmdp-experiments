'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward2
from pysrc.problems.stdrwfreqreward import StdRWFreqReward2
from pysrc.algorithms.tdprediction.offpolicy.wtd import WTD
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
    initdist = np.zeros(ns)
    initdist[(ns - 1) / 2] = 1.
    config = {
              'offpolicy' : True,
              'mdpseed'   : 1000,
              'Gamma'     : Gamma,
              'initsdist' : initdist,
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
              'initd'     : 0.0
              }
    alg         = WTD(config)
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
    initdist = np.zeros(ns)
    initdist[(ns - 1) / 2] = 1.
    config = {
              'offpolicy' : True,
              'mdpseed'   : 1000,
              'Gamma'     : Gamma,
              'initsdist' : initdist,
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
              'initd'     : 0.01
              }
    alg         = WTD(config)
    rwprob      = StdRWSparseReward2(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp2.runoneconfig(config, rwprob, alg, perf)
    print "binary"
    print perf.thstarMSPBE.T
    print alg.th
    assert (abs(perf.thstarMSPBE.T - alg.th) < 0.05).all()

  def testonfreqrewardtabular(self):
    ns = 7
    gamma = 0.9
    gm = np.ones(ns) * gamma
    gm[0] = gm[ns - 1] = 0
    Gamma = np.diag(gm)
    nzG              = np.diag(Gamma)!=0.0
    initdist = np.zeros(ns)
    initdist[(ns - 1) / 2] = 1.
    config = {
              'offpolicy' : True,
              'mdpseed'   : 1000,
              'Gamma'     : Gamma,
              'initsdist' : initdist,
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
              'eta'       : 0.00,
              'initd'     : 0.00
              }
    alg         = WTD(config)
    rwprob      = StdRWFreqReward2(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp2.runoneconfig(config, rwprob, alg, perf)
    print perf.thstar.T
    print alg.th
    assert (abs(perf.thstar.T - alg.th) < 0.05).all()

  def testOnSimpleTwoStateFuncApprox(self):
    config = \
    {
      'nf'        : 1,
      'ftype'     : None,
      'Rstd'      : 0.0,
      'initsdist' : 'steadystate',
      'Gamma'     : 0.9*np.eye(2),
      'mdpseed'   : 1000,
      'lmbda'    : 0.0,
      'eta'       : 0.005,
      'initd'     : 0.0
    }
    T         = 5000
    prob      = SimpleTwoState(config)
    prob.Phi  = np.array([[1], [1]])
    alg       = WTD(config)
    ''' Test fixed points '''
    
    # off-policy fixed point
    thstar3 = mdp.MDP.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, prob.dsb,\
                      prob.Gamma, config['lmbda'])
    print(thstar3)
    
    runseed = 0
    prob.initTrajectory(runseed)
    for t in range(T):
      probstep  = prob.step()
      s                 = probstep['s']
      a                 = probstep['act']
      probstep['l']     = config['lmbda']
      probstep['lnext'] = config['lmbda']
      probstep['rho']   = prob.getRho(s,a)
      alg.step(probstep)
    print(alg.th)
    assert((abs(thstar3-alg.th)<0.06).all())
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()