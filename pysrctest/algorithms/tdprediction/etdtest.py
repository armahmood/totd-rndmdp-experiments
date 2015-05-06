'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import pylab as pl
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.tdprediction.etd import ETD
import pysrc.experiments.stdrwexp as stdrwexp
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.problems import mdp
from pysrc.problems.simpletwostate import SimpleTwoState

class Test(unittest.TestCase):

  def testetdonsparserewardtabular(self):
    ns = 13
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
              'alpha'     : 0.005,
              'beta'      : 0.0
              }
    alg         = ETD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.th
    assert (abs(perf.thstarMSE.T[0] - alg.th) < 0.01).all()

  def testetdonsparserewardbinary(self):
    ns = 13
    config = {
              'neps'      : 2000,
              'ftype'     : 'binary',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : int(np.ceil(np.log(ns-1)/np.log(2))),
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'alpha'     : 0.0005,
              'beta'      : 0.0
              }
    alg         = ETD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSPBE.T[0]
    print alg.th
    assert (abs(perf.thstarMSPBE.T[0] - alg.th) < 0.02).all()
   
  def testetdonfreqrewardtabular(self):
    ns = 7
    config = {
              'neps'      : 2000,
              'ftype'     : 'tabular',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : ns-2,
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'alpha'     : 0.005,
              'beta'      : 0.0
              }
    alg         = ETD(config)
    rwprob      = StdRWFreqReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.th
    assert (abs(perf.thstarMSE.T[0] - alg.th) < 0.06).all()

  def testEtdOnSimpleTwoStateTabular(self):
    config = \
    {
      'nf'        : 2,
      'ftype'     : 'tabular',
      'Rstd'      : 0.0,
      'initsdist' : 'steadystate',
      'Gamma'     : 0.9*np.eye(2),
      'mdpseed'   : 1000,
      'lambda'    : 0.5,
      'alpha'     : 0.00005,
    }
    T       = 300000
    prob    = SimpleTwoState(config)
    alg     = ETD(config)
    ''' Test fixed points '''
    
    # off-policy fixed point
    thstar3 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, prob.dsb,\
                      prob.Gamma, 1.)
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
    print("Tabular: "+ str(alg.th))
    assert((abs(thstar3-alg.th)<0.1).all())

  def testEtdOnSimpleTwoStateFuncApprox(self):
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
      'alpha'     : 0.0001,
    }
    T         = 10000
    nruns     = 1
    prob      = SimpleTwoState(config)
    prob.Phi  = np.array([[1], [1]])
    alg       = ETD(config)
    ''' Test fixed points '''
    
    # target on-policy fixed point
    thstar0 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, mdp.steadystateprob(prob.Psst),\
                      prob.Gamma, lmbda)
    print(mdp.steadystateprob(prob.Psst))
    print thstar0

    # off-policy fixed point
    thstar1 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, prob.dsb,\
                      prob.Gamma, lmbda)
    print(prob.dsb)
    print(thstar1)
    
    # emphatic fixed point
    ImPG    = np.eye(prob.ns) - np.dot(prob.Psst, prob.Gamma)
    ImPLG   = np.eye(prob.ns) - np.dot(lmbda*prob.Psst, prob.Gamma)
    ImPL    = np.dot( pl.inv(ImPLG), ImPG )
    m       = np.dot(prob.dsb.T, pl.inv(ImPL))
    m       = m/np.sum(m)
    thstar2 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, m,\
                      prob.Gamma, lmbda)
    print(m)
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
        alg.step(probstep)
      ths[runseed]  = alg.th
      meanth        = np.mean(ths, 0)
    print("FuncApprox: "+ str(meanth))
    assert((abs(thstar2-meanth)<0.1).all())
                       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()