'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import pylab as pl
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.tdprediction.offpolicy.etd import ETD
import pysrc.experiments.stdrwexp as stdrwexp
from pysrc.problems.stdrw import PerformanceMeasure
from pysrc.problems import mdp
from pysrc.problems.simpletwostate import SimpleTwoState

class Test(unittest.TestCase):

  def testetdonsparserewardtabular(self):
    ns = 13
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
              'alpha'     : 0.01,
              'beta'      : 0.0
              }
    alg         = ETD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstarMSE.T[0]
    print alg.th
    assert (abs(perf.thstarMSE.T[0] - alg.th) < 0.02).all()

  def testetdonsparserewardbinary(self):
    ns = 13
    config = {
              'N'      : 500,
              'ftype'     : 'binary',
              'ns'        : ns,
              'inits'     : (ns-1)/2,
              'mright'    : 0.5,
              'pright'    : 0.9,
              'runseed'   : 1,
              'nf'        : int(np.ceil(np.log(ns-1)/np.log(2))),
              'gamma'     : 0.9,
              'lambda'    : 0.5,
              'alpha'     : 0.001,
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
              'N'      : 2000,
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

  def testEtdOnSimpleTwoStateFuncApprox(self):
    lmbda   = 0.
    config  = \
    {
      'offpolicy' : True,
      'nf'        : 1,
      'ftype'     : None,
      'Rstd'      : 0.0,
      'initsdist' : 'steadystate',
      'Gamma'     : 0.9*np.eye(2),
      'mdpseed'   : 1000,
      'lambda'    : lmbda,
      'alpha'     : 0.01,
    }
    T         = 1000
    nruns     = 1
    rwprob1      = SimpleTwoState(config)
    rwprob1.Phi  = np.array([[1], [1]])
    alg       = ETD(config)
    ''' Test fixed points '''
    
    # target on-policy fixed point
    thstar0 = mdp.MDP.getFixedPoint(rwprob1.Psst, rwprob1.exprt,\
                      rwprob1.Phi, mdp.steadystateprob(rwprob1.Psst),\
                      rwprob1.Gamma, lmbda)
    print(mdp.steadystateprob(rwprob1.Psst))
    print thstar0

    # off-policy fixed point
    thstar1 = mdp.MDP.getFixedPoint(rwprob1.Psst, rwprob1.exprt,\
                      rwprob1.Phi, rwprob1.dsb,\
                      rwprob1.Gamma, lmbda)
    print(rwprob1.dsb)
    print(thstar1)
    
    # emphatic fixed point
    ImPG    = np.eye(rwprob1.ns) - np.dot(rwprob1.Psst, rwprob1.Gamma)
    ImPLG   = np.eye(rwprob1.ns) - np.dot(lmbda*rwprob1.Psst, rwprob1.Gamma)
    ImPL    = np.dot( pl.inv(ImPLG), ImPG )
    m       = np.dot(rwprob1.dsb.T, pl.inv(ImPL))
    m       = m/np.sum(m)
    thstar2 = mdp.MDP.getFixedPoint(rwprob1.Psst, rwprob1.exprt,\
                      rwprob1.Phi, m,\
                      rwprob1.Gamma, lmbda)
    print(m)
    print(thstar2)
    
    ths   = np.zeros((nruns, config['nf']))
    for runseed in range(nruns): 
      rwprob1.initTrajectory(runseed)
      for t in range(T):
        probstep  = rwprob1.step()
        s                 = probstep['s']
        a                 = probstep['act']
        probstep['l']     = config['lambda']
        probstep['lnext'] = config['lambda']
        probstep['rho']   = rwprob1.getRho(s,a)
        alg.step(probstep)
      ths[runseed]  = alg.th
      meanth        = np.mean(ths, 0)
    print("FuncApprox: "+ str(meanth))
    assert((abs(thstar2-meanth)<0.15).all())
                       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()