'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''

import unittest
import numpy as np
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.problems.stdrwfreqreward import StdRWFreqReward
from pysrc.algorithms.wislstd import WISLSTD
import pysrc.experiments.stdrwexp as stdrwexp
from pysrc.problems.stdrw import PerformanceMeasure

class Test(unittest.TestCase):

  def testgtdonsparserewardtabular(self):
    ns = 7
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
              'inita'     : 0.01,
              }
    alg         = WISLSTD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstar.T[0]
    print alg.th
    assert (abs(perf.thstar.T[0] - alg.th) < 0.01).all()

  def testgtdonsparserewardbinary(self):
    ns = 7
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
              'inita'     : 0.01,              
              }
    alg         = WISLSTD(config)
    rwprob      = StdRWSparseReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstar.T[0]
    print alg.th
    assert (abs(perf.thstar.T[0] - alg.th) < 0.05).all()

  def testgtdonfreqrewardtabular(self):
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
              'inita'     : 0.1,              
              }
    alg         = WISLSTD(config)
    rwprob      = StdRWFreqReward(config)
    perf      = PerformanceMeasure(config, rwprob)
    stdrwexp.runoneconfig(config, rwprob, alg, perf)
    print perf.thstar.T[0]
    print alg.th
    assert (abs(perf.thstar.T[0] - alg.th) < 0.05).all()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()