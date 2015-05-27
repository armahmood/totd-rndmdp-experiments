'''
Created on Apr 16, 2015

@author: A. Rupam Mahmood
'''
import unittest
import sys
import numpy as np
from pysrc.problems import offrandommdp
from pysrc.experiments import offrndmdpexp
from pysrc.problems.mdp import PerformanceMeasure
from pysrc.algorithms.tdprediction.offpolicy import wislstd
from pysrc.algorithms.tdprediction.offpolicy import gtd

class Test(unittest.TestCase):

  def testwithwislstdtabular(self):
    ns          = 10
    N           = 1000
    mdpseed     = 1000
    gamma       = 0.9
    config     = \
                     {
                     'mdpseed'    : mdpseed,
                     'runseed'    : 1,
                     'ftype'      : 'tabular',
                     'algname'    : 'td',
                     'Gamma'      : gamma,
                     'numzerogs'  : 1,
                     'T'          : N,
                     'N'          : N,
                     'ns'         : ns,
                     'na'         : 3,
                     'nf'         : ns,
                     'b'          : 3,
                     'rtype'      : 'uniform', 
                     'rparam'     : 1,
                     'Rstd'       : 0.0,
                     'initsdist'  : 'statezero',
                     'inita'      : 1,
                     #'alpha'      : 0.1,
                     #'beta'       : 0.0,
                     'lmbda'      : 0.9,
                     'bpoltype'   : 'uniform',
                     'tpoltype'   : 'random'
                     }
    prob    = offrandommdp.OffRandomMDP(config)
    alg     = wislstd.WISLSTD(config)
    #alg     = gtd.GTD(config)
    perf    = PerformanceMeasure(config, prob)
    offrndmdpexp.runoneconfig(config, prob, alg, perf)
    print perf.thstar
    print alg.th
    assert((np.abs(perf.thstar-alg.th)<0.2).all())

  def testwithwislstdbinary(self):
    ns          = 10
    N           = 1000
    mdpseed     = 1000
    gamma       = 0.9
    config     = \
                     {
                     'mdpseed'    : mdpseed,
                     'runseed'    : 1,
                     'ftype'      : 'binary',
                     'algname'    : 'td',
                     'Gamma'      : gamma,
                     'numzerogs'  : 1,
                     'T'          : N,
                     'N'          : N,
                     'ns'         : ns,
                     'na'         : 3,
                     'nf'         : int(np.ceil(np.log(ns+1)/np.log(2))),
                     'b'          : 3,
                     'rtype'      : 'uniform', 
                     'rparam'     : 1,
                     'Rstd'       : 0.0,
                     'initsdist'  : 'statezero',
                     'inita'      : 1,
                     #'alpha'      : 0.1,
                     #'beta'       : 0.0,
                     'lmbda'      : 0.9,
                     'bpoltype'   : 'uniform',
                     'tpoltype'   : 'random'
                     }
    prob    = offrandommdp.OffRandomMDP(config)
    alg     = wislstd.WISLSTD(config)
    #alg     = gtd.GTD(config)
    perf    = PerformanceMeasure(config, prob)
    offrndmdpexp.runoneconfig(config, prob, alg, perf)
    print perf.thstar
    print alg.th
    assert((np.abs(perf.thstar-alg.th)<0.2).all())
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()