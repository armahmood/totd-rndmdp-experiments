'''
Created on Apr 16, 2015

@author: A. Rupam Mahmood
'''
import unittest
import sys
import numpy as np
from pysrc.problems import mdp
from pysrc.problems.randommdp import RandomMDP

class Test(unittest.TestCase):

  def testRandomMDPTabular(self):
    ns      = 10
    config  = \
    {
      'ns'        : ns,
      'na'        : 3,
      'b'         : 3,
      'ftype'     : 'tabular',
      'nf'        : 10,
      'rtype'     : 'uniform',
      'rparam'    : 1,
      'Rstd'      : 0.0,
      'initsdist' : 'steadystate',
      'Gamma'     : 0.9*np.eye(ns),
      'mdpseed'   : 1000,
    }
    T       = 10000
    rwprob1    = RandomMDP(config)
    runseed = 0
    rwprob1.initTrajectory(runseed)
    eds = np.zeros(config['ns'])
    for t in range(T):
      eds[rwprob1.s]  += 1.
      rwprob1.step()
    eds = eds/(np.sum(eds))
    print(eds)
    print(rwprob1.dsb)
    assert((abs(eds-rwprob1.dsb)<0.05).all())
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()