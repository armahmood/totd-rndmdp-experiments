'''
Created on Apr 16, 2015

@author: A. Rupam Mahmood
'''
import unittest
import sys
import numpy as np
from pysrc.problems import mdp
from pysrc.problems.simpletwostate import SimpleTwoState

class Test(unittest.TestCase):

  def testSimpleTwoStateTabular(self):
    config = \
    {
      'nf'        : 2,
      'ftype'     : 'tabular',
      'fs'        : 2,
      'Rstd'      : 0.0,
      'initsdist' : 'steadystate',
      'Gamma'     : 0.9*np.eye(2),
      'mdpseed'   : 1000,
    }
    T       = 10000
    prob    = SimpleTwoState(config)
    
    ''' Test fixed points '''
    
    # behavior on-policy fixed point
    thstar1 = mdp.getFixedPoint(prob.Pssb, prob.exprb,\
                      prob.Phi, prob.dsb, prob.Gamma, 1.)
    assert((abs(thstar1-np.array([2.25, 2.75]))<10**-10).all())
    
    # target on-policy fixed point
    thstar2 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, mdp.steadystateprob(prob.Psst),\
                      prob.Gamma, 1.)
    assert((abs(thstar2-np.array([8.8209, 9.8109]))<10**-10).all())

    # off-policy fixed point
    thstar3 = mdp.getFixedPoint(prob.Psst, prob.exprt,\
                      prob.Phi, prob.dsb,\
                      prob.Gamma, 1.)
    assert((abs(thstar2-thstar3)<10**-10).all())
    
    runseed = 0
    prob.initTrajectory(runseed)
    eds = np.zeros(2)
    for t in range(T):
      eds[prob.s]  += 1.
      prob.step()
    eds = eds/(np.sum(eds))
    assert((abs(eds-prob.dsb)<0.05).all())
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()