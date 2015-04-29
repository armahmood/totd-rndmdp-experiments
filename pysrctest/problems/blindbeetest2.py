'''
Created on Apr 28, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems import blindbee
from matplotlib import pyplot as ppl
import copy

class BlindBeeTest2(object):
  NEPS = 5000

  @staticmethod
  def episodes(params, neps):
    prob            = blindbee.BlindBee(params)
    displacement    = [0.]
    ntraps          = 0
    while ntraps<neps:
      prob.initEpisode()
      while not prob.state.isTerminal() and ntraps<neps:
        prob.step()
        displacement[ntraps] += prob.state.updisplacement
        if prob.state.inTrap():
          displacement.append(0.) 
          ntraps += 1
          
    displacement = np.array(displacement[:-1]) 
    
    cummean = lambda x:x.cumsum() / np.arange(1, len(x) + 1)
    cumstd = lambda x:cummean(x ** 2) - cummean(x) ** 2
    cumstdrr = lambda x:cumstd(x) / np.arange(1, len(x) + 1)
    xs = np.arange(1, len(displacement) + 1, 20)
    ppl.errorbar(xs, cummean(displacement)[xs], cumstdrr(displacement)[xs])
    print(cummean(displacement)[-1])
    ppl.show()
    
  @staticmethod
  def getparams():
    params = {'runseed':1, 
      'nrows':10, 
      'ncols':10, 
      'trapprob':0.1, 
      'trapduralim':5}
    return params

  def testUniformPolicy(self):
    neps              = self.NEPS
    params            = self.getparams()
    params['policy']  = 'uniform'
    print("testUniformPolicy")
    self.episodes(params, neps)

  def testUpwardPolicy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = 'upward'
    print("testUpwardPolicy")
    self.episodes(params, neps)

  def testUpwardishPolicy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = 'upwardish'
    print("testUpwardishPolicy")
    self.episodes(params, neps)

  def testSmartPolicy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = 'smart'
    print("testSmartPolicy")
    self.episodes(params, neps)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    #unittest.main()
    test = BlindBeeTest2()
    test.testUniformPolicy()
    test.testUpwardPolicy()
    test.testUpwardishPolicy()
    test.testSmartPolicy()
