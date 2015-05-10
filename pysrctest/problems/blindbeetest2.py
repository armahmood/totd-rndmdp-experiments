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
  NEPS = 10000

  @staticmethod
  def episodes(params, neps):
    policy  = params['policy']
    state   = blindbee.BlindBeeState(params)
    displacement    = [0.]
    trap            = 0
    while trap<neps:
      a           = policy.getAction(state)
      state.getNextState(a)
      displacement[trap] += state.updisplacement
      if state.inTrap():
        displacement.append(0.)
        trap += 1 
      state.checkState()
          
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
    rdrun   = np.random.RandomState(1)
    params  = {'rdrun':rdrun,
      'nrows':4, 
      'ncols':4, 
      'trapprob':0.4, 
      'trapduralim':2}
    return params

  def testUniformPolicy(self):
    neps              = self.NEPS
    params            = self.getparams()
    params['policy']  = blindbee.UniformPolicy(params)
    print("testUniformPolicy")
    self.episodes(params, neps)

  def testUpwardPolicy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = blindbee.UpwardPolicy(params)
    print("testUpwardPolicy")
    self.episodes(params, neps)

  def testUpward2Policy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = blindbee.Upward2Policy(params)
    print("testUpward2Policy")
    self.episodes(params, neps)

  def testUpwardishPolicy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = blindbee.UpwardishPolicy(params)
    print("testUpwardishPolicy")
    self.episodes(params, neps)

  def testSmartPolicy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = blindbee.SmartPolicy(params)
    print("testSmartPolicy")
    self.episodes(params, neps)

  def testSmart2Policy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = blindbee.Smart2Policy(params)
    print("testSmart2Policy")
    self.episodes(params, neps)

  def testSmart3Policy(self):
    neps = self.NEPS
    params = self.getparams()
    params['policy']  = blindbee.Smart3Policy(params)
    print("testSmart3Policy")
    self.episodes(params, neps)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    #unittest.main()
    test = BlindBeeTest2()
    #test.testUpwardPolicy()
    test.testUpward2Policy()
    #test.testUpwardishPolicy()
    test.testSmart3Policy()
    #test.testSmart2Policy()
    #test.testSmartPolicy()
    test.testUniformPolicy()
