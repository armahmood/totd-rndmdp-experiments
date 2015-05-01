'''
Created on Apr 28, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems import blindbee
from matplotlib import pyplot as ppl
import copy

class BlindBeeTest(object):
  NEPS = 2000
  def testblindbee(self):
    rdrun   = np.random.RandomState(1)
    params  = {'rdrun':rdrun,
                 'nrows'        :4,
                 'ncols'        :4,
                 'trapprob'     :0.1,
                 'trapduralim'  :2,
                 }
    params['policy'] = blindbee.UniformPolicy(params)
    neps    = 3
    policy  = params['policy']
    state   = blindbee.BlindBeeState(params)
    state.printState()
    ep      = 0
    while ep < neps:
      a     = policy.getAction(state)
      state.getNextState(a)
      state.printState()
      if state.isTerminal():
        ep    += 1
      state.checkState()

  @staticmethod
  def episodes(params, neps):
    policy  = params['policy']
    state   = blindbee.BlindBeeState(params)
    steps   = np.zeros(neps+1)
    ep      = 0
    while ep<neps:
      a           = policy.getAction(state)
      state.getNextState(a)
      steps[ep] += 1
      if state.isTerminal():
        ep    += 1
      state.checkState()
    
    cummean = lambda x:x.cumsum() / np.arange(1, len(x) + 1)
    cumstd = lambda x:cummean(x ** 2) - cummean(x) ** 2
    cumstdrr = lambda x:cumstd(x) / np.arange(1, len(x) + 1)
    xs = np.arange(1, neps + 1, 20)
    ppl.errorbar(xs, cummean(steps)[xs], cumstdrr(steps)[xs])
    print cummean(steps)[neps]
    ppl.show()

  @staticmethod
  def getparams():
    rdrun   = np.random.RandomState(1)
    params  = {'rdrun':rdrun,
      'nrows':10, 
      'ncols':10, 
      'trapprob':0.1, 
      'trapduralim':10}
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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    #unittest.main()
    test = BlindBeeTest()
    test.testblindbee()
    test.testUpwardPolicy()
    test.testUpwardishPolicy()
    test.testSmartPolicy()
    test.testUniformPolicy()
    
