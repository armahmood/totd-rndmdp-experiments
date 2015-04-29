'''
Created on Apr 28, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
from pysrc.problems import blindbee
from matplotlib import pyplot as ppl

class BlindBeeTest(object):
  NEPS = 2000
  def testblindbee(self):
    params    = {
                 'runseed'      :1,
                 'policy'       :'upward',
                 'nrows'        :4,
                 'ncols'        :4,
                 'trapprob'     :1,
                 'trapduralim'  :20,
                 }
    prob      = blindbee.BlindBee(params)
    neps      = 3
    for ep in range(neps):
      prob.initEpisode()
      prob.state.printState()
      while not prob.state.isTerminal():
        prob.step()
        prob.state.printState()

  @staticmethod
  def episodes(params, neps):
    prob = blindbee.BlindBee(params)
    steps = np.zeros(neps)
    for ep in range(neps):
      prob.initEpisode()
      while not prob.state.isTerminal():
        prob.step()
        steps[ep] += 1
    
    cummean = lambda x:x.cumsum() / np.arange(1, len(x) + 1)
    cumstd = lambda x:cummean(x ** 2) - cummean(x) ** 2
    cumstdrr = lambda x:cumstd(x) / np.arange(1, len(x) + 1)
    xs = np.arange(1, len(steps) + 1, 20)
    ppl.errorbar(xs, cummean(steps)[xs], cumstdrr(steps)[xs])
    print cummean(steps)[-1]
    ppl.show()

  @staticmethod
  def getparams():
    params = {'runseed':1, 
      'nrows':10, 
      'ncols':10, 
      'trapprob':0.75, 
      'trapduralim':10}
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
    test = BlindBeeTest()
    test.testblindbee()
    test.testUpwardPolicy()
    test.testUniformPolicy()
    test.testUpwardishPolicy()
    test.testSmartPolicy()
    
