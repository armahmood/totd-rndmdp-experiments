'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import os
import sys
from pysrc.problems import stdrw
from pysrc.problems.stdrwsparsereward import StdRWSparseReward
from pysrc.algorithms.tdprediction.offpolicy import gtd
from pysrc.problems import mdp
from pysrc.problems.stdrwsparsereward import StdRWSparseReward2
from pysrc.algorithms.tdprediction.offpolicy import gtd
from pysrc.experiments import stdrwexp, stdrwexp2 

class Test(unittest.TestCase):


  def getConfig(self):
    ns = 13
    ftype = 'binary'
    gamma = 1.0
    gm = np.ones(ns) * gamma
    gm[0] = gm[ns - 1] = 0
    Gamma = np.diag(gm)
    nzG = np.diag(Gamma) != 0.0
    config = {'offpolicy':True, 
      'runseed':1, 
      'algname':'gtd', 
      'gamma':gamma, 
      'N':30, 
      'T':30, 
      'ftype':ftype, 
      'ns':ns, 
      'inits':(ns - 1) / 2, 
      'mright':0.5, 
      'pright':0.99, 
      'alpha':0.01, 
      'beta':0.0, 
      'lambda':0.9, 
      'lmbda':0.9, 
      'mdpseed':1000, 
      'Gamma':Gamma, 
      'initsdist':'statemiddle', 
      'Rstd':0.0, 
      'ns':ns, 
      'na':2, 
      'nf':int(np.ceil(np.log(np.sum(nzG) - 1) / np.log(2))), 
      'behavRight':0.5, 
      'targtRight':0.99}
    return config

  def testmakestdrwexp2exact(self):
    config = self.getConfig()

    rwprob1    = StdRWSparseReward(config)
    perf1      = stdrw.PerformanceMeasure(config, rwprob1)
    alg1       = gtd.GTD(config)
    rwprob2    = StdRWSparseReward2(config)
    perf2      = mdp.PerformanceMeasure(config, rwprob2)
    alg2       = gtd.GTD(config)
    
    rwprob1.setrunseed(config['runseed'])
    rwprob1.initepisode()
    alg1.initepisode()
    rwprob2.initTrajectory(config['runseed'])
    ep1 = 0
    ep2 = 0
    while ep1 < config['N'] and ep2 < config['N']:
      if ep1 < config['N']:
        probstep          = rwprob1.step()
        s1                = probstep['s']
        a1                = probstep['act']
        probstep['l']     = config['lambda']
        probstep['lnext'] = config['lambda']
        probstep['rho']   = rho1 = rwprob1.getRho(s1,a1)
        alg1.step(probstep)
        if rwprob1.isterminal():  
          perf1.calcMse(alg1, ep1)
          ep1 += 1
          rwprob1.initepisode()
          alg1.initepisode()
      if ep2 < config['N']:
        probstep          = rwprob2.step()
        s2                = probstep['s']
        a2                = probstep['act']
        probstep['l']     = config['lmbda']
        probstep['lnext'] = config['lmbda']
        probstep['rho']   = rho2 = rwprob2.getRho(s2,a2)
        alg2.step(probstep)
        if rwprob2.isTerminal():  
          perf2.calcMSPVE(alg2, ep2)
          ep2 += 1
          probstep          = rwprob2.step()
          alg2.initepisode()
      assert(s1==s2)
      assert(a1==a2)
      assert(rho1==rho2)
      assert((np.abs(perf1.MSPVE[ep1-1]-perf2.MSPVE[ep2-1])<0.1).all())
      
  def testmakingOneConfigsExact(self):
    config = self.getConfig()

    rwprob1    = StdRWSparseReward(config)
    perf1      = stdrw.PerformanceMeasure(config, rwprob1)
    alg1       = gtd.GTD(config)
    rwprob2    = StdRWSparseReward2(config)
    perf2      = mdp.PerformanceMeasure(config, rwprob2)
    alg2       = gtd.GTD(config)
    
    stdrwexp.runoneconfig(config, rwprob1, alg1, perf1)
    stdrwexp2.runoneconfig(config, rwprob2, alg2, perf2)
    
    for ep in range(config['N']):
      assert((np.abs(perf1.MSPVE[ep]-perf2.MSPVE[ep])<10**-10).all())

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testStdRandomWalkExp']
  unittest.main()
    