'''
Created on Apr 16, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
import pylab as pl
import random
from pysrc.problems import mdp

class RandomMDP(mdp.MDP):
  '''
  It creates a randomly generated MDP object.
  Its description is close to that of garnet problems (Bhatnagar et al. 2009).
  '''

  def __init__(self, params):
    self.b          = params['b']   # branching factor    
    self.rtype      = params['rtype'] # distribution used to generate expected reward 
    self.rparam     = params['rparam'] # parameter for rtype
    mdp.MDP.__init__(self, params)
    
  def getPssa(self):
    Pssa = np.reshape(self.rdmdp.uniform(0, 1, self.ns*self.ns*self.na) + 10**-10 , \
                                (self.ns, self.ns, self.na))
    Pssa = np.zeros((self.ns, self.ns, self.na))
    for a in range(self.na):
      for s in range(self.ns):
        nextss = self.rdmdp.choice(range(self.ns), self.b, replace=False)
        nextscutpoints = np.sort(self.rdmdp.uniform(0,1,self.b-1))
        nextscutpoints = np.concatenate(([0], nextscutpoints, [1]))
        Pssa[s, nextss, a] = np.diff(nextscutpoints)
    return Pssa
    
  def getRssa(self):
    if self.rtype=='uniform':
      Rssa = np.reshape(self.rdmdp.uniform(0, self.rparam, self.ns*self.ns*self.na), 
                                  (self.ns, self.ns, self.na))
    if self.rtype=='normal':
      Rssa = np.reshape(self.rdmdp.normal(0, self.rparam, self.ns*self.ns*self.na), 
                                  (self.ns, self.ns, self.na))
    return Rssa
  
  def getBPolicy(self):
    return mdp.getRandomlySampledPolicy(self.ns, self.na, self.rdmdp, coverage=True)
  
  def getTPolicy(self):
    return mdp.getRandomlySampledPolicy(self.ns, self.na, self.rdmdp) 
    
    