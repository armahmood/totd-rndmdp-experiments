'''
Created on May 2, 2014

@author: ashique
'''

import numpy as np
import pylab as pl
import random
from stdrw import StdRandomWalk, StdRandomWalk2

class StdRWSparseReward(StdRandomWalk):
  '''
  Standard random walk with sparse reward:
  +1 reward only from the rightmost transition
  '''

  def __init__(self, params):
    StdRandomWalk.__init__(self, params)

  @staticmethod
  def getRmat(ns):
    Rmat              = np.zeros((ns, ns)); 
    Rmat[ns-2, ns-1]  = 1
    return Rmat

class StdRWSparseReward2(StdRandomWalk2):
  
  def __init__(self, params):
    StdRandomWalk2.__init__(self,params)
    
  def getRssa(self):
    Rssa              = np.zeros((self.ns, self.ns, 2)); 
    Rssa[self.ns-2, self.ns-1, 1] = 1    
    return Rssa
    
