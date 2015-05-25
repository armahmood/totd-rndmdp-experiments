'''
Created on May 2, 2014

@author: ashique
'''

import numpy as np
import pylab as pl
import random
from stdrw import StdRandomWalk, StdRandomWalk2

class StdRWFreqPosNegReward(StdRandomWalk):
  '''
  Standard random walk with frequent reward:
  +1 reward for going right
  '''

  def __init__(self, params):
    StdRandomWalk.__init__(self, params)

  @staticmethod
  def getRmat(ns):
    Rmat              = np.zeros((ns, ns)); 
    Rmat[(range(ns-1), range(1,ns))] = 1    
    Rmat[(range(1,ns), range(ns-1))] = -1    
    return Rmat

class StdRWFreqPosNegReward2(StdRandomWalk2):
  
  def __init__(self, params):
    StdRandomWalk2.__init__(self,params)
    
  def getRssa(self):
    Rssa              = np.zeros((self.ns, self.ns, 2)); 
    Rssa[(range(self.ns-1), range(1,self.ns), 1)] = 1    
    Rssa[(range(1,self.ns), range(self.ns-1))] = -1    
    return Rssa
    
  
  