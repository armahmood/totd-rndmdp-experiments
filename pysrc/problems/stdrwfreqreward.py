'''
Created on May 2, 2014

@author: ashique
'''

import numpy as np
import pylab as pl
import random
from stdrw import StdRandomWalk

class StdRWFreqReward(StdRandomWalk):
  '''
  Standard random walk with sparse reward:
  +1 reward for going right
  '''

  def __init__(self, params):
    StdRandomWalk.__init__(self, params)

  @staticmethod
  def getRmat(ns):
    Rmat              = np.zeros((ns, ns)); 
    Rmat[(range(ns-1), range(1,ns))] = 1    
    return Rmat
