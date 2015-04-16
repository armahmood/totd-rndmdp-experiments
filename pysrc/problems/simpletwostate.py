'''
Created on Apr 16, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
from pysrc.problems import mdp


class SimpleTwoState(mdp.MDP):
  '''
    Simple two state MDP from MDP class
  '''

  def __init__(self, params):
    params['ns']      = 2
    params['na']      = 2
    mdp.MDP.__init__(self, params)

  def getPssa(self):
    Pssa        = np.zeros((2, 2, 2))
    L           = 0
    R           = 1
    Pssa[L,L,L] = 1.
    Pssa[L,R,R] = 1.
    Pssa[R,L,L] = 1.
    Pssa[R,R,R] = 1. 
    return Pssa
  
  def getRssa(self):
    Rssa        = np.zeros((2, 2, 2))
    L           = 0
    R           = 1
    Rssa[L,L,L] = -1
    Rssa[L,R,R] = 0.
    Rssa[R,L,L] = 0.
    Rssa[R,R,R] = 1. 
    return Rssa
  
  def getBPolicy(self):
    return mdp.getUniformRandomPolicy(self.ns, self.na)
  
  def getTPolicy(self):
    return np.array([[0.1, 0.9], [0.1, 0.9]])

  