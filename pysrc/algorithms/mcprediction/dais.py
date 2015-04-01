'''
Created on Mar 31, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
from pysrc.algorithms.mcprediction.mcprediction import MCPrediction

class DAIS(MCPrediction):
  '''
  Discounting-aware importance sampling estimator.
  This class requires a full trajectory to be generated
  first in order to update its estimate.
  '''

  def __init__(self, config):
    '''
    initialize the estimator and the occupancy measure
    '''
    
    MCPrediction.__init__(self, config)
    
    self.nvisits    = np.zeros(self.ns)
    self.rewards    = []
    self.rhos       = []
    self.gammas     = []
    self.features   = []
  
  def initepisode(self):
    self.rewards    = []
    self.rhos       = []
    self.gammas     = []
    self.features   = []
  
  def step(self, params):
    R=params['R']
    f=params['phi']; fnext=params['phinext']
    gnext=params['gnext']
    rho=params['rho']; 
    
    self.rewards.append(R)
    self.rhos.append(rho)
    self.gammas.append(gnext)
    self.features.append(f)
    if (fnext==0.0).all() or gnext==0.0: # end of an episode
      G   = 0.0
      D   = 0.0
      for t in np.arange(len(self.rewards)-1, -1, -1):
        D                   = self.rhos[t]*(1+self.gammas[t]*(D-1))
        G                   = self.gammas[t]*self.rhos[t]*G + D*self.rewards[t]
        f_t                 = self.features[t]
        self.nvisits[f_t==1]  += 1.0
        self.V[f_t==1]        += (G - self.V[f_t==1])/self.nvisits[f_t==1]
      self.initepisode()
        







        