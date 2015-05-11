'''
Created on Jun 2, 2014

@author: A. Rupam Mahmood

Implementation of GTD(lambda)
by Maei (2011, Maei & Sutton 2010)

'''

import numpy as np
from pysrc.algorithms.tdprediction.tdprediction import TDPrediction

class GTD(TDPrediction):
  
  def __init__(self, config):
    
    TDPrediction.__init__(self,config)
    self.z = np.zeros(self.nf)
    self.w = np.zeros(self.nf)
    self.alpha = config['alpha']
    self.beta = config['beta']
    
  def initepisode(self):
    self.z = np.zeros(self.nf)
    
  def step(self, params):
    phi=params['phi']; R=params['R']; phinext=params['phinext']
    g=params['g']; l=params['l']; gnext=params['gnext']
    rho=params['rho']; lnext=params['lnext']
    
    
    delta = R + gnext*np.dot(phinext,self.th) - np.dot(phi, self.th)
    self.z = rho*(g*l*self.z + phi)
    self.th += self.alpha*delta*self.z \
      - self.alpha*gnext*(1-lnext)*np.dot(self.z, self.w)*phinext
    self.w += self.beta*(delta*self.z - np.dot(phi, self.w)*phi)

