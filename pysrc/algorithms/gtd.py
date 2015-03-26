'''
Created on Jun 2, 2014

@author: ashique
'''

import numpy as np
import pylab as pl

class GTD(object):
  
  def __init__(self, config):
    
    self.nf = config['nf']
    self.th = np.zeros(self.nf)
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


