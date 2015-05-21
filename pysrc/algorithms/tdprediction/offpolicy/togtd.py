'''
Created on Jun 2, 2014

@author: A. Rupam Mahmood
'''

import numpy as np
import pylab as pl
from pysrc.algorithms.tdprediction.tdprediction import TDPrediction

class TOGTD(TDPrediction):
  
  def __init__(self, config):
    
    self.nf = config['nf']
    self.th = np.zeros(self.nf)
    self.z = np.zeros(self.nf)
    self.w = np.zeros(self.nf)
    self.zw = np.zeros(self.nf)
    self.zg = np.zeros(self.nf)
    self.alpha = config['alpha']
    self.beta = config['beta']
    self.rhoprev = 0.
    self.predprev = 0.
    
  def initepisode(self):
    self.z = np.zeros(self.nf)
    
  def step(self, params):
    phi=params['phi']; R=params['R']; phinext=params['phinext']
    g=params['g']; l=params['l']; gnext=params['gnext']
    rho=params['rho']; lnext=params['lnext']
    
    prednext = np.dot(phinext,self.th)
    delta = R + gnext*prednext - np.dot(phi, self.th)
    self.z = rho*(g*l*self.z + \
                  self.alpha*(1-rho*g*l*np.dot(phi, self.z))*phi)
    self.zw = self.rhoprev*g*l*self.zw \
      + self.beta*(1 - self.rhoprev*g*l*np.dot(phi, self.zw))*phi
    self.zg = rho*g*l*self.zg + rho*phi
    self.th += delta*self.z \
      + ( np.dot(self.th, phi) - self.predprev )*(self.z - self.alpha*rho*phi)\
       - self.alpha*gnext*(1-lnext)*np.dot(self.w, self.zg)*phinext
    self.w += rho*delta*self.zw - self.beta*phi*np.dot(self.w, phi)
    self.rhoprev = rho
    self.predprev = prednext


