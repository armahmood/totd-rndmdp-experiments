'''
Created on Jan 2015

@author: A. Rupam Mahmood
'''

import numpy as np
import pylab as pl
from pysrc.algorithms.tdprediction.tdprediction import TDPrediction

class WTOGTD(TDPrediction):
  
  def __init__(self, config):
    
    self.nf = config['nf']
    self.th = np.zeros(self.nf)
    self.z = np.zeros(self.nf)
    self.w = np.zeros(self.nf)
    self.zw = np.zeros(self.nf)
    self.zg = np.zeros(self.nf)
    self.beta = config['beta']
    self.eta    = config['eta']
    self.initd  = config['initd']
    self.d      = np.ones(self.nf)*self.initd
    self.v      = np.zeros(self.nf)
    self.rhoprev = 0.
    self.predprev = 0.
    
  def initepisode(self):
    self.z = np.zeros(self.nf)
    
  def step(self, params):
    phi=params['phi']; R=params['R']; phinext=params['phinext']
    g=params['g']; l=params['l']; gnext=params['gnext']
    rho=params['rho']; lnext=params['lnext']
    
    self.d        = self.d - self.eta*phi*phi*self.d \
            + rho*phi*phi \
            + (rho-1)*g*l*(self.v - self.eta*phi*phi*self.v)
    self.dtemp    = np.copy(self.d)
    self.dtemp[self.dtemp==0.0] = 1
    alpha         = 1/self.dtemp
    alphaphi      = alpha*phi
    self.v        = rho*g*l*(self.v-self.eta*phi*phi*self.v) \
            + rho*phi*phi

    prednext = np.dot(phinext,self.th)
    delta = R + gnext*prednext - np.dot(phi, self.th)
    self.z = rho*(g*l*self.z + \
                  (1-rho*g*l*np.dot(phi, self.z))*alphaphi)
    self.zw = self.rhoprev*g*l*self.zw \
      + self.beta*(1 - self.rhoprev*g*l*np.dot(phi, self.zw))*phi
    self.zg = rho*g*l*self.zg + rho*phi
    self.th += delta*self.z \
      + ( np.dot(self.th, phi) - self.predprev )*(self.z - rho*alphaphi)\
       - gnext*(1-lnext)*np.dot(self.w, self.zg)*alpha*phinext
    self.w += rho*delta*self.zw - self.beta*phi*np.dot(self.w, phi)
    self.rhoprev = rho
    self.predprev = prednext


