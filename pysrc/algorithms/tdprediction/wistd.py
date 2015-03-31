'''
Created on Jun 2, 2014

@author: ashique
'''

import numpy as np
import pylab as pl
from pysrc.algorithms.tdprediction.prediction import Prediction

class WISTD(Prediction):
  
  def __init__(self, config):
    
    self.nf     = config['nf']
    self.eta    = config['eta']
    self.initd  = config['initd']
    self.th     = np.zeros(self.nf)
    self.z      = np.zeros(self.nf)
    self.u      = np.zeros(self.nf)
    self.d      = np.ones(self.nf)*self.initd
    self.v      = np.zeros(self.nf)
    self.predprev = 0.
    
  def initepisode(self):
    self.z = np.zeros(self.nf)
    self.u = np.zeros(self.nf)
    self.v = np.zeros(self.nf)
    
  def step(self, params):
    phi=params['phi']; R=params['R']; phinext=params['phinext']
    g=params['g']; l=params['l']; gnext=params['gnext']
    rho=params['rho']; lnext=params['lnext']
    
    self.d        = self.d - self.eta*phi*phi*self.d \
            + rho*phi*phi \
            + (rho-1)*g*l*(self.v - self.eta*phi*phi*self.v)
    dtemp    = np.copy(self.d)
    dtemp[dtemp==0.0] = 1
    alpha         = 1/dtemp
    alphaphi      = alpha*phi
    self.v        = rho*g*l*(self.v-self.eta*phi*phi*self.v) \
            + rho*phi*phi
    self.z        = rho*alphaphi \
            + g*l*rho*(self.z - rho*np.dot(phi,self.z)*alphaphi)
    prednext      = np.dot(phinext,self.th)
    self.th       = self.th + (R+gnext*prednext-self.predprev)*self.z \
            + rho*(self.predprev-np.dot(self.th,phi))*alphaphi\
            + (rho-1)*g*l*(self.u-rho*np.dot(self.u,phi)*alphaphi)
    self.u        = rho*g*l*(self.u-rho*np.dot(self.u,phi)*alphaphi)\
            + (R+prednext-self.predprev)*self.z

    self.predprev = prednext
    


