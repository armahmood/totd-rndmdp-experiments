'''
Created on Jun 2, 2014

@author: A. Rupam Mahmood

Implementation of off-policy-LSTD(lambda)
by Yu, (2010, Yu & Bertsekas 2009)

'''

import numpy as np
import pylab as pl

class OffPolicyLSTD(object):
  
  def __init__(self, config):
    
    self.nf = config['nf']
    self.th = np.zeros(self.nf)
    self.z = np.zeros(self.nf)
    self.inita = config['inita']
    self.A = np.eye(self.nf)*self.inita
    self.b = np.zeros(self.nf)
    
  def initepisode(self):
    self.z = np.zeros(self.nf)
    
  def step(self, params):
    f=params['phi']; R=params['R']; fnext=params['phinext']
    g=params['g']; l=params['l']; gnext=params['gnext']
    rho=params['rho']; lnext=params['lnext']
    
    delf = f - gnext*rho*fnext
    self.z = g*l*self.z + f
    self.A = self.A + np.outer(self.z, delf)
    self.b = self.b + rho*R*self.z
    self.th = np.dot(pl.pinv(self.A), self.b)
    self.z = self.z*rho
    