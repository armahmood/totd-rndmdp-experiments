'''
Created on Jun 2, 2014

@author: A. Rupam Mahmood

Implementation of WIS-LSTD(lambda)
by Mahmood, van Hasselt & Sutton (2014)

'''

import numpy as np
import pylab as pl
from pysrc.algorithms.tdprediction.prediction import Prediction

class WISLSTD(Prediction):
  
  def __init__(self, config):
    
    self.nf = config['nf']
    self.th = np.zeros(self.nf)
    self.z = np.zeros(self.nf)
    self.inita = config['inita']
    self.A = np.eye(self.nf)*self.inita
    self.b = np.zeros(self.nf)
    self.u = np.zeros(self.nf)
    self.V = np.zeros((self.nf, self.nf))
    
  def initepisode(self):
    self.z = np.zeros(self.nf)
    self.u = np.zeros(self.nf)
    self.V = np.zeros((self.nf, self.nf))
    
  def step(self, params):
    f=params['phi']; R=params['R']; fnext=params['phinext']
    g=params['g']; l=params['l']; gnext=params['gnext']
    rho=params['rho']; lnext=params['lnext']
    
    self.z = rho*(g*l*self.z + f)
    self.b = self.b + R*self.z + (rho-1)*self.u
    self.A = self.A + np.outer(self.z, (f - gnext*fnext)) + (rho-1)*self.V
    self.th = np.dot(pl.pinv(self.A), self.b)
    self.u = gnext*lnext*(rho*self.u + R*self.z)
    self.V = gnext*lnext*(rho*self.V + np.outer(self.z, (f - fnext)))
    