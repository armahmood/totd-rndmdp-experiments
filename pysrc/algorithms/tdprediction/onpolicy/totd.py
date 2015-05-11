'''
Created on May 2, 2014

@author: ashique
'''

import numpy as np
import pylab as pl
from pysrc.algorithms.tdprediction.tdprediction import TDPrediction

class TOTD(TDPrediction):
  '''
  classdocs
  '''

  def __init__(self, config):
    '''
    Constructor
    '''
    self.nf       = config['nf']
    self.th       = np.zeros(self.nf)
    self.z        = np.zeros(self.nf)
    self.predprev = 0.
    self.alpha    = config['alpha']
        
  def initepisode(self):
    self.z = np.zeros(self.nf)
    
  def step(self, params):
    phi=params['phi']; R=params['R']; phinext=params['phinext']
    g=params['g']; l=params['l']; gnext=params['gnext']
    prednext = np.dot(phinext,self.th)
    delta = R + gnext*prednext - self.predprev
    self.z = g*l*self.z + self.alpha*phi - self.alpha*g*l*np.dot(self.z,phi)*phi
    self.th += delta*self.z + self.alpha*(self.predprev - np.dot(phi, self.th))*phi
    self.predprev = prednext
      