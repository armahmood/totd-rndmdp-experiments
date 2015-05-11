'''
Created on May 2, 2014

@author: ashique
'''

import numpy as np
import pylab as pl
from pysrc.algorithms.tdprediction.tdprediction import TDPrediction

class TDR(TDPrediction):
  '''
  classdocs
  '''

  def __init__(self, config):
    '''
    Constructor
    '''
    self.nf = config['nf']
    self.th = np.zeros(self.nf)
    self.z = np.zeros(self.nf)
    self.alpha=config['alpha']
        
  def initepisode(self):
    self.z = np.zeros(self.nf)
    
  def step(self, params):
    phi=params['phi']; R=params['R']; phinext=params['phinext']
    g=params['g']; l=params['l']; gnext=params['gnext']
    delta = R + gnext*np.dot(phinext,self.th) - np.dot(phi, self.th)
    #if not (np.sum(phi==0.) + np.sum(phi==1.)==len(phi)):
    #  exit('not binary features')
    self.z = g*l*self.z*(phi==0.) + (phi!=0.)*phi
    self.th += self.alpha*delta*self.z

      