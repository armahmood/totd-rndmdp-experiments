'''
Created on Jun 2, 2014

@author: A. Rupam Mahmood

Implementation of Emphatic TD(lambda)
by Sutton, Mahmood & White (2015)

'''

import numpy as np
from pysrc.algorithms.tdprediction.tdprediction import TDPrediction

class ETD(TDPrediction):
  
  def __init__(self, config):
    
    TDPrediction.__init__(self,config)
    self.z      = np.zeros(self.nf)
    self.F      = 0.0
    self.alpha  = config['alpha']
    
  def initepisode(self):
    self.z = np.zeros(self.nf)
    
  def step(self, params):
    phi =params['phi']; R     =params['R']; phinext =params['phinext']
    g   =params['g'];   l     =params['l']; gnext   =params['gnext']
    rho =params['rho']; lnext =params['lnext']
    key ='I'
    I   =key if key in params else 1.  
        
    delta   = R + gnext*np.dot(phinext,self.th) - np.dot(phi, self.th)
    self.F  += I
    M       = l*I + (1-l)*self.F
    self.z  = rho*(g*l*self.z + M*phi)
    self.th += self.alpha*delta*self.z
    self.F  *= gnext*rho

