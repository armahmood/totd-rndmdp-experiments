'''
Created on May 27, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
from pysrc.algorithms.tdprediction.tdprediction import TDPrediction

class ZeroPredictor(TDPrediction):
  
  def __init__(self, config):
    TDPrediction.__init__(self,config)
    self.nf     = config['nf']
    
  def initepisode(self):
    return
    
  def step(self, params):
    return