'''
Created on Mar 31, 2015

@author: A. Rupam Mahmood
'''

import numpy as np

class TDPrediction(object):
  '''
  TDPrediction abstract class 
  '''
  
  def __init__(self, config):
    self.nf   = config['nf']
    self.th   = np.zeros(self.nf)

  def estimate(self):
    return self.th
