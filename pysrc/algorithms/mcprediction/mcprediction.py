'''
Created on Mar 31, 2015

@author: A. Rupam Mahmood
'''

import numpy as np

class MCPrediction(object):
  '''
  Abstract class MCPrediction
  '''

  def __init__(self, config):
    '''
    Constructor
    '''
    self.ns         = config['nf']
    self.V          = np.zeros(self.ns)
    assert( config['ftype']=='tabular' )
        
  def estimate(self):
    return self.V