'''
Created on Apr 28, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
import pylab as pl
import copy

class BlindBee(object):
  '''
  A blind bee wander in a gridworld, only knows whether a trap is there
  and whether it is in the trap, indicated by four features. One of them 
  is never activated. Another four features indicate in which quadrant
  the bee is in. The question is how much upward displacement before
  the bee is in the trap.
  '''

  def __init__(self, params):
    self.rdrun      = np.random.RandomState(params['runseed'])
    params['rdrun'] = self.rdrun
    self.state      = BlindBeeState(params)
    if params['policy']=='uniform':
      self.policy   = UniformPolicy(params)
    elif params['policy']=='upward':
      self.policy   = UpwardPolicy(params)
    elif params['policy']=='upwardish':
      self.policy   = UpwardishPolicy(params)
    elif params['policy']=='smart':
      self.policy   = SmartPolicy(params)

  def initEpisode(self):
    self.state.setInitPos()

  def step(self):
    curstate    = copy.copy(self.state)
    a           = self.policy.getAction(self.state)
    self.state.getNextState(a)
    return {'a':a, 's':curstate, 'snext':self.state}
    
class BlindBeeState(object):
  def __init__(self, params):
    self.nrows        = params['nrows']
    self.ncols        = params['ncols']
    self.trapprob     = params['trapprob']
    self.trapduralim  = params['trapduralim'] 
    self.rdrun        = params['rdrun']
    
  def setInitPos(self):
    self.row      = self.nrows-1
    self.col      = self.rdrun.randint(0, self.ncols)
    self.traprow  = None
    self.trapdura = 0
    self.updisplacement  = 0
  
  def inTrap(self):
    return self.row==self.traprow  
  
  def getNextState(self, a):
    self.updisplacement = 0.
    if self.inTrap():
      self.row     = self.nrows-1
      self.col     = self.rdrun.randint(0, self.ncols)
    else:
      if a==BlindBeePolicy.LEFT and self.col>0:
        self.col -= 1
      elif a==BlindBeePolicy.RIGHT and self.col<self.ncols-1:
        self.col += 1
      elif a==BlindBeePolicy.UP and self.row>0:
        self.row -= 1
        self.updisplacement = 1.
      elif a==BlindBeePolicy.DOWN and self.row<self.nrows-1:
        self.row += 1
        self.updisplacement = -1.
      
    if self.trapdura==self.trapduralim:
      self.trapdura = 0
      self.traprow  = None
    if self.trapdura==0:
      rd1    = self.rdrun.uniform()
      if rd1<self.trapprob:
        self.traprow  = self.rdrun.randint(1, self.nrows-1)
        self.trapdura = 1
    else:
      self.trapdura += 1

  def isTerminal(self):
    return self.row==0
  
  def printState(self):
    print("row: "+str(self.row)+", col: "+str(self.col)+", traprow: "+str(self.traprow))

class BlindBeePolicy(object):
  LEFT  = 0
  UP    = 1
  RIGHT = 2
  DOWN  = 3
  nas   = 4  

  def __init__(self, params):
    self.rdrun  = params['rdrun']
  
  def getAction(self, state):
    rd      = self.rdrun.uniform()
    return pl.find(rd<np.cumsum(self.getPolicy(state)))[0]

class UniformPolicy(BlindBeePolicy):
  
  def __init__(self, params):
    BlindBeePolicy.__init__(self, params)
  
  def getPolicy(self, state):
    return np.ones(self.nas)/self.nas
  
  
class UpwardPolicy(BlindBeePolicy):
  
  def __init__(self, params):
    BlindBeePolicy.__init__(self, params)
  
  def getAction(self, state):
    return self.UP
  
class UpwardishPolicy(BlindBeePolicy):
  
  def __init__(self, params):
    BlindBeePolicy.__init__(self, params)

  def getPolicy(self, state):
    policy              = np.zeros(self.nas)
    policy[self.UP]     = 0.5
    policy[self.LEFT]   = policy[self.RIGHT]\
     =policy[self.DOWN]=(1-policy[self.UP])/3.

    return policy 
  
class SmartPolicy(BlindBeePolicy):
  
  def __init__(self, params):
    BlindBeePolicy.__init__(self, params)

  def getPolicy(self, state):
    policy = np.zeros(self.nas)
    if state.traprow==None:
      policy[self.UP]   = 1.
    else:
      policy[self.LEFT]=policy[self.RIGHT]=0.5

    return policy 
  
class Phi1(object):  

  def __init__(self):
    self.nf     = 4
  
  def getPhi1(self, state):
    phi     = np.zeros(self.nf)
    if state.traprow==None and not state.inTrap():
      phi[0]  = 1.
    elif state.traprow!=None and not state.inTrap():
      phi[1]  = 1.
    elif state.traprow==None and state.inTrap():
      raise
    elif state.traprow!=None and state.inTrap():
      phi[3]  = 1.
    
    return phi
 
  
  
      
    
    
    