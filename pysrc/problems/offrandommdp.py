'''
Created on Apr 16, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
import pylab as pl
import random
from pysrc.problems import mdp, randommdp

class OffRandomMDP(randommdp.RandomMDP):
  '''
  It creates a randomly generated MDP object.
  Its description is close to that of garnet problems (Bhatnagar et al. 2009).
  This class is specially designed to do various off-policy experiments
  '''

  def __init__(self, params):
    params['offpolicy'] = True
    self.b          = params['b']   # branching factor    
    self.rtype      = params['rtype'] # distribution used to generate expected reward 
    self.rparam     = params['rparam'] # parameter for rtype
    self.bpoltype   = params['bpoltype']
    self.tpoltype   = params['tpoltype']
    gm      = np.ones(params['ns'])*params['Gamma']
    np.random.seed(params['mdpseed'])
    gm[np.random.choice(params['ns'], params['numzerogs'], replace=False)] = 0.0
    Gamma             = np.diag(gm)
    params['Gamma']   = Gamma
    mdp.MDP.__init__(self, params)
    
  def getPssa(self):
    Pssa = np.zeros((self.ns, self.ns, self.na))
    for a in range(self.na):
      for s in range(self.ns):
        nextss = self.rdmdp.choice(range(self.ns), self.b, replace=False)
        nextscutpoints = np.sort(self.rdmdp.uniform(0,1,self.b-1))
        nextscutpoints = np.concatenate(([0], nextscutpoints, [1]))
        Pssa[s, nextss, a] = np.diff(nextscutpoints)
    return Pssa
    
  def getRssa(self):
    if self.rtype=='uniform':
      Rssa = np.reshape(self.rdmdp.uniform(0, self.rparam, self.ns*self.ns*self.na), 
                                  (self.ns, self.ns, self.na))
    if self.rtype=='normal':
      Rssa = np.reshape(self.rdmdp.normal(0, self.rparam, self.ns*self.ns*self.na), 
                                  (self.ns, self.ns, self.na))
    return Rssa
  
  def getBPolicy(self):
    return self.getPolicy(self.bpoltype)
  
  def getTPolicy(self):
    return self.getPolicy(self.tpoltype) 
    
  def getPolicy(self, poltype):
    policies = {
      'random':mdp.getRandomlySampledPolicy(self.ns, self.na, self.rdmdp, coverage=True),
      'uniform'     :mdp.getUniformRandomPolicy(self.ns, self.na),
      'skewed'      :mdp.getSkewedPolicy1(self.ns, self.na, self.rdmdp)
                }
    return policies[poltype] 
    
  def getPhi(self, ftype, ns, nf=None, rndobj=None):
    Phi = 0.0
    if ftype=='tabular':
      Phi = np.eye(ns)
    elif ftype=='binary':
      nf = int(np.ceil(np.log(ns+1)/np.log(2)))
      Phi = np.zeros((ns, nf))
      for i in range(ns):
        for j in range(nf):
          Phi[i, nf-j-1] = ((i+1)>>j) & 1
        a = sum(Phi[i,]*Phi[i,])
        Phi[i,] = Phi[i,]/np.sqrt(a)
    elif ftype=='nbinary':
      nf = int(np.ceil(np.log(ns+1)/np.log(2)))
      Phi = np.zeros((ns, nf))
      for i in range(ns):
        for j in range(nf):
          Phi[i, nf-j-1] = ((i+1)>>j) & 1
    elif ftype=='normal':
      Phi = np.zeros((ns, nf))
      for i in range(ns):
        for j in range(nf):
          Phi[i, j] = rndobj.normal(0, 1)
        a = sum(Phi[i,]*Phi[i,])
        Phi[i,] = Phi[i,]/np.sqrt(a)
    elif ftype=='nnormal':
      Phi = np.zeros((ns, nf))
      for i in range(ns):
        for j in range(nf):
          Phi[i, j] = rndobj.normal(0, 1)
    return Phi    
     