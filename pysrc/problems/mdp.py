'''
Created on May 2, 2014

@author: A. Rupam Mahmood

'''

import numpy as np
import pylab as pl
import random
import matplotlib as mpl

class MDP(object):
  '''
  It creates a generic MDP object.
  Requires Pssa, Rssa and other structures to be specified.
  '''
  def __init__(self, params):
    '''
    Create all the necessary structure for the MDP
    '''
    self.ns         = params['ns']
    self.na         = params['na']
    self.ftype      = params['ftype'] # feature type
    if self.ftype=='tabular':   self.nf = self.ns  
    elif self.ftype=='binary':  self.nf = int(np.ceil(np.log(self.ns+1)/np.log(2)))
    else: self.nf = params['nf']

    self.Rstd       = params['Rstd'] # std dev. for generating random rewards
    self.initsdist  = params['initsdist'] # initial state distribution
    self.Gamma      = params['Gamma'] # diagonal matrix of state-dependent gammas
    self.mdpseed    = params['mdpseed'] # seed to generate mdps
    self.rdmdp      = np.random.RandomState(self.mdpseed)

    if type(self.initsdist)==str: 
      self.initsdistname = self.initsdist
      if self.initsdistname=='statemiddle':
        self.initsdist= np.zeros(self.ns)
        self.initsdist[(self.ns-1)/2] = 1.
    else:
      self.initsdistname = None
    
    self.Pssa       = self.getPssa()
    self.Rssa       = self.getRssa()
    self.bpol       = self.getBPolicy()
    (self.Pssb, self.exprb) = getPolInducedModel(self.Pssa, self.Rssa, self.bpol)
    if 'offpolicy' in params and params['offpolicy']==True:
      self.offpolicy  = True
      self.tpol       = self.getTPolicy()
      (self.Psst, self.exprt) = getPolInducedModel(self.Pssa, self.Rssa, self.tpol)
    else:
      self.offpolicy  = False
      self.tpol       = self.bpol
      (self.Psst, self.exprt) = (self.Pssb, self.exprb)
    
    self.Phi                = self.getPhi(self.ftype, self.ns, self.nf, rndobj=self.rdmdp)
    self.dsb                = self.steadystateprob(self.Pssb)
    if self.initsdist=='steadystate': self.initsdist = self.dsb

  def getPssa(self, params):
    raise NotImplementedError

  def getRssa(self, params):
    raise NotImplementedError

  def getBPolicy(self):
    raise NotImplementedError

  def getTPolicy(self):
    raise NotImplementedError

  def initTrajectory(self, runseed):
    self.rdrun    = np.random.RandomState(runseed)
    if self.initsdistname=='statezero':
      self.s  = 0
    elif self.initsdistname=='statemiddle':
      self.s  = (self.ns-1)/2
    else:
      self.s  = sampleFrom(self.initsdist, self.rdrun)

  def step(self):
    a = 0 if self.na==1 else self.getAction(self.bpol, self.s, self.rdrun)
    snext   = self.getNextState(self.Pssa, self.s, a, self.rdrun)
    noise   = self.rdrun.normal(0, self.Rstd) if self.Rstd>0 else 0.
    R       = self.getReward(self.Rssa, self.s, snext, a) + noise
    phi     = self.Phi[self.s]
    phinext = self.Phi[snext]
    if len(np.shape(self.Gamma))==1:
      a=0
    g       = self.Gamma[self.s, self.s]
    gnext   = self.Gamma[snext, snext]
    stemp   = self.s
    self.s  = snext
    return {'s':stemp, 'phi':phi, 'act':a, \
            'R':R, 'snext':snext, 'phinext':phinext, \
            'g':g, 'gnext':gnext}

  def getRho(self, s, a):
    return self.tpol[s,a]/self.bpol[s,a]

  def isTerminal(self):
    return self.Gamma[self.s, self.s] == 0.0
  
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
    elif ftype=='normal':
      Phi = np.zeros((ns, nf))
      for i in range(ns):
        for j in range(nf):
          Phi[i, j] = rndobj.normal(0, 1)
        a = sum(Phi[i,]*Phi[i,])
        Phi[i,] = Phi[i,]/np.sqrt(a)
    return Phi    

  def getAction(self, pol, s, rndobj):
    rndnum = rndobj.uniform(0, 1)
    return mpl.mlab.find(rndnum<np.cumsum(pol[s,:]))[0]
  
  def getNextState(self, Pssa, s, a, rndobj):
    rndnum = rndobj.uniform(0, 1)
    return mpl.mlab.find(rndnum<np.cumsum(Pssa[s,:,a]))[0]
  
  def getReward(self, Rssa, s, snext, a):
    return Rssa[s,snext,a]

  @staticmethod
  def getFixedPoint(Pss, ExpR, Phi, ds, Gamma, Lmbda):
    (ns, ns) = np.shape(Pss)
    D = np.diag(ds)
    ImPGLinv = pl.inv(np.eye(ns)- np.dot(Pss, np.dot(Gamma, Lmbda)))
    PhiTD = np.dot(Phi.T, D)
    ImPG = np.eye(ns) - np.dot(Pss, Gamma)
    A = np.dot(np.dot(PhiTD, ImPGLinv), np.dot(ImPG, Phi))
    b = np.dot(PhiTD, np.dot(ImPGLinv, ExpR))
    thstar = np.dot(pl.inv(A), b)
    return thstar

  def steadystateprob(self, Pss):
    (eigvals, eigvecs) = pl.eig(Pss.T)
    eigi = np.argmax(eigvals)
    diotas = eigvecs[:,eigi]/sum(eigvecs[:,eigi])
    return np.real(diotas)

## general ergodic MDP, general gamma 
## Pssa is of s X s' X a form

def getRandomMDP(ns, na, rndobj, rtype='uniform', rparam=1):
  Pssa = np.reshape(rndobj.uniform(0, 1, ns*ns*na) + 10**-10 , \
                                (ns, ns, na))
  
  rsums = np.sum(Pssa, 1)
  Pssa = Pssa / rsums[:, None]
  if rtype=='uniform':
    Rssa = np.reshape(rndobj.uniform(0, rparam, ns*ns*na), 
                                (ns, ns, na))
  if rtype=='normal':
    Rssa = np.reshape(rndobj.normal(0, rparam, ns*ns*na), 
                                (ns, ns, na))
  return (Pssa, Rssa)

def getPolInducedModel(Pssa, Rssa, pol):
    Pss = getPss(Pssa, pol)
    Rsa = getRsa(Rssa, Pssa)  
    ExpR = getExpR(Rsa, pol)

    return (Pss, ExpR)
     
def followon(Pss, startstatedist):
  (ns, ns) = np.shape(Pss)
  return np.dot(pl.inv(np.eye(ns) - Pss.T), startstatedist)

def getPss(Pssa, pol):
  (ns, t1, nacts) = np.shape(Pssa)
  Pss = np.zeros((ns, ns)) 
  for i in range(ns):
    for j in range(ns):
      Pss[i, j] = np.dot(Pssa[i, j, ], pol[i, ])
  return Pss

def getRsa(Rssa, Pssa):
  (ns, t1, nacts) = np.shape(Rssa)
  Rsa = np.zeros((ns, nacts)) 
  for i in range(ns):
    for j in range(nacts):
        Rsa[i, j] = np.dot(Pssa[i,:,j], Rssa[i,:,j])
  return Rsa

def getExpR(Rsa, pol):
  (ns, t1) = np.shape(Rsa)
  ExpR = np.zeros(ns)
  for i in range(ns): ExpR[i] = np.dot(pol[i,], Rsa[i,])
  return ExpR

def getRandomlySampledPolicy(ns, na, rndobj, coverage=True):
  pol = np.reshape(np.array([rndobj.uniform(0, 1) + coverage*10**-15 for i in range(ns*na)]), (ns, na) )
  pol = np.array([ pol[i]/sum(pol[i]) for i in range(ns) ])
  return pol

def getUniformRandomPolicy(ns, na):
  pol = np.ones((ns, na))/na
  return pol

def getSkewedPolicy1(ns, na, rndobj):
  pol = np.zeros((ns, na))
  for i in range(ns):
    a = rndobj.randint(0, na)
    if na>1: pol[i,:] = 0.01/(na-1)
    pol[i, a] = 0.99
  return pol
    
def sampleFrom(dist, rndobj):
  rndnum = rndobj.uniform(0, 1)
  return mpl.mlab.find(rndnum<np.cumsum(dist))[0]

def getPhiIndices(Phi):
  (ns, nf) = np.shape(Phi)
  PhiIndices = [[]]*ns
  for s in range(ns):
    PhiIndices[s] = mpl.mlab.find(Phi[s]!=0)
  return PhiIndices
  
class PerformanceMeasure(object):
  '''
  This class takes the responsibility of measuring performance of an estimate.
  Only for on-policy case at this point.
  '''
  
  def __init__(self, params, prob):
    self.T            = params['T'] # number of total steps
    self.N            = params['N'] # number of data points to store
    self.Phi          = prob.Phi
    self.prob         = prob
    if 'offpolicy' in params and params['offpolicy']==True:
      self.Pss             = prob.Psst
      self.expr            = prob.exprt
    else:
      self.Pss             = prob.Pssb
      self.expr            = prob.exprb      
    self.thstar       = prob.getFixedPoint(self.Pss, self.expr, prob.Phi, prob.dsb, prob.Gamma, 1.)
    self.VTrueProj    = np.dot(prob.Phi, self.thstar)
    self.D            = np.diag(prob.dsb)
    self.normFactor   = np.dot(self.VTrueProj, np.dot(self.D, self.VTrueProj))
    self.MSPVE          = np.zeros(self.N)

  def getThstarMSPBE(self, Lmbda):
    return self.prob.getFixedPoint(self.Pss, self.expr, self.Phi, self.prob.dsb, self.prob.Gamma, Lmbda)
  
  def calcMSPVE(self, alg, t): # t is the number of the current step
    index               = t*self.N/self.T 
    msediff             = np.dot(self.Phi, alg.estimate()) - self.VTrueProj
    self.MSPVE[index]  += np.dot( np.dot(msediff, self.D), msediff)
    if np.mod(t+1, self.T/self.N)==0:
      self.MSPVE[index]  /= self.T/self.N
  
  def getNormMSPVE(self):
    return self.MSPVE/self.normFactor
  
  
