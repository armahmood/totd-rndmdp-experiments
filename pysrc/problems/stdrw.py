'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood

'''

import numpy as np
import pylab as pl
import random
from pysrc.problems.mdp import MDP

class StdRandomWalk(object):
  '''
  Standard random walk with a line of states and two
  terminal states on both sides. This provides the
  generic problem where the reward function is taken 
  as an input. Problems with specific reward function
  are created separately by instantiating this class. 
  '''

  def __init__(self, params):
    '''
      Creates all the necessary structure for the
      random walk problem.
    '''
    self.ftype    = params['ftype']
    self.ns       = params['ns']
    self.inits    = params['inits']
    self.mright   = params['mright']  # prob of going right under behavior policy
    if self.ftype=='normal' or self.ftype=='nnormal':        
      self.rdprob = random.Random(params['randseed'])
    self.nf       = params['nf']
    if 'bandsize' in params: self.bandsize = params['bandsize']
    self.g        = params['gamma']
    self.Rmat     = self.getRmat(self.ns)
    self.mpol     = self.getpol(self.ns, params['mright'])
    self.ppol     = self.getpol(self.ns, params['pright'])

    self.s = self.inits

  def setrunseed(self, runseed):
    self.rdrun = random.Random(runseed)
          
  def initepisode(self):
    self.s = self.inits
    
  def step(self):
    rnd     = self.rdrun.uniform(0, 1)
    act     = 0 if rnd < 1-self.mright else +1
    snext   = self.s + act*2-1
    R       = self.Rmat[self.s, snext]
    f       = self.getfeatures(self.s)
    fnext   = self.getfeatures(snext)
    stemp   = self.s
    self.s  = snext
    gnext   = 0 if self.isterminal() else self.g
    return {'s':stemp, 'phi':f, 'act':act, \
            'R':R, 'snext':snext, 'phinext':fnext, \
            'g':self.g, 'gnext':gnext}

  def getfeatures(self, s):
    if self.ftype=='tabular':
      Phi = StdRandomWalk.getPhi(self.ftype, self.ns)
      return Phi[s,]
    if self.ftype=='compressed1':
      Phi = StdRandomWalk.getPhi(self.ftype, self.ns, self.nf, self.bandsize)
      return Phi[s,]
    if self.ftype=='compressed2':
      Phi = StdRandomWalk.getPhi(self.ftype, self.ns)
      return Phi[s,]
    if self.ftype=='binary':
      Phi = StdRandomWalk.getPhi(self.ftype, self.ns)
      return Phi[s,]
    if self.ftype=='normal':
      Phi = StdRandomWalk.getPhi(self.ftype, self.ns, nf=self.nf, randobj=self.rdprob)
      return Phi[s,]
    if self.ftype=='nnormal':
      Phi = StdRandomWalk.getPhi(self.ftype, self.ns, nf=self.nf, randobj=self.rdprob)
      return Phi[s,]
    
  def isterminal(self):
    return True if self.s==0 or self.s==self.ns-1 else False

  def getRho(self, s, a):
    return self.ppol[s,a]/self.mpol[s,a]

  @staticmethod
  def getpol(ns, mright):
    pol = np.zeros((ns,2))
    pol[1:ns-1,0] = 1-mright
    pol[1:ns-1,1] = mright
    return pol
    
  @staticmethod
  def getPsa(ns):
    Psa = np.zeros((ns, ns, 2));
    Psa[0,0,0] = Psa[0,0,1] = Psa[ns-1,ns-1,0] \
      = Psa[ns-1,ns-1,1] = 0.5
    Psa[tuple(range(1,ns)), tuple(range(0,ns-1)), 0] = 1
    Psa[tuple(range(0,ns-1)), tuple(range(1,ns)), 1] = 1
    return Psa

  @staticmethod
  def getP(ns, pol, Psa):
    P = np.zeros((ns,ns)); P[ns-1,ns-1] = P[0,0] = 1
    for i in range(1, ns-1):
        for j in range(0, ns):
            P[i, j] = Psa[i, j,0]*pol[i,0] + Psa[i, j,1]*pol[i,1]
    return P

  @staticmethod #
  def getRmat(ns):
    raise NotImplementedError
  
  @staticmethod
  def getR(ns, Rmat, P):
    R = np.zeros(ns)
    for i in range(ns): R[i] = np.dot(P[i,], Rmat[i,])
    return R

  @staticmethod
  def getinitstateprob(ns, inits):
    initstateprob = np.zeros(ns-2); initstateprob[inits-1] = 1
    return initstateprob

  @staticmethod
  def getD(ns, P, initstateprob):
    Q = P[1:ns-1, 1:ns-1]
    mu = np.dot(pl.inv(np.eye(ns-2)-Q.T), pl.matrix(initstateprob).T)
    mu = mu/sum(mu)
    D = np.zeros((ns,ns))
    D[1:ns-1,1:ns-1] = np.diag(np.array(mu.T)[0])
    return D

  @staticmethod
  def getPhi(ftype, ns, nf=None, bandsize=None, randobj=None):
    if ftype=='tabular':
      Phi = np.zeros((ns, ns-2))
      Phi[1:ns-1,0:ns-2] = np.eye(ns-2)
      return Phi
    if ftype=='compressed1': # sort of normalized binary representation
      Phi = np.zeros((ns,nf))
      Phitemp = np.zeros((ns-2,nf))
      for i in range(ns-2):
        for j in range(nf):
          Phitemp[i,j] = 0 if i-j>bandsize-1 or j-i>0 else 1
        a = sum(Phitemp[i,]*Phitemp[i,])
        Phitemp[i,] = Phitemp[i,]/np.sqrt(a)
      Phi[1:(ns-1), ] = Phitemp
      return Phi
    if ftype=='compressed2': # state aggregation, three features, left, middle, right 
      Phi = np.zeros((ns,3))
      Phi[1:(ns-1)/2,] = np.array([1,0,0])
      Phi[(ns-1)/2,] = np.array([0,1,0])
      Phi[((ns-1)/2+1):-1,] = np.array([0,0,1])
      return Phi
    if ftype=='binary':
      nf = int(np.ceil(np.log(ns-1)/np.log(2)))
      Phi = np.zeros((ns, nf))
      for i in range(1,ns-1):
        for j in range(nf):
          Phi[i, nf-j-1] = (i>>j) & 1
      return Phi
    if ftype=='nbinary':
      nf = int(np.ceil(np.log(ns-1)/np.log(2)))
      Phi = np.zeros((ns, nf))
      for i in range(1,ns-1):
        for j in range(nf):
          Phi[i, nf-j-1] = (i>>j) & 1
        a = sum(Phi[i,]*Phi[i,])
        Phi[i,] = Phi[i,]/np.sqrt(a)
      return Phi
    if ftype=='normal':
      Phi = np.zeros((ns, nf))
      for i in range(1,ns-1):
        for j in range(nf):
          Phi[i, j] = randobj.gauss(0, 1)
      return Phi    
    if ftype=='nnormal':
      Phi = np.zeros((ns, nf))
      for i in range(1,ns-1):
        for j in range(nf):
          Phi[i, j] = randobj.gauss(0, 1)
        a = sum(Phi[i,]*Phi[i,])
        Phi[i,] = Phi[i,]/np.sqrt(a)
      return Phi    
    
  @staticmethod
  def fixedPointLambda(ns, g, l, Phi, P, R, D):
    m = np.eye(ns-2)-g*l*P[1:ns-1,1:ns-1]
    
    A = np.dot(np.dot(pl.matrix(Phi[1:ns-1,:]).T, np.dot(D[1:ns-1,1:ns-1], pl.inv(m))), \
                          np.dot((np.eye(ns-2)-g*P[1:ns-1,1:ns-1]),Phi[1:ns-1,:]))
    
    b = np.dot(np.dot(pl.matrix(Phi[1:ns-1,:]).T, \
                          np.dot(D[1:ns-1,1:ns-1], pl.inv(m))), pl.matrix(R[1:ns-1]).T)
    return (np.dot(pl.pinv(A), b), np.array(A), b)
  
  @classmethod
  def getthstarVTrue(cls, ftype, ns, g, l, inits, mright, pright, 
                     nf=None, bandsize=None, randobj=None):
    Phi         =cls.getPhi(ftype, ns, nf, bandsize, randobj=randobj)
    initsprob   =cls.getinitstateprob(ns, inits)
    ppol        =cls.getpol(ns, pright)
    mpol        =cls.getpol(ns, mright)
    Psa         =cls.getPsa(ns)
    Pp          =cls.getP(ns, ppol, Psa)
    Pm          =cls.getP(ns, mpol, Psa)
    Dm          =cls.getD(ns, Pm, initsprob)
    Rmat        =cls.getRmat(ns)
    R           =cls.getR(ns, Rmat, Pp)
    (thstar,A,b) =cls.fixedPointLambda(ns, g, l, Phi, Pp, R, Dm)
    VTrue = np.zeros(ns)
    VTrue[1:ns-1] = np.dot( pl.pinv(np.eye(ns-2) - g*Pp[1:ns-1,1:ns-1]), R[1:ns-1]).T
    return (thstar, VTrue)
      
class PerformanceMeasure(object):
  '''
  This class takes the responsibility of measuring performance
  of an estimate on the standard random walk problem
  '''
  
  def __init__(self, params, prob):
    self.Phi            = prob.getPhi(params['ftype'], params['ns'])
    self.thstarMSE, self.VTrue  = prob.getthstarVTrue(params['ftype'], params['ns'], 
                                               params['gamma'], 1, 
                                               params['inits'], params['mright'], 
                                               params['pright'])
    self.thstarMSPBE, self.VTrue  = prob.getthstarVTrue(params['ftype'], params['ns'], 
                                               params['gamma'], params['lambda'], 
                                               params['inits'], params['mright'], 
                                               params['pright'])
    self.VTrueProj      = np.dot(self.Phi, np.squeeze(np.array(self.thstarMSE)))
    self.Psa            = prob.getPsa(params['ns'])
    self.initstateprob  = prob.getinitstateprob(
                                                params['ns'], 
                                                params['inits'])
    self.mpol           = prob.getpol(params['ns'], 
                                     params['mright'])
    self.ppol           = prob.getpol(params['ns'], 
                                     params['pright'])
    self.Pm             = prob.getP(params['ns'], self.mpol, self.Psa)
    self.Dm             = prob.getD(params['ns'], self.Pm, self.initstateprob)
    self.MSPVE            = np.zeros(params['N'])

  def calcMse(self, alg, ep):
    msediff         = np.dot(self.Phi, alg.estimate()) - self.VTrueProj
    self.MSPVE[ep]    = np.dot( np.dot(msediff, self.Dm), msediff)
    
class StdRandomWalk2(MDP): # same problem implemented through MDP class
  def __init__(self, params):
    self.behavRight   = params['behavRight']
    self.targtRight   = params['targtRight']
    MDP.__init__(self,params)

  def initTrajectory(self, runseed):
    self.rdrun    = random.Random(runseed)
    self.s  = (self.ns-1)/2

  def getPhi(self, ftype, ns, nf=None, rndobj=None):
    nzG              = np.diag(self.Gamma)!=0.0
    if ftype=='tabular':
      Phi                   = np.zeros((ns, np.sum(nzG)))
      Phi[nzG, :] = np.eye(np.sum(nzG))
    elif ftype=='binary':
      nf = int(np.ceil(np.log(np.sum(nzG)+1)/np.log(2)))
      Phi   = np.zeros((ns, nf))
      _Phi  = Phi[nzG, :]
      for i in range(np.sum(nzG)):
        for j in range(nf):
          _Phi[i, nf-j-1] = ((i+1)>>j) & 1
      Phi[nzG,:] = _Phi
    elif ftype=='nbinary':
      nf = int(np.ceil(np.log(np.sum(nzG)+1)/np.log(2)))
      Phi   = np.zeros((ns, nf))
      _Phi  = Phi[nzG, :]
      for i in range(np.sum(nzG)):
        for j in range(nf):
          _Phi[i, nf-j-1] = ((i+1)>>j) & 1
        a = sum(_Phi[i,]*_Phi[i,])
        _Phi[i,] = _Phi[i,]/np.sqrt(a)
      Phi[nzG,:] = _Phi
    elif ftype=='normal':
      Phi = np.zeros((ns, nf))
      for i in range(ns):
        for j in range(nf):
          Phi[i, j] = rndobj.normal(0, 1)
      Phi[~nzG,:]       = 0.0
    elif ftype=='nnormal':
      Phi = np.zeros((ns, nf))
      for i in range(ns):
        for j in range(nf):
          Phi[i, j] = rndobj.normal(0, 1)
        a = sum(Phi[i,]*Phi[i,])
        Phi[i,] = Phi[i,]/np.sqrt(a)
      Phi[~nzG,:]       = 0.0
    return Phi    

  def getPssa(self):
    Pssa = np.zeros((self.ns, self.ns, 2));
    Pssa[0,0,0] = Pssa[0,0,1] = Pssa[self.ns-1,self.ns-1,0] \
      = Pssa[self.ns-1,self.ns-1,1] = 0.5
    Pssa[tuple(range(1,self.ns)), tuple(range(0,self.ns-1)), 0] = 1
    Pssa[tuple(range(0,self.ns-1)), tuple(range(1,self.ns)), 1] = 1
    self._setTermToInitTrans(Pssa)
    return Pssa

  def _setTermToInitTrans(self, Pssa):
    zeroGammas              = np.diag(self.Gamma)==0.0
    Pssa[zeroGammas, :, :]   = np.reshape(np.repeat(self.initsdist, 2), (self.ns, 2))

  def getBPolicy(self):
    pol = np.zeros((self.ns,2))
    pol[0:self.ns,0] = 1-self.behavRight
    pol[0:self.ns,1] = self.behavRight
    return pol
    
  def getTPolicy(self):
    pol = np.zeros((self.ns,2))
    pol[0:self.ns,0] = 1-self.targtRight
    pol[0:self.ns,1] = self.targtRight
    return pol
  
  def getNextState(self, Pssa, s, a, rndobj):
    nzG              = np.diag(self.Gamma)!=0.0
    return s + a*2-1 if nzG[s] else (self.ns-1)/2

  def getAction(self, pol, s, rndobj):
    nzG              = np.diag(self.Gamma)!=0.0
    if nzG[s]:
      rnd     = rndobj.uniform(0, 1)
      return 0 if rnd < 1-self.behavRight else +1 
    else:
      return 0

  def steadystateprob(self, Pss):
    ds          = MDP.steadystateprob(self, Pss)
    zG          = np.diag(self.Gamma)==0.0
    ds[zG]      = 0.0
    ds          = ds / np.sum(ds)
    return ds
    
  @staticmethod
  def getFixedPoint(Pss, ExpR, Phi, ds, Gamma, Lmbda):
    D = np.diag(ds)
    nzG                 = np.diag(Gamma)!=0.0
    ns                  = np.sum(nzG) 
    if np.isscalar(Lmbda): Lmbda = np.diag(np.ones(ns)*Lmbda)
    ImPGLinv = pl.inv(np.eye(ns)- np.dot(Pss[np.ix_(nzG,nzG)], np.dot(Gamma[np.ix_(nzG,nzG)], Lmbda)))
    PhiTD = np.dot(Phi[nzG,:].T, D[np.ix_(nzG,nzG)])
    ImPG = np.eye(ns) - np.dot(Pss[np.ix_(nzG,nzG)], Gamma[np.ix_(nzG,nzG)])
    A = np.dot(np.dot(PhiTD, ImPGLinv), np.dot(ImPG, Phi[nzG,:]))
    b = np.dot(PhiTD, np.dot(ImPGLinv, ExpR[nzG]))
    thstar = np.dot(pl.inv(A), b)
    return thstar

