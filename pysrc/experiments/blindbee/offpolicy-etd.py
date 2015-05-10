'''
Created on Apr 28, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
from pysrc.problems import blindbee
from pysrc.algorithms.tdprediction import etd
from pysrc.algorithms.tdprediction import td
from matplotlib import pyplot as ppl
import copy
import pickle
import os

def onerun(config, ntraps):
  bpolicy            = config['bpolicy']  \
                    = blindbee.UpwardishPolicy(config)
  uppolicy          = blindbee.Upward2Policy(config)
  smartpolicy       = blindbee.Smart3Policy(config)
  uniformpolicy     = blindbee.UniformPolicy(config)
  state         = blindbee.BlindBeeState(config)
  #fet           = blindbee.Phi2(state.nrows)
  fet           = blindbee.Phi3()
  config['nf']  = fet.nf
  upalg       = etd.ETD(config)
  smartalg    = etd.ETD(config)
  uniformalg  = etd.ETD(config)
  valsup        = np.zeros(ntraps)
  valssmart     = np.zeros(ntraps)
  valsuniform   = np.zeros(ntraps)
  trap = 0
  I = 1
  while trap < ntraps:
    s = copy.copy(state)
    a = bpolicy.getAction(state)
    state.getNextState(a)
    probstep = {'a':a, 
      's':s, 
      'snext':state}
    probstep['l'] = config['lmbda']
    probstep['lnext'] = config['lmbda']
    probstep['g'] = config['gamma']
    if probstep['snext'].inTrap():
      probstep['gnext'] = 0.0
    else:
      probstep['gnext'] = config['gamma']
    probstep['R'] = state.updisplacement
    probstep['phi'] = fet.getPhi(probstep['s'])
    probstep['phinext'] = fet.getPhi(probstep['snext'])
    probstep['I'] = I #1*probstep['phi'][0]
    probstep['rho'] = uppolicy.getPolicy(s)[a] / bpolicy.getPolicy(s)[a]
    upalg.step(probstep)
    probstep['rho'] = smartpolicy.getPolicy(s)[a] / bpolicy.getPolicy(s)[a]
    smartalg.step(probstep)
    probstep['rho'] = uniformpolicy.getPolicy(s)[a] / bpolicy.getPolicy(s)[a]
    uniformalg.step(probstep)
    I = 0
    if state.inTrap():
      I = 1
      valsup[trap] = upalg.th[0]
      valssmart[trap] = smartalg.th[0]
      valsuniform[trap] = uniformalg.th[0]
      trap += 1
      upalg.initepisode()
      smartalg.initepisode()
      uniformalg.initepisode()
    state.checkState()
  
  return valsup, valssmart, valsuniform

def main1():
  runseed = 1
  rdrun = np.random.RandomState(runseed)
  config = {'rdrun':rdrun, 'nrows':10, 
    'ncols':10, 
    'trapprob':0.25, 
    'trapduralim':3, 
    'alpha':0.005, 
    'lmbda':0.99, 
    'gamma':1.}
  valsup, valssmart, valsuniform = onerun(config, 1000)
  ppl.plot(valsup)
  ppl.plot(valssmart)
  ppl.plot(valsuniform)
  print(valsup[-1])
  print(valssmart[-1])
  print(valsuniform[-1])
  ppl.show()

def main():
  ntraps      = 5000
  nruns       = 30
  valsup          = np.zeros((nruns, ntraps))
  valssmart       = np.zeros((nruns, ntraps))
  valsuniform     = np.zeros((nruns, ntraps))
  for run in range(nruns):
    rdrun = np.random.RandomState(run)
    config = {'rdrun':rdrun, 'nrows':5, 
      'ncols':5, 
      'trapprob':0.4, 
      'trapduralim':2, 
      'alpha':0.001, 
      'lmbda':0.99, 
      'gamma':1.}
    valsup[run], valssmart[run], valsuniform[run] = onerun(config, ntraps)
  ppl.plot(np.mean(valsup, 0))
  ppl.plot(np.mean(valssmart, 0))
  ppl.plot(np.mean(valsuniform, 0))
  ppl.show()
  filepathname    = "../../../results/ewrl-etd-results/nruns:"+str(nruns)+":ntraps:"+str(ntraps)
  pickle.dump(valsup, open(filepathname+":up.dat", "wb"))
  pickle.dump(valssmart, open(filepathname+":smart.dat", "wb"))
  pickle.dump(valsuniform, open(filepathname+":uniform.dat", "wb"))

if __name__ == '__main__':
  main()
  
  
  