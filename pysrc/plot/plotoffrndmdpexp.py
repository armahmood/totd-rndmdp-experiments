'''
Created on Apr 8, 2015

@author: A. Rupam Mahmood
'''

import os
import sys
sys.path.insert(0, os.getcwd())
from pysrc.plot import plotdataprocess
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as ppl
import pickle

def plotonealg(algname, params):
  path = "./results/offpolicy-rndmdp-experiments/state-10-bpol-random-tpol-skewed-ftype-binary/"+algname+"/"
  if not os.path.exists(path):
    path = "../."+path
  pathfileprefix      = path+"mdpseed_1000_runseed_"
  #if not os.path.isfile(pathfileprefix+"perfvslmbda.plot"):
  sys.argv  = ["", "10", pathfileprefix]
  sys.argv.extend(params)
  plotdataprocess.main()
  oisdata   = pickle.load(file(pathfileprefix+"perfvslmbda.plot"))
  ppl.errorbar(oisdata[:,0], oisdata[:,1], oisdata[:,2], label=algname)

def main():

  plotonealg("gtd", ["3", "alpha", "beta", "lmbda", "1", "lmbda"])
  plotonealg("togtd", ["3", "alpha", "beta", "lmbda", "1", "lmbda"])
  plotonealg("wtd", ["3", "eta", "initd", "lmbda", "1", "lmbda"])
  plotonealg("wgtd", ["4", "eta", "initd", "beta", "lmbda", "1", "lmbda"])
  plotonealg("wtogtd", ["4", "eta", "initd", "beta", "lmbda", "1", "lmbda"])
  plotonealg("oislstd", ["2", "inita", "lmbda", "1", "lmbda"])
  plotonealg("wislstd", ["2", "inita", "lmbda", "1", "lmbda"])
  plotonealg("olstd2", ["2", "inita", "lmbda", "1", "lmbda"])
  #ppl.ylim([0, 0.2])
  ppl.yscale('log')
  ppl.legend()
  #ppl.savefig('tmp.png')
  
if __name__ == '__main__':
    main()
    ppl.show()
