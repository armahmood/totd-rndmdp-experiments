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

def plotonealg(algname, nparams, params, nparamssub, paramssub):
  path = "./results/offpolicy-rndmdp-experiments/state-10-bpol-random-tpol-skewed-ftype-binary/"+algname+"/"
  if not os.path.exists(path):
    path = "../."+path
  pathfileprefix      = path+"mdpseed_1000_runseed_"
  plotfilesuffix      = "perfvs"+paramssub[-1]+".plot"
  nruns               = 5
  #if not os.path.isfile(pathfileprefix+plotfilesuffix):
  plotdataprocess.main2(nruns, pathfileprefix, nparams, params, nparamssub, paramssub, 0)
  oisdata   = pickle.load(file(pathfileprefix+plotfilesuffix))
  ppl.errorbar(oisdata[:,0], oisdata[:,1], oisdata[:,2], label=algname)

def main():

  plotonealg("gtd", 3, ["alpha", "beta", "lmbda"], 1, ["lmbda"])
#   plotonealg("togtd", 3, ["alpha", "beta", "lmbda"], 1, ["lmbda"])
#   plotonealg("wtd", 3, ["eta", "initd", "lmbda"], 1, ["lmbda"])
#   plotonealg("wgtd", 4, ["eta", "initd", "beta", "lmbda"], 1, ["lmbda"])
#   plotonealg("wtogtd", 4, ["eta", "initd", "beta", "lmbda"], 1, ["lmbda"])
#   plotonealg("oislstd", 2, ["inita", "lmbda"], 1, ["lmbda"])
#   plotonealg("wislstd", 2, ["inita", "lmbda"], 1, ["lmbda"])
#   plotonealg("olstd2", 2, ["inita", "lmbda"], 1, ["lmbda"])
  ppl.ylim([0.01, 10])
  ppl.yscale('log')
  #ppl.xscale('log')
  ppl.legend()
  #ppl.savefig('tmp.png')
  
if __name__ == '__main__':
    main()
    ppl.show()
