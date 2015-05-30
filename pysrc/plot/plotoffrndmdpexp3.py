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
import numpy as np

def plotonealg(algname, params):
  path = "./results/offpolicy-rndmdp-experiments/state-10-bpol-random-tpol-skewed-ftype-binary/"+algname+"/"
  if not os.path.exists(path):
    path = "../."+path
  pathfileprefix      = path+"mdpseed_1000_runseed_"
  nruns       = 5
  data        = plotdataprocess.loaddata(nruns, pathfileprefix)
  neps        = data[0]['N'] # number of data points
  table       = plotdataprocess.createtable(data, params, neps)
  (avg, std)  = plotdataprocess.createtablelearningcurves(table, nruns, neps)
  index       = 50
  print np.mean(avg[index][(len(params)):])   
  print  avg[index][:len(params)]
  ppl.plot(range(len(avg[index][len(params):])), avg[index][len(params):], label=algname)

def main():

  plotonealg("gtd", ["alpha", "beta", "lmbda"])
#   plotonealg("togtd", ["3", "alpha", "beta", "lmbda", "1", "alpha"])
#   plotonealg("wtd", ["eta", "initd", "lmbda"])
#   plotonealg("wgtd", ["4", "eta", "initd", "beta", "lmbda", "1", "initd"])
#   plotonealg("wtogtd", ["4", "eta", "initd", "beta", "lmbda", "1", "initd"])
#   plotonealg("oislstd", ["2", "inita", "lmbda", "1", "inita"])
#   plotonealg("wislstd", ["inita", "lmbda"])
#   plotonealg("olstd2", ["2", "inita", "lmbda", "1", "inita"])
  #ppl.ylim([0, 0.2])
  ppl.yscale('log')
  #ppl.xscale('log')
  ppl.legend()
  #ppl.savefig('tmp.png')
  
if __name__ == '__main__':
    main()
    ppl.show()
