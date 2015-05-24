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
  path = "./results/usage-experiments/stdrw-sparse-reward-11-states/"+algname+"/"
  if not os.path.exists(path):
    path = "../."+path
  pathfileprefix      = path+"run_"
  if not os.path.isfile(pathfileprefix+"perfvslambda.plot"):
    sys.argv  = ["", "30", pathfileprefix]
    sys.argv.extend(params)
    plotdataprocess.main()
  oisdata   = pickle.load(file(pathfileprefix+"perfvslambda.plot"))
  ppl.errorbar(oisdata[:,0], oisdata[:,1], oisdata[:,2], label=algname)

def main():

  plotonealg("gtd", ["3", "alpha", "beta", "lambda", "1", "lambda"])
  plotonealg("togtd", ["3", "alpha", "beta", "lambda", "1", "lambda"])
  plotonealg("wtd", ["3", "eta", "initd", "lambda", "1", "lambda"])
  plotonealg("wgtd", ["4", "eta", "initd", "beta", "lambda", "1", "lambda"])
  plotonealg("wtogtd", ["4", "eta", "initd", "beta", "lambda", "1", "lambda"])
  plotonealg("oislstd", ["2", "inita", "lambda", "1", "lambda"])
  plotonealg("wislstd", ["2", "inita", "lambda", "1", "lambda"])
  plotonealg("olstd2", ["2", "inita", "lambda", "1", "lambda"])
  ppl.ylim([0, 0.2])
  ppl.legend()
  #ppl.savefig('tmp.png')
  
if __name__ == '__main__':
    main()
    ppl.show()
