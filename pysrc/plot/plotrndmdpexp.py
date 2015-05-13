'''
Created on May 12, 2015

@author: A. Rupam Mahmood
'''

import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as ppl
from pysrc.plot import plotdataprocess 
import cPickle as pickle

def plotperfvslmbda(pathfileprefix, label):
  plotfilename      = pathfileprefix+"perfvslmbda.plot"
  if not os.path.isfile(plotfilename):
    sys.argv  = ["", "10", pathfileprefix, 
                 "2", "alpha", "lmbda",
                 "1", "lmbda"]
    plotdataprocess.main()
  plotfile    = file(plotfilename, "rb")
  data        = pickle.load(plotfile)
  ppl.errorbar(data[:,0], data[:,1], data[:,2], label=label)

def main():
  path                = "./results/totd-rndmdp-experiments/small/"
  if not os.path.exists(path):
    path = "../."+path
  pathfileprefix      = path+"td/mdpseed_1000_ftype_tabular_runseed_"
  plotperfvslmbda(pathfileprefix, "TD")
  pathfileprefix      = path+"tdr/mdpseed_1000_ftype_tabular_runseed_"
  plotperfvslmbda(pathfileprefix, "TDR")
  pathfileprefix      = path+"totd/mdpseed_1000_ftype_tabular_runseed_"
  plotperfvslmbda(pathfileprefix, "TOTD")
  ppl.legend()

if __name__ == '__main__':
  main()
  ppl.show()
  