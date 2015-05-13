'''
Created on May 12, 2015

@author: A. Rupam Mahmood
'''

import os
import sys
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
  pathfileprefix      = "../../results/totd-rndmdp-experiments/small/td/mdpseed_1000_ftype_tabular_runseed_"
  plotperfvslmbda(pathfileprefix, "TD")
  pathfileprefix      = "../../results/totd-rndmdp-experiments/small/tdr/mdpseed_1000_ftype_tabular_runseed_"
  plotperfvslmbda(pathfileprefix, "TDR")
  pathfileprefix      = "../../results/totd-rndmdp-experiments/small/totd/mdpseed_1000_ftype_tabular_runseed_"
  plotperfvslmbda(pathfileprefix, "TOTD")
  ppl.legend()

if __name__ == '__main__':
  main()
  ppl.show()
  