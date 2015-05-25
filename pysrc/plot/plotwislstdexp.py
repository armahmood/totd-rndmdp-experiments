'''
Created on Apr 8, 2015

@author: A. Rupam Mahmood
'''

import os
import sys
from pysrc.plot import plotdataprocess
import matplotlib.pyplot as ppl
import pickle

def main():
  path = "./results/wislstd-experiments/stdrw-sparse-reward-states/oislstd/"
  if not os.path.exists(path):
    path = "../."+path
  pathfileprefix      = path+"run_"
  sys.argv  = ["", "5", pathfileprefix, \
              "2", "inita", "lambda", \
              "1", "lambda"]
  plotdataprocess.main()
  oisdata   = pickle.load(file(pathfileprefix+"perfvslambda.plot"))

  path = "./results/wislstd-experiments/stdrw-sparse-reward-states/wislstd/"
  if not os.path.exists(path):
    path = "../."+path
  pathfileprefix      = path+"run_"
  sys.argv  = ["", "5", pathfileprefix, \
              "2", "inita", "lambda", \
              "1", "lambda"]
  plotdataprocess.main()
  wisdata   = pickle.load(file(pathfileprefix+"perfvslambda.plot"))
  
  ppl.errorbar(oisdata[:,0], oisdata[:,1], oisdata[:,2], label="OIS")
  ppl.errorbar(wisdata[:,0], wisdata[:,1], wisdata[:,2], label="WIS")
  ppl.legend()
  
if __name__ == '__main__':
    main()
    ppl.show()
