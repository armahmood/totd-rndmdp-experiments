'''
Created on May 12, 2015

@author: A. Rupam Mahmood
'''

import sys
import numpy as np
from pysrc.plot import plotdataprocess 
import cPickle as pickle

def main():
  path      = "../../results/totd-rndmdp-experiments/small/td"
  sys.argv  = ["", "10", path, 
               "2", "alpha", "lmbda",
               "1", "lmbda"]
  plotdataprocess.main()
  tddata    = pickle.load(file(path+"perfvslambda.plot"))

if __name__ == '__main__':
  main()
  