'''
Created on Apr 8, 2015

@author: A. Rupam Mahmood
'''

import sys
from pysrc.plot import plotdataprocess
import matplotlib.pyplot as ppl
import pickle

def main():
  path      = "../../results/wislstdexperiments/oislstd/" 
  sys.argv  = ["", "10", path, \
              "2", "inita", "lambda", \
              "1", "lambda"]
  plotdataprocess.main()
  oisdata   = pickle.load(file(path+"perfvslambda.plot"))
  path      = "../../results/wislstdexperiments/wislstd/" 
  sys.argv  = ["", "10", "../../results/wislstdexperiments/wislstd/", \
              "2", "inita", "lambda", \
              "1", "lambda"]
  plotdataprocess.main()
  wisdata   = pickle.load(file(path+"perfvslambda.plot"))
  
  ppl.errorbar(oisdata[:,0], oisdata[:,1], oisdata[:,2], label="OIS")
  ppl.errorbar(wisdata[:,0], wisdata[:,1], wisdata[:,2], label="WIS")
  ppl.legend()
  
if __name__ == '__main__':
    main()
    ppl.show()
