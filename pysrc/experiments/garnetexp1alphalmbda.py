'''
Created on Sep 30, 2014

@author: A. Rupam Mahmood
'''

import os
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd())) # to run it from pysrc/ folder
import numpy as np
import pickle
import pysrc.experiments.rndmdpexp as garnetexp1

def main():
  mdpseed = int(sys.argv[1])
  runseed = int(sys.argv[2])
  savepath = sys.argv[3]
  configfilename = sys.argv[4]
  configfilepathname = savepath + configfilename
  nalgs = int(sys.argv[5])
  algs = np.array([sys.argv[6+i] for i in range(nalgs)])
  config = pickle.load(open(configfilepathname, 'rb'))
  alphas = config['alphas']
  lmbdas = config['lmbdas']
  nargv = 7+nalgs
  temp = list(sys.argv)
  sys.argv = [None]*nargv
  sys.argv[0:(5+nalgs)] = temp
  for alphaindex in range(len(alphas)):
    for lmbdaindex in range(len(lmbdas)):
      sys.argv[5] = str(alphaindex)
      sys.argv[6] = str(lmbdaindex)
      sys.argv[7] = str(nalgs)
      for i in range(nalgs): sys.argv[8+i] = algs[i] 
      garnetexp1.main()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    