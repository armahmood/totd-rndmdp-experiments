'''
Created on May 23, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
import pylab as pl
import cPickle as pickle

def main():
  path = "../../results/usage-experiments/stdrw-sparse-reward-11-states/gtd/"
  data = pickle.load(file(path+"run_30.dat", "rb"))
  print data

if __name__ == '__main__':
    main()