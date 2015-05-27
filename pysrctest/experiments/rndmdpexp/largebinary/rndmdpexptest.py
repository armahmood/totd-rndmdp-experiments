'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import os
import sys
from pysrc.experiments import rndmdpexp
import cPickle as pickle

class Test(unittest.TestCase):
  def testStdRandomWalkExp(self):
    dirpath   = "./pysrctest/experiments/rndmdpexp/largebinary/"
    dirpath   = "" if not os.path.isdir(dirpath) else dirpath

    sys.argv  = ["", "1000", "binary", "1", dirpath]
    rndmdpexp.main()
    data        = pickle.load(open(dirpath+"mdpseed_1000_ftype_binary_runseed_1.dat", "rb"))
    groundtruth = pickle.load(open(dirpath+"exp1_TDR__mdpseed_1000_runseed_1_conf_config_binary_large.dat_alphaindex_0_lmbdaindex_0", "rb"))
    print data['error']
    print groundtruth['TDRMSPVE']
    assert(abs(groundtruth['TDRMSPVE']-data['error'])<0.005)

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testStdRandomWalkExp']
  unittest.main()
    