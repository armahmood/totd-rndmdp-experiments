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
    dirpath   = "./pysrctest/experiments/rndmdpexp/smalltabular/"
    dirpath   = "" if not os.path.isdir(dirpath) else dirpath
    sys.argv  = ["", "1000", "tabular", "1", dirpath]
    rndmdpexp.main()
    data        = pickle.load(open(dirpath+"mdpseed_1000_ftype_tabular_runseed_1.dat", "rb"))
    groundtruth = pickle.load(open(dirpath+"exp1_TD__mdpseed_1000_runseed_1_conf_config_tabular.dat_alphaindex_0_lmbdaindex_0", "rb"))
    print data
    print groundtruth
    assert(abs(groundtruth['TDMSPVE']-data['NMSPVE'])<10**-10)

    sys.argv  = ["", "1000", "tabular", "2", dirpath]
    rndmdpexp.main()
    data        = pickle.load(open(dirpath+"mdpseed_1000_ftype_tabular_runseed_2.dat", "rb"))
    groundtruth = pickle.load(open(dirpath+"exp1_TD__mdpseed_1000_runseed_2_conf_config_tabular.dat_alphaindex_0_lmbdaindex_0", "rb"))
    print data
    print groundtruth
    assert(abs(groundtruth['TDMSPVE']-data['NMSPVE'])<10**-10)

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testStdRandomWalkExp']
  unittest.main()
    