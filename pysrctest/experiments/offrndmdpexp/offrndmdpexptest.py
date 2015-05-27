'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import os
import sys
from pysrc.experiments import offrndmdpexp
import cPickle as pickle

class Test(unittest.TestCase):
  def testStdRandomWalkExp(self):
    dirpath   = "./pysrctest/experiments/offrndmdpexp/"
    dirpath   = "" if not os.path.isdir(dirpath) else dirpath
    algname   = "gtd"
    sys.argv  = ["", "1000", "1", dirpath, algname]
    offrndmdpexp.main()

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testStdRandomWalkExp']
  unittest.main()
    