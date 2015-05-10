'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import os
import sys
import pysrc.experiments.stdrwexp as stdrwexp

class Test(unittest.TestCase):

  def testStdRandomWalkExp(self):
    dirpath  = "./pysrctest/experiments/stdrwexp/"
    dirpath = "" if not os.path.isdir(dirpath) else dirpath
    print os.getcwd()
    sys.argv = ["", "1", "StdRWSparseReward", dirpath]
    stdrwexp.main()
    sys.argv = ["", "1", "StdRWFreqReward", dirpath]
    stdrwexp.main()

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testStdRandomWalkExp']
  unittest.main()
    