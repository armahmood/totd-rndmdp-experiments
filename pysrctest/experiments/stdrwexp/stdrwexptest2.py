'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import os
import sys
from pysrc.experiments import stdrwexp2

class Test(unittest.TestCase):

  def testStdRandomWalkExp(self):
    dirpath  = "./pysrctest/experiments/stdrwexp/"
    dirpath = "" if not os.path.isdir(dirpath) else dirpath
    sys.argv = ["", "1", "StdRWSparseReward2", dirpath]
    stdrwexp2.main()
    sys.argv = ["", "1", "StdRWFreqReward2", dirpath]
    stdrwexp2.main()
    sys.argv = ["", "1", "StdRWFreqPosNegReward2", dirpath]
    stdrwexp2.main()

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testStdRandomWalkExp']
  unittest.main()
    