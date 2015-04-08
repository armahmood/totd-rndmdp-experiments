'''
Created on Mar 25, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import sys
import pysrc.experiments.stdrwexp as stdrwexp

class Test(unittest.TestCase):

  def testStdRandomWalkExp(self):
    sys.argv = ["", "1", "StdRWSparseReward", "./pysrctest/experiments/stdrwexp/"]
    stdrwexp.main()
    sys.argv = ["", "1", "StdRWFreqReward", "./pysrctest/experiments/stdrwexp/"]
    stdrwexp.main()

if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.testStdRandomWalkExp']
  unittest.main()
    