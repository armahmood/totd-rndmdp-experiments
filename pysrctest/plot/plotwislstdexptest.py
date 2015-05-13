'''
Created on May 13, 2015

@author: A. Rupam Mahmood
'''
import unittest
from pysrc.plot import plotwislstdexp
  
class Test(unittest.TestCase):

  def testName(self):
    plotwislstdexp.main()    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()