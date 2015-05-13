'''
Created on Mar 27, 2015

@author: A. Rupam Mahmood
'''
import unittest
import numpy as np
import pysrc.plot.plotdataprocess as plotutil
from pyparsing import alphas

class Test(unittest.TestCase):

  def createdummydata(self,alphas,betas,lmbdas,mse,runs):
    '''
    This method creates a dummy data object, which is a list 
    of configurations. In plotutil, it is obtained from different
    output files (based on different runs) and concatenated together. 
    This method can be used to avoid the need to have some already 
    created output files.
    '''
    data = [
               {'error'   : mse[alpha==alphas,beta==betas,lmbda==lmbdas]+run,
                'alpha' : alpha,
                'beta'   : beta,
                'lmbda'  : lmbda,
                'x1'     : 0.1}
               for run in runs
               for alpha in alphas
               for beta in betas
               for lmbda in lmbdas
               ]
    return data

  def tableexample1(self):
    '''
    This method creates a small table with 2 episodes 
    in a form created by createtable of plotutil.
     
    '''
    
    alphas  = np.array([0, 0.5, 1.0])
    betas   = np.array([0, 1.0])
    lmbdas  = np.array([0])
    mse     = np.array([
                        [[[0.11,0.09]], [[0.41,0.39]] ],
                        [[[0.21,0.19]], [[0.51,0.49]] ],
                        [[[0.31,0.29]], [[0.61,0.59]] ]
                        ]
                       )
    runs    = np.array([1,2,3])
    table   = np.array([ [alpha, beta, lmbda, 
                          mse[alpha==alphas,beta==betas,lmbda==lmbdas,0]+run,
                          mse[alpha==alphas,beta==betas,lmbda==lmbdas,1]+run  
                          ] 
                       for run in runs
                       for alpha in alphas
                       for beta in betas
                       for lmbda in lmbdas
                        ])
    nrows, ncols    = np.shape(table) 
    nruns           = len(runs)
    tableavgstd        = np.zeros((nrows/nruns, 5))
    tableavgstd[:,:3]  = table[:nrows/nruns,:3]
    i = 0
    for alpha in alphas:
      for beta in betas:
        for lmbda in lmbdas:
          temp = np.array([1+mse[alpha==alphas,beta==betas,lmbda==lmbdas],\
                      2+mse[alpha==alphas,beta==betas,lmbda==lmbdas],\
                      3+mse[alpha==alphas,beta==betas,lmbda==lmbdas],\
                      ])
          tableavgstd[i,3] = np.mean(temp)
          tableavgstd[i,4] = np.std(temp)/np.sqrt(2*nruns)
          i += 1
    return {'alphas':alphas, 'betas':betas, 
            'lmbdas':lmbdas, 'error':mse, 
            'runs':runs, 'table':table, 
            'tableavgstd':tableavgstd}


  def tableexample1explicit(self):
    tableavgstd2 = np.array([[0., 0., 0., 2.1, 0.33335833], 
        [0., 1.0, 0., 2.4, 0.33335833], 
        [0.5, 0., 0., 2.2, 0.33335833], 
        [0.5, 1.0, 0., 2.5, 0.33335833], 
        [1.0, 0., 0., 2.3, 0.33335833], 
        [1.0, 1.0, 0., 2.6, 0.33335833]])
    perfvsalpha       = np.array(
                           [[0,    2.1],
                            [0.5,  2.2],
                            [1.0,  2.3]]
                           )
    perfvsbeta        = np.array(
                           [[0,    2.1],
                            [1.0,  2.4]]
                           )
    perfvslmbda       = np.array([[0,   2.1]])
    perfvsalphabeta   = np.array([
                                 [0.,   0.,  2.1],
                                 [0.,   1.0, 2.4],
                                 [0.5,  0.,  2.2],
                                 [0.5,  1.0, 2.5],
                                 [1.0,  0.,  2.3],
                                 [1.0,  1.0, 2.6]]                                 
                                 )
    perfvsbetaalpha   = np.array([
                                 [0.,   0.,   2.1],
                                 [0.,   0.5,  2.2],
                                 [0.,   1.0,  2.3],
                                 [1.0,  0.0,  2.4],
                                 [1.0,  0.5,  2.5],
                                 [1.0,  1.0,  2.6]]                                 
                                 )
    return {'tableavgstd2':tableavgstd2, 'perfvsalpha':perfvsalpha,
            'perfvsbeta':perfvsbeta, 'perfvslmbda':perfvslmbda,
            'perfvsalphabeta':perfvsalphabeta, 'perfvsbetaalpha':perfvsbetaalpha}

  def testtableexample1(self):
    rets          = self.tableexample1()
    tableavgstd   = rets['tableavgstd']
    rets          = self.tableexample1explicit()
    tableavgstd2  = rets['tableavgstd2']
    assert((abs(tableavgstd-tableavgstd2)<0.000001).all())
  
  def testcreatetable(self):
    rets      = self.tableexample1()
    alphas=rets['alphas']; betas=rets['betas'];lmbdas=rets['lmbdas']
    mse=rets['error']; runs=rets['runs']; table=rets['table']
    data      = self.createdummydata(alphas,betas,lmbdas,mse,runs)
    params    = np.array(['alpha', 'beta', 'lmbda'])
    table2    = plotutil.createtable(data, params, 2)
    assert(((table-table2)==0.0).all())

  def testcreatetableavg(self):
    rets      = self.tableexample1()
    alphas=rets['alphas']; betas=rets['betas'];lmbdas=rets['lmbdas']
    mse=rets['error']; runs=rets['runs']; tableavgstd=rets['tableavgstd']
    params        = np.array(['alpha', 'beta', 'lmbda'])
    data          = self.createdummydata(alphas,betas,lmbdas,mse,runs)
    table         = plotutil.createtable(data, params, 2)
    tableavgstd2  = plotutil.createtableavg(table, len(runs), 2)
    assert(((tableavgstd-tableavgstd2)==0.0).all())
    
  def testperformancevsparams(self):
    rets      = self.tableexample1()
    alphas=rets['alphas']; betas=rets['betas'];lmbdas=rets['lmbdas']
    mse=rets['error']; runs=rets['runs']; tableavgstd=rets['tableavgstd']
    data          = self.createdummydata(alphas,betas,lmbdas,mse,runs)
    params        = np.array(['alpha', 'beta', 'lmbda'])
    table         = plotutil.createtable(data, params, 2)
    tableavgstd   = plotutil.createtableavg(table, len(runs), 2)
    perfvsalpha2  = plotutil.performancevsparams(tableavgstd, params, np.array(['alpha']))
    perfvsbeta2   = plotutil.performancevsparams(tableavgstd, params, np.array(['beta']))
    perfvslmbda2  = plotutil.performancevsparams(tableavgstd, params, np.array(['lmbda']))
    perfvsalphabeta2  = plotutil.performancevsparams(tableavgstd, params, np.array(['alpha', 'beta']))
    perfvsbetaalpha2  = plotutil.performancevsparams(tableavgstd, params, np.array(['beta', 'alpha']))
    rets            = self.tableexample1explicit()
    perfvsalpha     = rets['perfvsalpha']
    perfvsbeta      = rets['perfvsbeta']
    perfvslmbda     = rets['perfvslmbda']
    perfvsalphabeta = rets['perfvsalphabeta']
    perfvsbetaalpha = rets['perfvsbetaalpha']
    assert((abs(perfvsalpha[:,:2] - perfvsalpha2[:,:2])<0.000001).all())
    assert((abs(perfvsbeta[:,:2] - perfvsbeta2[:,:2])<0.000001).all())
    assert((abs(perfvslmbda[:,:2] - perfvslmbda2[:,:2])<0.000001).all())
    assert((abs(perfvsalphabeta[:,:2] - perfvsalphabeta2[:,:2])<0.000001).all())
    assert((abs(perfvsbetaalpha[:,:2] - perfvsbetaalpha2[:,:2])<0.000001).all())
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testcreatetableavg']
    unittest.main()
    
