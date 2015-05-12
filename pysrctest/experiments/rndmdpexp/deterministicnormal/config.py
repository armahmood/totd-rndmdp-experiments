
import numpy as np
import cPickle as pickle
  
def main():
  
  ns          = 100
  lmbdas      = np.array([0.1])
  alphas      = np.array([10**-3])
  configs     = [
                   {
                   'algname'    : 'totd',
                   'Gamma'      : 0.99*np.eye(ns),
                   'T'          : 1000,
                   'N'          : 1,
                   'ns'         : ns,
                   'na'         : 1,
                   'nf'         : 5,
                   'b'          : 3,
                   'rtype'      :'normal', 
                   'rparam'     :1,
                   'Rstd'       : 0.0,
                   'initsdist'  : 'statezero',
                   'alpha'      : alpha,
                   'lmbda'      : lmbda
                   }
                   for alpha in alphas
                   for lmbda in lmbdas
                ]
  
  f = open('config.pkl', 'wb')
  
  pickle.dump(configs, f)
  
if __name__ == "__main__":
  main()  

