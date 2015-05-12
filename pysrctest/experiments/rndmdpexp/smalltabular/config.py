
import numpy as np
import cPickle as pickle
  
def main():
  
  ns          = 10
  lmbdas      = np.array([0])
  alphas      = np.array([10**-3])
  configs     = [
                   {
                   'algname'    : 'td',
                   'Gamma'      : 0.99*np.eye(ns),
                   'T'          : 100,
                   'N'          : 1,
                   'ns'         : ns,
                   'na'         : 1,
                   'b'          : 3,
                   'rtype'      :'normal', 
                   'rparam'     :1,
                   'Rstd'       : 0.1,
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

