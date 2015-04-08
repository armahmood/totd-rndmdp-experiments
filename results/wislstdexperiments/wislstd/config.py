
import numpy as np
import cPickle as pickle
  
def main():
  
  ns          = 5
  ftype       = 'binary'
  if ftype=='tabular':  nf = ns-2
  elif ftype=='binary': nf = int(np.ceil(np.log(ns-1)/np.log(2)))
  initas    = 10**np.arange(-3, 4, 1.)
  lms       = np.array([0, 0.5, 0.9, 0.95, 0.99, 1.0])
  configs     = [
                   {
                   'algname'   : 'oislstd',
                   'gamma'     : 1.0,
                   'neps'      : 100,
                   'ftype'     : ftype,
                   'ns'        : ns,
                   'nf'        : nf,
                   'inits'     : (ns-1)/2,
                   'mright'    : 0.5,
                   'pright'    : 0.99,
                   'inita'     : inita,
                   'lambda'    : lm
                   }
                   for inita in initas
                   for lm in lms
                ]
  
  f = open('config.pkl', 'wb')
  
  pickle.dump(configs, f)
  
if __name__ == "__main__":
  main()  

