
import numpy as np
import cPickle as pickle
  
def main():
  
  ns          = 13
  ftype       = 'tabular'
  alphas    = np.arange(0, 1.6, .1)
  betas     = np.array([0, 0.1, 1.])
  lms       = np.concatenate((np.arange(0, .9, .1), np.arange(.9, 1.01, .01)))
  gamma   = 1.0
  gm      = np.ones(ns)*gamma
  gm[0]   = gm[ns-1] = 0
  Gamma   = np.diag(gm)
  nzG              = np.diag(Gamma)!=0.0
  configs     = [
                   {
                   'offpolicy' : True,
                   'algname'   : 'gtd',
                   'mdpseed'   : 1000,
                   'Gamma'     : Gamma,
                   'initsdist' : 'statemiddle',
                   'Rstd'      : 0.0,
                   'T'         : 100,
                   'N'         : 100,
                   'ftype'     : ftype,
                   'ns'        : ns,
                   'na'        : 2,
                   'nf'        : np.sum(nzG),
                   'inits'     : (ns-1)/2,
                   'behavRight': 0.5,
                   'targtRight': 0.99,
                   'alpha'     : alpha,
                   'beta'      : beta,
                   'lmbda'    : lm
                   }
                   for alpha in alphas
                   for beta in betas
                   for lm in lms
                ]
  
  f = open('config2.pkl', 'wb')
  
  pickle.dump(configs, f)
  
if __name__ == "__main__":
  main()  

