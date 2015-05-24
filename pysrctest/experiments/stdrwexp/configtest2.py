
import numpy as np
import cPickle as pickle
  
def main():
  ns          = 13
  ftype       = 'tabular'
  alphas    = np.arange(0, .2, .1)
  betas     = np.array([0, 0.1, 1.])
  lms       = np.arange(0, .2, .1)
  gamma   = 1.0
  gm      = np.ones(ns)*gamma
  gm[0]   = gm[ns-1] = 0
  Gamma   = np.diag(gm)
  nzG              = np.diag(Gamma)!=0.0
  initdist= np.zeros(ns)
  initdist[(ns-1)/2] = 1.
  configs     = [
                   {
                   'offpolicy' : True,
                   'algname'   : 'gtd',
                   'mdpseed'   : 1000,
                   'Gamma'     : Gamma,
                   'initsdist' : initdist,
                   'T'         : 10,
                   'N'         : 10,
                   'Rstd'      : 0.0,
                   'ftype'     : ftype,
                   'ns'        : ns,
                   'na'        : 2,
                   'nf'        : np.sum(nzG),
                   'behavRight': 0.5,
                   'targtRight': 0.99,
                   'alpha'     : alpha,
                   'beta'      : beta,
                   'lmbda'     : lm
                   }
                   for alpha in alphas
                   for beta in betas
                   for lm in lms
                ]
  
  f = open('config2.pkl', 'wb')
  
  pickle.dump(configs, f)

if __name__ == "__main__":
  main()  
