
import numpy as np
import cPickle as pickle
  
def main():
  
  ns          = 13
  ftype       = 'binary'
  etas      = np.concatenate(([0], 10**np.arange(-6, 0.1, 0.25)))
  initds    = 10**np.arange(0, 3.1, .25)
  ratios    = np.array([0, 0.001, 0.01, 0.1, 1.] )
  betas     = np.array([0])
  lms       = np.concatenate((np.arange(0, .9, .1), np.arange(.9, 1.01, .01)))
  gamma   = 1.0
  gm      = np.ones(ns)*gamma
  gm[0]   = gm[ns-1] = 0
  Gamma   = np.diag(gm)
  nzG              = np.diag(Gamma)!=0.0
  initdist= np.zeros(ns)
  initdist[(ns-1)/2] = 1.
  configs     = [
                   {
                   'algname'   : 'wgtd',
                   'mdpseed'   : 1000,
                   'Gamma'     : Gamma,
                   'initsdist' : initdist,
                   'Rstd'      : 0.0,
                   'T'         : 100,
                   'N'         : 100,
                   'ftype'     : ftype,
                   'ns'        : ns,
                   'na'        : 2,
                   'nf'        : int(np.ceil(np.log(np.sum(nzG)-1)/np.log(2))),
                   'behavRight': 0.5,
                   'targtRight': 0.99,
                   'eta'       : min(etas, key=lambda x:abs(x-ratio/initd)),
                   'initd'     : initd,
                   'beta'      : beta,
                   'lmbda'    : lm
                   }
                 for ratio in ratios
                 for initd in initds
                 for beta in betas
                 for lm in lms
                ]
  
  f = open('config2.pkl', 'wb')
  
  pickle.dump(configs, f)
  
if __name__ == "__main__":
  main()  

