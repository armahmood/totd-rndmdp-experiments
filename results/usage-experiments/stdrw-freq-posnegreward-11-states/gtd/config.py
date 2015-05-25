
import numpy as np
import cPickle as pickle

  
ns          = 13
ftype       = 'binary'
if ftype=='tabular':  nf = ns-2
elif ftype=='binary': nf = int(np.ceil(np.log(ns-1)/np.log(2)))
alphas    = 10**np.arange(-3, 0.1, .25)
betas     = np.concatenate(([0], 10**np.arange(-3, -0.9, 1.)))
lms       = np.concatenate((np.arange(0, .9, .1), np.arange(.9, 1.01, .01)))
configs     = [
                 {
                   'ns'      : ns,
                 'algname'   : 'gtd',
                 'gamma'     : 1.0,
                 'N'         : 100,
                 'ftype'     : ftype,
                 'nstates'   : ns,
                 'nf'        : nf,
                 'inits'     : (ns-1)/2,
                 'mright'    : 0.5,
                 'pright'    : 0.99,
                 'alpha'     : alpha,
                 'beta'      : beta,
                 'lambda'    : lm
                 }
                 for alpha in alphas
                 for beta in betas
                 for lm in lms
              ]

f = open('config.pkl', 'wb')

pickle.dump(configs, f)
