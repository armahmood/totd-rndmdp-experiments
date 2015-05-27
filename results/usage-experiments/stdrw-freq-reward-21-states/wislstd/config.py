
import numpy as np
import cPickle as pickle

  
ns          = 23
ftype       = 'binary'
if ftype=='tabular':  nf = ns-2
elif ftype=='binary': nf = int(np.ceil(np.log(ns-1)/np.log(2)))
initas    = 10**np.arange(-3, 3.1, .2)
lms       = np.concatenate((np.arange(0, .9, .1), np.arange(.9, 1.01, .01)))
configs     = [
                 {
                   'ns'        : ns,
                 'algname'   : 'wislstd',
                 'gamma'     : 1.0,
                 'N'      : 200,
                 'ftype'     : ftype,
                 'nstates'   : ns,
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
