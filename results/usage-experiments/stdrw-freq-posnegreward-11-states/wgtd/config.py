
import numpy as np
import cPickle as pickle

  
ns          = 13
ftype       = 'binary'
if ftype=='tabular':  nf = ns-2
elif ftype=='binary': nf = int(np.ceil(np.log(ns-1)/np.log(2)))
etas      = np.concatenate(([0], 10**np.arange(-6, 0.1, 0.25)))
initds    = 10**np.arange(0, 3.1, .25)
ratios    = np.array([0, 0.001, 0.01, 0.1, 1.] )
betas     = np.array([0])
lms       = np.concatenate((np.arange(0, .9, .1), np.arange(.9, 1.01, .01)))
configs     = [
                 {
                   'ns'        : ns,
                 'algname'   : 'wgtd',
                 'gamma'     : 1.0,
                 'N'      : 100,
                 'ftype'     : ftype,
                 'nstates'   : ns,
                 'nf'        : nf,
                 'inits'     : (ns-1)/2,
                 'mright'    : 0.5,
                 'pright'    : 0.99,
                 'eta'       : min(etas, key=lambda x:abs(x-ratio/initd)),
                 'initd'     : initd,
                 'beta'      : beta,
                 'lambda'    : lm
                 }
                 for ratio in ratios
                 for initd in initds
                 for beta in betas
                 for lm in lms
              ]

f = open('config.pkl', 'wb')

pickle.dump(configs, f)
