
import numpy as np
import cPickle as pickle

  
etas      = np.concatenate(([0], 10**np.arange(-6, 0.1, 0.25)))
initds    = 10**np.arange(-2, 1.1, .25)
ratios    = np.array([0, 0.001, 0.01, 0.1, 1.] )
lms       = np.concatenate((np.arange(0, .9, .1), np.arange(.9, 1.01, .01)))
configs     = [
                 {
                 'eta'       : min(etas, key=lambda x:abs(x-ratio/initd)),
                 'initd'     : initd,
                 'lmbda'    : lm
                 }
                 for ratio in ratios
                 for initd in initds
                 for lm in lms
              ]

f = open('configalg.pkl', 'wb')

pickle.dump(configs, f)
