
import numpy as np
import cPickle as pickle

  
initas    = 10**np.arange(-3, 3.1, .2)
lms       = np.concatenate((np.arange(0, .9, .1), np.arange(.9, 1.01, .01)))
configs     = [
                 {
                 'inita'     : inita,
                 'lambda'    : lm
                 }
                 for inita in initas
                 for lm in lms
              ]

f = open('config.pkl', 'wb')

pickle.dump(configs, f)
