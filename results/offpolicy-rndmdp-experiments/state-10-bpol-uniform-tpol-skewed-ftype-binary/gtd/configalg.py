
import numpy as np
import cPickle as pickle

alphas    = 10**np.arange(-3, 0.1, .5)
betas     = np.array([0])
#betas     = np.concatenate(([0], 10**np.arange(-3, -0.9, 1.)))
lms       = np.concatenate((np.arange(0, .9, .1), np.arange(.9, 1.01, .01)))
configs     = [
                 {
                 'alpha'     : alpha,
                 'beta'      : beta,
                 'lmbda'    : lm
                 }
                 for alpha in alphas
                 for beta in betas
                 for lm in lms
              ]

f = open('configalg.pkl', 'wb')

pickle.dump(configs, f)
