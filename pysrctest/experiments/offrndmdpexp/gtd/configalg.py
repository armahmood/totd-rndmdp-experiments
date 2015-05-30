
import numpy as np
import cPickle as pickle

alphas    = np.array([0.01])
betas     = np.array([0.001])
lms       = np.array([0.9])
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
