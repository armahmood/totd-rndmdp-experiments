'''
Created on Apr 28, 2015

@author: A. Rupam Mahmood
'''

import numpy as np
from pysrc.problems import blindbee
from pysrc.algorithms.tdprediction import etd
from matplotlib import pyplot as ppl

def main():
    config    = {
                 'runseed'      :1,
                 'policy'       :'upward',
                 'nrows'        :10,
                 'ncols'        :10,
                 'trapprob'     :0.75,
                 'trapduralim'  :10,
                 'alpha'        :0.001,
                 'lmbda'        :1.0,
                 'gamma'        :1.
                 }
    neps          = 5000
    prob          = blindbee.BlindBee(config)
    fet           = blindbee.Phi1()
    config['nf']  = fet.nf
    alg1           = etd.ETD(config)
    vals          = np.zeros(neps)
    for ep in range(neps):
      prob.initEpisode()
      alg1.initepisode()
      I = 1.
      while not prob.state.isTerminal():
        probstep              = prob.step()
        probstep['l']         = config['lmbda']
        probstep['lnext']     = config['lmbda']
        probstep['g']         = config['gamma']
        if probstep['snext'].isTerminal():
          probstep['gnext']     = 0.0
        else:
          probstep['gnext']     = config['gamma']        
        probstep['rho']       = 1.
        probstep['R']         = 1.
        probstep['I']         = I
        probstep['phi']       = fet.getPhi1(probstep['s'])
        probstep['phinext']   = fet.getPhi1(probstep['snext'])
        alg1.step(probstep)
        I = 1
      vals[ep]=alg1.th[0]
    print(alg1.th)
    ppl.plot(vals)
    ppl.show()

if __name__ == '__main__':
  main()
  
  
  