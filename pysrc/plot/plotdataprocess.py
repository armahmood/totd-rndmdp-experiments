'''
Created on Jan 25, 2015

@author: A. Rupam Mahmood

Responsibility of this program is to take several output files
from different runs for a particular algorithm on an experiment
and extract simple tables containing performance vs. a given
set of parameter by averaging over the runs and choosing the
best values for the rest of the parameters. This table can then
be easily loaded up and plotted using standard matplotlib tools.

'''

import numpy as np
import cPickle as pickle
import itertools
import sys
import argparse


def loaddata(nruns, pathfileprefix):
  data = []
  for run in range(1, nruns + 1):
    filepathname = pathfileprefix + str(run) + ".dat"
    f = open(filepathname, 'rb')
    try:
      while True:
        d = pickle.load(f)
        data.append(d)
    
    except EOFError:
      print 'End of file reached'
  return data

def createtable(data, params, neps):
  table       = np.zeros((len(data), len(params)+neps))
  nparams     = len(params)

  for i in range(len(data)):
    for j in range(nparams):
      table[i,j]    = data[i][params[j]]
    table[i,nparams:]   = data[i]['error'] # error, maybe in squared form 
  return table

def createtableavg(table, nruns, neps):
  (tablerows, tablecols)      = np.shape(table)
  tableavgrows                = tablerows/nruns
  nparams                     = tablecols-neps
  tableavgcols                = nparams+2
  tableavgstd                 = np.zeros((tableavgrows, tableavgcols))
  tabletemp                   = np.zeros((tableavgrows, neps))
  tableavgstd[:, :nparams]  = table[:tableavgrows, :nparams]
  tabletemp[:,:neps]          = table[:tableavgrows, nparams:]
  
  for i in range(1, nruns):
    tabletemp = np.concatenate((tabletemp, \
                table[(i)*tableavgrows:(i+1)*tableavgrows, nparams:]), 1)
    #print np.shape(tabletemp)
    #print np.shape(table[(i)*tableavgrows:(i+1)*tableavgrows, nparams:])
    
  tableavgstd[:, nparams] = np.mean(tabletemp, 1)
  tableavgstd[:, nparams+1] = np.std(tabletemp, 1)/np.sqrt(neps*nruns)
  
  return tableavgstd
  
def performancevsparams(tableavgstd, params, paramssub):
  
  (tableavgrows, tableavgcols)    = np.shape(tableavgstd)

  paramvals = {}
  
  nparamssubvals = 1.
  for param in params:
      paramvals[param]   = np.unique(tableavgstd[:, param==params])
      if (param==paramssub).any():
        nparamssubvals *= len(paramvals[param])
  
  paramsubvalcomblist = list(itertools.product(\
                      *[paramvals[param] for param in paramssub ]))

  perftable           = np.zeros((len(paramsubvalcomblist), \
                tableavgcols - len(params)+len(paramssub)))
  row = 0
  for paramsubvalcomb in paramsubvalcomblist:
    paramsubvalcomb = np.array(paramsubvalcomb)
    condition = np.array(np.repeat(True, tableavgrows))
    for param in params:
      if (param==paramssub).any():
        condition = condition * \
          (tableavgstd[:,param==params] == paramsubvalcomb[param==paramssub])\
            .reshape(tableavgrows)
    perftable[row,:len(paramssub)] = paramsubvalcomb
    perftable[row,len(paramssub)] = np.nanmin(tableavgstd[condition,len(params)])
    perftable[row,len(paramssub)+1] = np.nanmin(tableavgstd[condition,len(params)+1])
    
    row += 1
    
  return perftable
  
def main():

  if len(sys.argv)>1:
    nruns  = int(sys.argv[1])
    pathfileprefix   = sys.argv[2]
    nparams = int(sys.argv[3])
    params = np.array([ sys.argv[4+i] for i in range(nparams) ])
    nparamssub = int(sys.argv[4+nparams]) 
    paramssub = np.array( [ sys.argv[4+nparams+1+i] for i in range(nparamssub) ] )
  data        = loaddata(nruns, pathfileprefix)
  neps        = data[0]['N'] # number of data points
  table       = createtable(data, params, neps)
  tableavgstd = createtableavg(table, nruns, neps)
  
  perftable = performancevsparams(tableavgstd, params, paramssub)
  
  print perftable    
  fsname    = pathfileprefix+'perfvs'
  for i in range(len(paramssub)): fsname += paramssub[i]
  fsname    += ".plot"
  fs           = open(fsname, "wb")
  pickle.dump(perftable, fs)

if __name__ == '__main__':
    main()
    
    
    
