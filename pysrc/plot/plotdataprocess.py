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
import os.path

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

def createtablelearningcurves(table, nruns, neps):
  (tablerows, tablecols)      = np.shape(table)
  tableavgrows                = tablerows/nruns
  nparams                     = tablecols-neps
  tableavg                    = np.zeros((tableavgrows, tablecols))
  tablestd                    = np.zeros((tableavgrows, tablecols))
  tabletemp                   = np.zeros((tableavgrows, neps))
  tableavg[:, :nparams]       = table[:tableavgrows, :nparams]
  tablestd[:, :nparams]       = table[:tableavgrows, :nparams]
  tabletemp[:,:neps]          = table[:tableavgrows, nparams:]
  
  for i in range(1, nruns):
    tabletemp = np.concatenate((tabletemp, \
                table[(i)*tableavgrows:(i+1)*tableavgrows, nparams:]), 1)
  print np.shape(tabletemp)
    #print np.shape(table[(i)*tableavgrows:(i+1)*tableavgrows, nparams:])
    
  tableavg[:, nparams:] = np.mean(np.reshape(tabletemp, (tableavgrows, nruns, neps)), 1)
  tablestd[:, nparams:] = np.std(np.reshape(tabletemp, (tableavgrows, nruns, neps)), 1)/np.sqrt(nruns)
  
  return (tableavg, tablestd)

def createtableavg(table, nruns, neps, startstep=0):
  (tablerows, tablecols)      = np.shape(table)
  tableavgrows                = tablerows/nruns
  nparams                     = tablecols-neps
  tableavgcols                = nparams+2
  tableavgstd                 = np.zeros((tableavgrows, tableavgcols))
  tableavgstd[:, :nparams]    = table[:tableavgrows, :nparams]
  tabletemp                   = table[:tableavgrows, (nparams+startstep):]
  
  for i in range(1, nruns):
    tabletemp = np.concatenate((tabletemp, \
                table[(i)*tableavgrows:(i+1)*tableavgrows, (nparams+startstep):]), 1)
    
  r, c                      = np.shape(tabletemp)
  tabletemp2                = np.mean(np.reshape(tabletemp, (r, c/nruns, nruns),1), 1) 
  tableavgstd[:, nparams]   = np.mean(tabletemp2, 1)
  tableavgstd[:, nparams+1] = np.std(tabletemp2, 1)/np.sqrt(nruns)
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

''' same as main, but provides an option to average out
    only the last part of a run
'''
def main2(nruns, pathfileprefix, nparams, params, nparamssub, paramssub, startstep):
  params            = np.array(params)
  paramssub         = np.array(paramssub)
  tablefilename     = pathfileprefix+"perftable.plot.pkl"
  if not os.path.isfile(tablefilename):
    # Produce and dump the averaged table first
    data        = loaddata(nruns, pathfileprefix)
    neps        = data[0]['N'] # number of data points
    table       = createtable(data, params, neps)
    tableavgstd = createtableavg(table, nruns, neps, startstep)
  
    fs           = open(tablefilename, "wb")
    pickle.dump(tableavgstd, fs)
  else:
    tableavgstd = pickle.load(open(tablefilename, "rb"))
 
  perftable = performancevsparams(tableavgstd, params, paramssub)
  
  print perftable    
  fsname    = pathfileprefix+'perfvs'
  for i in range(len(paramssub)): fsname += paramssub[i]
  fsname    += ".plot.pkl"
  fs           = open(fsname, "wb")
  pickle.dump(perftable, fs)
  
def main():

  if len(sys.argv)>1:
    nruns  = int(sys.argv[1])
    pathfileprefix   = sys.argv[2]
    nparams = int(sys.argv[3])
    params = np.array([ sys.argv[4+i] for i in range(nparams) ])
  tablefilename = pathfileprefix+"perftable.plot.pkl"
  if not os.path.isfile(tablefilename):
    # Produce and dump the averaged table first
    data        = loaddata(nruns, pathfileprefix)
    neps        = data[0]['N'] # number of data points
    table       = createtable(data, params, neps)
    tableavgstd = createtableavg(table, nruns, neps)
  
    fs           = open(tablefilename, "wb")
    pickle.dump(tableavgstd, fs)
  else:
    tableavgstd = pickle.load(open(tablefilename, "rb"))

  if (len(sys.argv)==4+nparams):
    return
  # If a subset of parameters provided, then produce a table 
  # with respect to those parameters
  nparamssub = int(sys.argv[4+nparams]) 
  paramssub = np.array( [ sys.argv[4+nparams+1+i] for i in range(nparamssub) ] )
  perftable = performancevsparams(tableavgstd, params, paramssub)
  
  print perftable    
  fsname    = pathfileprefix+'perfvs'
  for i in range(len(paramssub)): fsname += paramssub[i]
  fsname    += ".plot.pkl"
  fs           = open(fsname, "wb")
  pickle.dump(perftable, fs)

if __name__ == '__main__':
    main()
    
    
    
