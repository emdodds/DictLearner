# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:31:30 2016

@author: Eric
"""
import argparse
import pickle
import LCALearner
import numpy as np

parser = argparse.ArgumentParser(description="Learn dictionaries for LCA with given parameters.")
parser.add_argument('-o', '--overcompleteness', default=4, type=float)
parser.add_argument('-d', '--datafile', default='../audition/Data/speech_ptwisecut', type=str)
parser.add_argument('-r', '--resultsfolder', default='../audition/Results/',type=str)
parser.add_argument('-s', '--datasuffix', default='ptwise', type=str)
parser.add_argument('-i', '--niter', default=200, type=int)
parser.add_argument('-l', '--lam', default=0.6, type=float)
args=parser.parse_args()

datafile = args.datafile
resultsfolder = args.resultsfolder
oc = args.overcompleteness
datasuffix = args.datasuffix
niter = args.niter
lam = args.lam

numinput = 200
numunits = int(oc*numinput)

with open(datafile+'_pca.pickle', 'rb') as f:
    mypca, origshape = pickle.load(f)
data = np.load(datafile+'.npy')*283

lca = LCALearner.LCALearner(data, numunits, datatype="spectro", pca = mypca,  stimshape=origshape, paramfile='dummy')

lca.min_thresh = lam
lca.max_iter = 1
lca.niter = niter
lca.infrate = 0.01

savestr = resultsfolder+str(oc)+'OC' + str(lam) + datasuffix
lca.save(savestr+'.pickle')
lca.run(ntrials=50000)
lca.run(ntrials=200000, rate_decay=.99995)
lca.sort_dict()
lca.save()
