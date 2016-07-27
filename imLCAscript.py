# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:54:12 2016

@author: Eric
"""

import argparse
import pickle
import LCALearner
import numpy as np
import scipy.io as io

parser = argparse.ArgumentParser(description="Learn dictionaries for LCA with given parameters.")
parser.add_argument('-o', '--overcompleteness', default=4, type=float)
parser.add_argument('-d', '--datafile', default='../vision/Data/IMAGES.mat', type=str)
parser.add_argument('-r', '--resultsfolder', default='../vision/Results/',type=str)
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

numinput = 256
numunits = int(oc*numinput)

data = io.loadmat(datafile)["IMAGES"]

lca = LCALearner.LCALearner(data, numunits, paramfile='dummy')

lca.min_thresh = lam
lca.max_iter = 1
lca.niter = niter
lca.infrate = 0.01

savestr = resultsfolder+'im'+str(oc)+'OC' + str(lam) + datasuffix
lca.save_params(savestr+'.pickle')
lca.run(ntrials=50000)
#lca.run(ntrials=200000, rate_decay=.99995)
lca.niter=500
lca.run(ntrials=10000)
lca.sort_dict()
lca.save_params()
