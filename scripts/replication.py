# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:31:30 2016

@author: Eric
"""
import argparse
import scipy.io as io
#import pickle
import LCALearner
import numpy as np
import sys
import pca.pca
sys.modules['pca'] = pca.pca #this is a workaround to get the pickled pca to load.  I think it basically tells python that pca.pca is an acceptable name for pca

parser = argparse.ArgumentParser(description="Learn dictionaries for LCA with given parameters.")
parser.add_argument('-o', '--overcompleteness', default=4, type=float)
parser.add_argument('-f', '--datafolder', default='../audition/Nicole Code/', type=str)
parser.add_argument('-r', '--resultsfolder', default='../audition/Results/',type=str)
parser.add_argument('-s', '--datasuffix', default='new', type=str)
args=parser.parse_args()

datafolder = args.datafolder#'../audition/Data/'
resultsfolder = args.resultsfolder
oc = args.overcompleteness
datasuffix = args.datasuffix

numinput = 200
numunits = int(oc*numinput)


stuff = io.loadmat(datafolder+'PCAmatrices'+datasuffix+'.mat')
mypca = pca.pca.PCA(dim=200,whiten=True)
mypca.eVectors = stuff['E'].reshape((25,256,200))[:,::-1,:].reshape((6400,200)).T # flip the PC spectrograms upside down
mypca.sValues = np.sqrt(np.diag(np.abs(stuff['D1'])))
mypca.sValues = mypca.sValues[::-1]
mypca.mean_vec = np.zeros(6400)
mypca.ready=True
origshape = (25,256)

spectros = io.loadmat(datafolder+'dMatPCA'+datasuffix+'.mat')['dMatPCA'].T

lca = LCALearner.LCALearner(spectros, numunits, datatype="spectro", pca = mypca,  stimshape=origshape, paramfile='dummy')

lca.tolerance = .01
lca.max_iter = 4

lambdas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

for lam in lambdas:
    savestr = resultsfolder+str(oc)+'OC' + str(lam) + datasuffix
    lca.Q = lca.rand_dict()
    lca.min_thresh = lam
    lca.save_params(savestr+'.pickle')
    lca.run(ntrials=10000)
    lca.run(ntrials=200000, rate_decay=.99995)
    lca.sort_dict()
    lca.save_params()
