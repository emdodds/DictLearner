# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:31:30 2016

@author: Eric
"""

import scipy.io as io
#import pickle
import LCALearner
import numpy as np
import sys
import pca.pca
sys.modules['pca'] = pca.pca #this is a workaround to get the pickled pca to load.  I think it basically tells python that pca.pca is an acceptable name for pca

datafolder = '../audition/Nicole Code/'#'../audition/Data/'

overcompleteness = 4
numinput = 200
numunits = int(overcompleteness*numinput)
#picklefile = datafolder + 'spectropca5.pickle'
#datafile = datafolder + 'processedspeech5.npy'

#with open(picklefile,'rb') as f:
#        pca, origshape, datamean, datastd = pickle.load(f)
#spectros = scipy.io.loadmat("../SAILnet/PythonSAILnet/Data/processedspeech.mat")["processedspeech"]
#spectros = np.load(datafile)
#center = np.mean(spectros)
#spectros = spectros - center
#scale = np.std(spectros)
#spectros = spectros/scale

stuff = io.loadmat(datafolder+'PCAmatricesold.mat')
mypca = pca.pca.PCA(dim=200,whiten=True)
mypca.eVectors = stuff['E'].reshape((25,256,200))[:,::-1,:].reshape((6400,200)).T # flip the PC spectrograms upside down
mypca.sValues = np.sqrt(np.diag(np.abs(stuff['D1'])))
mypca.sValues = mypca.sValues[::-1]
mypca.mean_vec = np.zeros(6400)
mypca.ready=True
origshape = (25,256)

spectros = io.loadmat(datafolder+'dMatPCAold.mat')['dMatPCA'].T

lca = LCALearner.LCALearner(spectros, numunits, datatype="spectro", pca = mypca,  stimshape=origshape, paramfile='dummy')

lca.tolerance = .01
lca.max_iter = 4

lambdas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


lca.Q = lca.rand_dict()

#lca.min_thresh = .6
#lca.save_params('halfOC_alt.pickle')
#lca.run(ntrials=2000)
#lca.eta = lca.eta/10
#lca.run(ntrials=100000, rate_decay=.9999)

#for i in range(20):
#    lca.run(ntrials=10000)
#    lca.eta = lca.eta*.9
#    lca.min_thresh = (lca.min_thresh + .05) if (lca.min_thresh < .9) else lca.min_thresh