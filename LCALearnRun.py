# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:32:36 2015

@author: Eric
"""

#import scipy.io
import pickle
import LCALearner
import numpy as np
import sys
import pca.pca
sys.modules['pca'] = pca.pca #this is a workaround to get the pickled pca to load.  I think it basically tells python that pca.pca is an acceptable name for pca

#images = scipy.io.loadmat('../SAILnet/PythonSAILnet/Data/Images.mat')["IMAGES"]
#lca = LCALearner.LCALearner(images, nunits=300, learn_rate = .001, batch_size=100, infrate=.01, niter=100,
#                            min_thresh = 0.8, adapt=.98)

datafolder = '../audition/Data/'

overcompleteness = 2
numinput = 200
numunits = int(overcompleteness*numinput)
picklefile = datafolder + 'spectropca2.pickle'
datafile = datafolder + 'processedspeech2.npy'
paramfile = datafolder + 'lcaspectros'
ntrials = 10**4

with open(picklefile,'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
#spectros = scipy.io.loadmat("../SAILnet/PythonSAILnet/Data/processedspeech.mat")["processedspeech"]
spectros = np.load(datafile)
lca = LCALearner.LCALearner(spectros, numunits, datatype="spectro", pca = pca,  stimshape=origshape, paramfile='dummy')

#lca.load_params(paramfile)

#lca.run(ntrials = ntrials)

#lca.save_params(paramfile)
