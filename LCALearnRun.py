# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:32:36 2015

@author: Eric
"""

import scipy.io
import pickle
import LCALearner

#images = scipy.io.loadmat('../SAILnet/PythonSAILnet/Data/Images.mat')["IMAGES"]
#lca = LCALearner.LCALearner(images, nunits=300, learn_rate = .001, batch_size=100, infrate=.01, niter=100)

ntimes = 25
nfreqs = 256
overcompleteness = 4
numinput = 200
numunits = int(overcompleteness*numinput)
with open("../SAILnet/PythonSAILnet/Pickles/spectropca.pickle",'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)
spectros = scipy.io.loadmat("../SAILnet/PythonSAILnet/Data/processedspeech.mat")["processedspeech"]
lca = LCALearner.LCALearner(spectros, numunits, datatype="spectro", pca = pca, stimshape=origshape[:2:-1])

#lca.run()