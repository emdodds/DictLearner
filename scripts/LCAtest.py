# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:30:27 2015

@author: Eric
"""

import pickle
import numpy as np
from LCALearner import LCALearner
import sys
import pca.pca
sys.modules['pca'] = pca.pca

datafolder = '../audition/Data/'
picklefile = datafolder + 'spectropca2.pickle'

data = np.eye(200)

with open(picklefile,'rb') as f:
        pca, origshape, datamean, datastd = pickle.load(f)

lca = LCALearner(data, 200, datatype="spectro", pca = pca,  stimshape=origshape, paramfile='dummy')

#lca.Q = np.eye(200)
#X = np.eye(200)
#s, errors, histories, shistories, threshhist = lca.infer(X, True)

lca.run()