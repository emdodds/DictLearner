# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:32:07 2015

@author: Eric
"""
import time
import scipy.io as io
#import pickle
import LCALearner
import numpy as np
import sys
import pca.pca
sys.modules['pca'] = pca.pca #this is a workaround to get the pickled pca to load.  I think it basically tells python that pca.pca is an acceptable name for pca

#images = scipy.io.loadmat('../SAILnet/PythonSAILnet/Data/Images.mat')["IMAGES"]
#lca = LCALearner.LCALearner(images, nunits=300, learn_rate = .001, batch_size=100, infrate=.01, niter=100,
#                            min_thresh = 0.8, adapt=.98)

datafolder = ''#/audition/Data/'

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

stuff = io.loadmat('../audition/Nicole Code/PCAmatrices2.mat')
mypca = pca.pca.PCA(dim=200,whiten=True)
mypca.eVectors = stuff['E'].T
mypca.sValues = np.sqrt(np.diag(np.abs(stuff['D1'])))
mypca.sValues = mypca.sValues[::-1]
mypca.mean_vec = np.zeros(6400)
mypca.ready=True
origshape = (25,256)

spectros = io.loadmat('../audition/Nicole Code/dMatPCA2.mat')['dMatPCA'].T

lca = LCALearner.LCALearner(spectros, numunits, datatype="spectro", pca = mypca,  stimshape=origshape, paramfile='dummy')

lca.tolerance = .01
lca.max_iter = 1

X = lca.stims.rand_stim()
t=time.time()
scpu = lca.infer(X)[0]
print("CPU inference time: " + str(time.time()-t)+'\n')
lca.gpu=True
t=time.time()
sgpu=lca.infer(X)[0]
print("GPU inference time: " + str(time.time()-t)+'\n')
if not np.allclose(scpu,sgpu):
    print("Results disagree.")
