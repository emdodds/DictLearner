# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:32:36 2015

@author: Eric
"""
import matplotlib.pyplot as plt
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

prep = 'new'
resultsfolder = '../audition/Results/halfOC/'+prep +'prep/'

overcompleteness = 0.5
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

stuff = io.loadmat('../audition/Nicole Code/PCAmatrices'+prep+'.mat')
mypca = pca.pca.PCA(dim=200,whiten=True)
mypca.eVectors = stuff['E'].reshape((25,256,200))[:,::-1,:].reshape((6400,200)).T # flip the PC spectrograms upside down
mypca.sValues = np.sqrt(np.diag(np.abs(stuff['D1'])))
mypca.sValues = mypca.sValues[::-1]
mypca.mean_vec = np.zeros(6400)
mypca.ready=True
origshape = (25,256)

spectros = io.loadmat('../audition/Nicole Code/dMatPCA'+prep+'.mat')['dMatPCA'].T

lca = LCALearner.LCALearner(spectros, numunits, datatype="spectro", pca = mypca,  stimshape=origshape, paramfile='dummy')

lca.load_params(resultsfolder+'lam0.6newdropiters.pickle')#'lam0.6new.pickle')
#lca.sort_dict(allstims=True, plot=True)
#plt.savefig(resultsfolder+'4OC0_6'+prep+'usage.png')
#lca.show_oriented_dict(batch_size=1000)
#plt.savefig(resultsfolder+'4OC0_6'+prep+'.png')

#lca.min_thresh = .6
#lca.save_params('halfOC_alt.pickle')
#lca.run(ntrials=2000)
#lca.eta = lca.eta/10
#lca.run(ntrials=100000, rate_decay=.9999)

#for i in range(20):
#    lca.run(ntrials=10000)
#    lca.eta = lca.eta*.9
#    lca.min_thresh = (lca.min_thresh + .05) if (lca.min_thresh < .9) else lca.min_thresh