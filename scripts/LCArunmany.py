# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 22:33:12 2015

@author: Eric
"""

import scipy.io as io
import LCALearner
import numpy as np
import sys
import pca.pca
sys.modules['pca'] = pca.pca

import matplotlib.pyplot as plt
plt.ioff()

overcompleteness = 0.5
numinput = 200
numunits = int(overcompleteness*numinput)

stuff = io.loadmat('../audition/Nicole Code/PCAmatrices4.mat')
mypca = pca.pca.PCA(dim=200,whiten=True)
mypca.eVectors = stuff['E'].reshape((25,256,200))[:,::-1,:].reshape((6400,200)).T # flip the PC spectrograms upside down
mypca.sValues = np.sqrt(np.diag(np.abs(stuff['D1'])))
mypca.sValues = mypca.sValues[::-1]
mypca.mean_vec = np.zeros(6400)
mypca.ready=True
origshape = (25,256)
spectros = io.loadmat('../audition/Nicole Code/dMatPCA4.mat')['dMatPCA'].T

lca = LCALearner.LCALearner(spectros, numunits, datatype="spectro", pca = mypca,  stimshape=origshape, paramfile='dummy')

lca.max_iter = 1
ntrials = 100000
rate_decay = .9999

#lambdas = [.2,.4,.6,.8,1.,1.2,1.4]
lambdas = [1.5, 2]

lca.infrate = .0005

savedir = '../audition/Results/halfOC/oldprep/'

for lam in lambdas:
    lamstring = savedir+'lambda'+str(lam).replace('.','')
    lca.save_params(lamstring+'.pickle')
    
    lca.learnrate = 1./lca.batch_size
    lca.min_thresh = lam
    lca.Q = lca.rand_dict()
    
    lca.run(ntrials = ntrials, show=False, rate_decay=.9999)
    
    plt.figure()
    lca.sort_dict(80000, plot=True)
    plt.savefig(lamstring+'usage.png')
    plt.figure()
    lca.show_dict(cmap='jet')
    plt.savefig(lamstring+'.png')
    lca.save_params()
    