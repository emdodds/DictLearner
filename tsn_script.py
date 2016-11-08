# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:11:07 2016

@author: Eric
"""

import argparse
import pickle
import TopoSparsenet
import numpy as np
import scipy.io as io

parser = argparse.ArgumentParser(description="Learn dictionaries for Topographic Sparsenet with given parameters.")
parser.add_argument('-d', '--data', default='images', type=str)
parser.add_argument('-r', '--resultsfolder', default='',type=str)
parser.add_argument('-s', '--suffix', default='ptwise', type=str)
parser.add_argument('-i', '--niter', default=200, type=int)
parser.add_argument('-l', '--lam', default=0.15, type=float)
parser.add_argument('-l2', '--lam2', default=0.05, type=float)
#parser.add_argument('--shape', default = (25,32), type=tuple)
parser.add_argument('--sigma', default = 1, type=float)
parser.add_argument('--binarize', action='store_true')
args=parser.parse_args()

data = args.data
resultsfolder = args.resultsfolder
shape = (25,32)#args.shape
suffix = args.suffix
niter = args.niter
lam = args.lam
lam2 = args.lam2
sigma = args.sigma
binarize = args.binarize

if data == 'images':
    datafile = '../vision/Data/IMAGES.mat'
    numinput = 256
    data = io.loadmat(datafile)["IMAGES"]
    if resultsfolder == '':
        resultsfolder = '../vision/Results/'  
    net = TopoSparsenet.TopoSparsenet(data, shape, paramfile='dummy')
    net.gain_rate = 0.001
elif data == 'spectros':
    datafile = '../audition/Data/speech_ptwisecut'
    numinput = 200
    
    with open(datafile+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'.npy')
    data = data/data.std()
    if resultsfolder == '':
        resultsfolder = '../audition/Results/'
    
    net = TopoSparsenet.TopoSparsenet(data=data, dict_shape=shape,
                                      learnrate = 0.0005, datatype='spectro', pca=mypca, 
                           stimshape=origshape,
                           sigma=sigma,
                          gain_rate=0.001, var_goal=0.033)

net.niter = niter
net.lamb = lam
net.lamb_2 = lam2
net.learnrate = 0.0005

if binarize:
    net.binarize_g()

savestr = resultsfolder+'TSN'+str(shape[0])+'x'+str(shape[1]) + 's'+str(sigma)+ suffix
net.save(savestr+'.pickle')
net.run(ntrials=10000)
net.save()
