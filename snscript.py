# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:11:07 2016

@author: Eric
"""

import argparse
import pickle
import sparsenet
import numpy as np
import scipy.io as io

parser = argparse.ArgumentParser(description="Learn dictionaries for Sparsenet with given parameters.")
parser.add_argument('-o', '--overcompleteness', default=4, type=float)
parser.add_argument('-d', '--data', default='images', type=str)
parser.add_argument('-r', '--resultsfolder', default='',type=str)
parser.add_argument('-s', '--datasuffix', default='ptwise', type=str)
parser.add_argument('-i', '--niter', default=200, type=int)
parser.add_argument('-l', '--lam', default=0.15, type=float)
args=parser.parse_args()

data = args.data
resultsfolder = args.resultsfolder
oc = args.overcompleteness
datasuffix = args.datasuffix
niter = args.niter
lam = args.lam

if data == 'images':
    datafile = '../vision/Data/IMAGES.mat'
    numinput = 256
    numunits = int(oc*numinput)
    data = io.loadmat(datafile)["IMAGES"]
    if resultsfolder == '':
        resultsfolder = '../vision/Results/'
    
    net = sparsenet.Sparsenet(data, numunits, paramfile='dummy')
elif data == 'spectros':
    datafile = '../audition/Data/speech_ptwisecut'
    numinput = 200
    numunits = int(oc*numinput)
    
    with open(datafile+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'.npy')
    data = data/data.std()
    if resultsfolder == '':
        resultsfolder = '../audition/Results/'
    
    net = sparsenet.Sparsenet(data, numunits, datatype="spectro", pca = mypca, stimshape=origshape, paramfile='dummy')

net.niter = niter
net.lamb = lam
net.learnrate = 0.0005
net.gain_rate = 0.001

savestr = resultsfolder+'SN'+str(oc)+'OC' + str(lam) + datasuffix
net.save(savestr+'.pickle')
net.run(ntrials=50000)
net.learnrate = net.learnrate/5
net.run(50000)
net.save()
