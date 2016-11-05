# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:31:30 2016

@author: Eric
"""
import argparse
import pickle
import LCALearner
import LCAmods
import numpy as np
import scipy.io as io

parser = argparse.ArgumentParser(description="Learn dictionaries for LCA with given parameters.")
parser.add_argument('-o', '--overcompleteness', default=4, type=float)
parser.add_argument('-d', '--data', default='images', type=str)
#parser.add_argument('-d', '--datafile', default='../audition/Data/speech_ptwisecut', type=str)
parser.add_argument('-r', '--resultsfolder', default='',type=str)
parser.add_argument('-s', '--savesuffix', default='', type=str)
parser.add_argument('-i', '--niter', default=200, type=int)
parser.add_argument('-l', '--lam', default=0.6, type=float)
parser.add_argument('--load', action='store_true')
parser.add_argument('--pos', default = False, type=bool)
args=parser.parse_args()

#datafile = args.datafile
data = args.data
resultsfolder = args.resultsfolder
oc = args.overcompleteness
savesuffix = args.savesuffix
niter = args.niter
lam = args.lam
load = args.load
pos = args.pos

if pos:
    Learner = LCAmods.PositiveLCA
else:
    Learner = LCALearner.LCALearner

if data == 'images':
    datafile = '../vision/Data/IMAGES.mat'
    numinput = 256
    numunits = int(oc*numinput)
    data = io.loadmat(datafile)["IMAGES"]
    if resultsfolder == '':
        resultsfolder = '../vision/Results/'
    lca = Learner(data, numunits, paramfile='dummy')
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
    lca = Learner(data, numunits, datatype="spectro", pca = mypca,  stimshape=origshape, paramfile='dummy')


lca.min_thresh = lam
lca.max_iter = 1
lca.niter = niter
lca.infrate = 0.01
lca.learnrate = 0.0005

savestr = resultsfolder+str(oc)+'OC' + str(lam) + savesuffix
if load:
    lca.load(savestr + '.pickle')
lca.save(savestr+'.pickle')
lca.run(ntrials=50000)
lca.run(ntrials=200000, rate_decay=.99995)
lca.save()
