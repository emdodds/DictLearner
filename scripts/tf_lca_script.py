from __future__ import division
from __future__ import print_function
import tf_lca
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description="Learn dictionaries for LCA with"
                                 "given parameters.")
parser.add_argument('-d', '--data', default='images', type=str)
parser.add_argument('-r', '--resultsfolder', default='', type=str)
parser.add_argument('-s', '--suffix', default='ptwise', type=str)
parser.add_argument('-i', '--niter', default=200, type=int)
parser.add_argument('-l', '--lam', default=0.1, type=float)
parser.add_argument('--nunits', '-n', default=200, type=int)
parser.add_argument('--snr', default=15.0, type=float)
parser.add_argument('--snr_rate', default=0.01, type=float)
args = parser.parse_args()

data = args.data
resultsfolder = args.resultsfolder
nunits = args.nunits
suffix = args.suffix
niter = args.niter
lam = args.lam
snr_goal = args.snr
snr_rate = args.snr_rate

numinput = 200
if data == 'images':
    dataprefix = '../vision/Data/300kvanHateren'
    data = np.load(dataprefix+'200.npy')
    if resultsfolder == '':
        resultsfolder = '../vision/Results/'
    with open(dataprefix+'PCA', 'rb') as f:
        mypca, origshape = pickle.load(f)
elif data == 'spectros':
    dataprefix = '...audition/Data/allTIMIT'
    with open(dataprefix+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(dataprefix+'.npy')
    if resultsfolder == '':
        resultsfolder = '../audition/Results/'
data /= data.std()

net = tf_lca.LCALearner(data=data, nunits=nunits,
                        pca=mypca, stimshape=origshape, lam=lam,
                        snr_goal=snr_goal,
                        seek_snr_rate=snr_rate,
                        learnrate=100.0,
                        threshfunc='soft')

savestr = (resultsfolder+'lca' +
           'im' if data == 'images' else 'sp' +
           str(nunits) +
           '.pickle')
net.save(savestr+'.pickle')
net.run(10000)
net.save()
net.set_learnrate(10.0)
net.run(20000)
net.set_learnrate(1.0)
net.run(20000)
