import tf_toposparse
import numpy as np 
import pca
import pickle
import argparse

parser = argparse.ArgumentParser(description="Learn dictionaries for Topographic Sparsenet with given parameters.")
parser.add_argument('-d', '--data', default='images', type=str)
parser.add_argument('-r', '--resultsfolder', default='',type=str)
parser.add_argument('-s', '--suffix', default='ptwise', type=str)
parser.add_argument('-i', '--niter', default=200, type=int)
parser.add_argument('-l', '--lam', default=0.0, type=float)
parser.add_argument('--lam_g', default=0.05, type=float)
#parser.add_argument('--shape', default = (25,32), type=tuple) doesn't work

parser.add_argument('--dictlength', default=30, type=int)
parser.add_argument('--sigma', default = 1, type=float)
parser.add_argument('--binarize', action='store_true')
parser.add_argument('--discs', default=True, type=bool)
parser.add_argument('--torus', default=True, type=bool)
parser.add_argument('--ncomponents', default=1, type=int)
args=parser.parse_args()

data = args.data
resultsfolder = args.resultsfolder
shape = (args.dictlength, args.dictlength)
suffix = args.suffix
niter = args.niter
lam = args.lam
lam_g = args.lam_g
sigma = args.sigma
binarize = args.binarize
discs = args.discs
torus = args.torus
ncomponents = args.ncomponents

g_shape = np.prod(shape)
g_shape = (g_shape, g_shape)
topo = tf_toposparse.topology(g_shape, discs=discs, torus=torus,
                              binary=binarize, sigma=sigma, ncomponents=ncomponents)

if data == 'images':
     # not updated recently!!!
    datafile = '../vision/Data/IMAGES.mat'
    numinput = 256
    data = io.loadmat(datafile)["IMAGES"]
    if resultsfolder == '':
        resultsfolder = '../vision/Results/'  
    net = TopoSparsenet.TopoSparsenet(data, shape, paramfile='dummy')
    net.gain_rate = 0.001
elif data == 'spectros':
    datafile = '../Data/speech_ptwisecut'
    numinput = 200
    
    with open(datafile+'_pca.pickle', 'rb') as f:
        mypca, origshape = pickle.load(f)
    data = np.load(datafile+'.npy')
    data = data/data.std()
    if resultsfolder == '':
        resultsfolder = '../audition/Results/'
    
    net = tf_toposparse.TopoSparsenet(data=data, dict_shape=shape, datatype='spectro', pca=mypca, 
                            topo=topo,
                           stimshape=origshape,
                           lam=lam, lam_g = lam_g,
                           niter = niter,
                           var_goal=0.04)


savestr = (resultsfolder+'TSN'+str(shape[0]) + 'lg' + str(lam_g) +
                ('b' if binarize else '') +
                ('d' if discs else '') +
                ('t' if torus else '') +
                (str(ncomponents) if ncomponents>1 else '') +
                's'+str(sigma))
net.save(savestr+'.pickle')
net.run(50000)
net.save()