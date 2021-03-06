# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from tf_sparsenet import Sparsenet as snet
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Can't import matplotlib. No plotting.")


def block_diag(*arrs):
    """
    copied from scipy.linalg.block_diag to avoid scipy dependency because long story
    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


class TopoSparsenet(snet):
    """Topographic Sparsenet with TensorFlow backend
     and a few methods for defining topologies."""

    def __init__(self, data, datatype="image", pca=None,
                 dict_shape=(30, 30), topo=None, lam_g=0.1,
                 **kwargs):
        """
        Topographic Sparsenet inherits from Sparsenet. Its unique
        attributes give the dictionary a shape and define the relative
        weight of the topographic term in the cost function.
        The topology matrix g is defined by the topology object topo.

        Args:
        lam_g : float, defines weight of topography term
        dict_shape : tuple (len, wid) of ints specifying shape of dictionary
        """
        self.lam_g = lam_g
        self.epsilon = 0.0001  # to regularize derivative of square root
        self.dict_shape = dict_shape
        nunits = int(np.prod(self.dict_shape))
        self.topo = topo or topology((nunits, nunits))
        nunits = self.topo.ncomponents * nunits
        try:
            kwargs['lam']
            snet.__init__(self, data, nunits=nunits, datatype=datatype, pca=pca, **kwargs)
        except KeyError:
            snet.__init__(self, data, nunits=nunits, datatype=datatype, pca=pca, lam=0, **kwargs)

    def build_graph(self):
        graph = tf.get_default_graph()

        self.g = tf.constant(self.topo.get_matrix(), dtype=tf.float32)

        self._infrate = tf.Variable(self.infrate, trainable=False)
        self._learnrate = tf.Variable(self.learnrate, trainable=False)

        self.phi = tf.Variable(self.Q)
        self.acts = tf.Variable(tf.zeros([self.nunits, self.batch_size]))
        self.reset_acts = self.acts.assign(tf.zeros([self.nunits, self.batch_size]))

        self.x = tf.Variable(tf.zeros([self.batch_size, self.stims.datasize]), trainable=False)
        self.xhat = tf.matmul(tf.transpose(self.acts), self.phi, name='xhat')
        self.resid = self.x - self.xhat
        self.mse = tf.reduce_sum(tf.square(self.resid))/self.batch_size/self.stims.datasize
        self.meanL1 = tf.reduce_sum(tf.abs(self.acts))/self.batch_size
        self.layer2 = tf.reduce_sum(tf.sqrt(tf.matmul(self.g, tf.square(self.acts),
            name='g_times_acts') + self.epsilon))/self.batch_size
        self.loss = 0.5*self.mse + (self.lam*self.meanL1 + self.lam_g*self.layer2)/self.stims.datasize

        inffactor = self.batch_size*self.stims.datasize
        inferer = tf.train.GradientDescentOptimizer(self._infrate*inffactor)
        self.inf_op = inferer.minimize(self.loss, var_list=[self.acts])

        learner = tf.train.GradientDescentOptimizer(self.learnrate)
        learn_step = tf.Variable(0,name='learn_step', trainable=False)
        self.learn_op = learner.minimize(self.loss, global_step=learn_step, var_list=[self.phi])

        self._ma_variances = tf.Variable(self.ma_variances, trainable=False)
        self._gains = tf.Variable(self.gains, trainable=False)
        _, self.variances = tf.nn.moments(self.acts, axes=[1])
        vareta = self.var_avg_rate
        newvar = (1.-vareta)*self._ma_variances + vareta*self.variances
        self.update_variance = self._ma_variances.assign(newvar)
        newgain = self.gains*tf.pow(self.var_goal/self._ma_variances,
                                    self.gain_rate)
        self.update_gains = self._gains.assign(newgain)
        normphi = (tf.expand_dims(self._gains,
                                  dim=1)*tf.nn.l2_normalize(self.phi, dim=1))
        self.renorm_phi = self.phi.assign(normphi)

        self._init_op = tf.global_variables_initializer()

        return graph

    def show_dict(self, cmap='RdBu_r', layout=None, savestr=None):
        Qs = self.Q
        layout = layout or self.dict_shape
        ncomp = self.topo.ncomponents
        per_comp = np.prod(layout)
        nn = 0
        display = self.stims.stimarray(Qs[nn*per_comp:(nn+1)*per_comp], layout=layout)
        for nn in range(1,ncomp):
            display = np.concatenate([display, self.stims.stimarray(Qs[nn*per_comp:(nn+1)*per_comp], layout=layout)],
                                        axis=0)
        plt.figure()
        arrayplot = plt.imshow(display, interpolation='nearest', cmap=cmap, aspect='auto', origin='lower')
        plt.axis('off')
        plt.colorbar()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')
        return display

    def sort(self, *args, **kwargs):
        print("The topographic order is meaningful, don't sort it away!")

    def get_param_list(self):
        params = snet.get_param_list(self)
        params['lam_g'] = self.lam_g
        return params


class topology():
    def __init__(self, shape, discs=True, torus=True, binary=True, sigma = 1.0, ncomponents = 1):
        """
        shape: (tuple) (nlayer2comp, nlayer1) shape of each component
        sigma : (float) defines stdev of default gaussian neighborhoods
        """
        self.shape = shape
        dict_side = int(np.sqrt(self.shape[1]))
        assert dict_side**2 == self.shape[1], 'Only square dictionaries supported.'
        self.dict_shape = (dict_side, dict_side)
        self.discs = discs
        self.torus = torus
        self.binary = binary
        self.sigma = sigma
        self.ncomponents = ncomponents

    def get_matrix(self):
        
        g = np.zeros(self.shape)
        
        if self.discs:
            g = self.make_discs(g, *self.shape)

        if self.ncomponents > 1:
            blocks = [g.copy() for ii in range(self.ncomponents)]
            g = block_diag(*blocks)

        if self.binary:
            g = self.binarize(g)
        
        return g



    def make_discs(self, g, nlayer2, nlayer1):
        sigsquared = self.sigma**2
        for i in range(nlayer2):
            for j in range(nlayer1):
                g[i, j] = np.exp(-self.distance(i, j)/(2 * sigsquared))
        return g

    def distance(self, i, j):
        """ This function measures the squared distance between element i and j. The distance 
        here is the distance between element i and j once the row vector has been 
        reshaped into a square matrix, treating the dictionary as a torus globally
        if torus is True."""
        
        rows, cols = self.dict_shape
        rowi = i // cols
        coli = i % cols
        rowj = j // cols
        colj = j % cols
        
        if self.torus:
            # global topology is a torus
            rowj = [rowj - rows, rowj, rowj + rows]
            colj = [colj - cols, colj, colj + cols]
            
            dist = []
            for r in rowj:
                for c in colj:
                    dist.append((rowi - r)**2 + (coli - c)**2)
            
            return np.min(dist)
        else:
            return (rowi - rowj)**2 + (coli - colj)**2

    def block_membership(self, i, j, width=5):
        """This returns 1 if j is in the ith block, otherwise 0. Currently only
        works for square dictionaries."""
        # FIXME: I think there's a bug here that makes the boundary conditions
        # and the sizes wrong
        size = self.dict_shape[0]
        if size != self.dict_shape[1]:
            raise NotImplementedError
        i = [i // size, i % size]
        j = [j // size, j % size]
        
        if (abs((i[0]%size)-(j[0]%size)) % (size-1) < width)  and (abs((i[1]%size)-(j[1]%size)) % (size-1) < width):
            return 1
        else:
            return 0

    def set_blocks(self, width=5):
        """Change the topography by making each second layer unit respond to
        a square block of layer one with given width. g becomes binary."""
        # FIXME: doesn't work because block_membership doesn't work
        self.g = np.zeros_like(self.g)
        nunits = np.prod(self.dict_shape)
        for i in range(nunits):
            for j in range(nunits):
                self.g[i, j] = self.block_membership(i, j, width)

    def binarize(self, g, thresh=1/2, width=None):
        if width is not None:
            thresh = np.exp(-width**2/(2*self.sigma**2))
        return np.array(g >= thresh, dtype=int)