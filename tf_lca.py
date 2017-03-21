# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:53:03 2017

@author: Eric
"""

import numpy as np
import tensorflow as tf
import tf_sparsenet
import matplotlib.pyplot as plt

class LCALearner(tf_sparsenet.Sparsenet):
    
    def __init__(self,
                 data,
                 datatype="image",
                 pca=None,
                 nunits = 200,
                 batch_size = 100,
                 paramfile = None,
                 moving_avg_rate = 0.01,
                 stimshape = None,
                 lam = 0.15,
                 niter = 200,
                 infrate = 1.0,
                 learnrate = 2.0,
                 snr_goal = None,
                 threshfunc = 'hard'):
        """
        Sparse dictionary learner using the L1 or L0 locally competitive algorithm
        from Rozell et al 2008 for inference.
        
        Parameters
        data        : [nsamples, ndim] numpy array of training data
        datatype    : (str) "image" or "spectro"
        pca         : PCA object with inverse_transform(), or None if pca not used
        nunits      : (int) number of units in sparsenet model
        batch_size  : (int) number of samples in each batch for learning
        paramfile   : (str) filename for pickle file storing parameters
        moving_avg_rate: (float) rate for updating average statistics
        stimshape   : (array-like) original shape of each training datum
        lam         : (float) sparsity parameter; higher means more sparse
        niter       : (int) number of time steps in inference (not dynamically adjustable)
        infrate     : (float) gradient descent rate for inference
        learnrate   : (float) gradient descent rate for learning
        threshfunc  : (str) specifies which thresholding function to use
        """
        # save input parameters
        self.nunits = nunits
        self.batch_size = batch_size
        self.paramfile = paramfile
        self.moving_avg_rate = moving_avg_rate
        self.stimshape = stimshape or ((16,16) if datatype == 'image' else (25,256))
        self.infrate = infrate / batch_size
        self._niter = niter
        self.learnrate = learnrate
        self.threshfunc = threshfunc
        self.snr_goal = snr_goal
        
        # initialize model
        self._load_stims(data, datatype, self.stimshape, pca)
        self.build_graph(lam)
        self.initialize_stats()
        

    def acts(self, uu):    
        if self.threshfunc.startswith('hard'):
            thresholded = tf.identity
        else:
            thresholded = lambda xx: tf.add(xx, -self.thresh)

        if self.threshfunc.endswith('pos') or self.threshfunc.endswith('rec'):
            rect = tf.identity
        else:
            rect = tf.abs
    
        return tf.select(tf.greater(rect(uu), self.thresh),
                          thresholded(uu), tf.multiply(0.0,uu),
                            name='activity')

    def build_graph(self, lam):
        self.infrate = tf.Variable(self.infrate, trainable=False)
        self.learnrate = tf.Variable(self.learnrate, trainable=False)
        self.thresh = tf.Variable(lam, trainable=False)
        
        self.phi = tf.Variable(tf.random_normal([self.nunits,self.stims.datasize]))

        self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.stims.datasize])

        # LCA inference
        self.lca_drive = tf.matmul(self.phi, tf.transpose(self.X))
        self.lca_gram = (tf.matmul(self.phi, tf.transpose(self.phi)) - 
            tf.constant(np.identity(int(self.nunits)),dtype=np.float32))

        def next_u(old_u, ii):
            lca_compet = tf.matmul(self.lca_gram, self.acts(old_u))
            du = self.lca_drive - lca_compet - old_u
            return old_u + self.infrate*du

        self._itercount = tf.constant(np.arange(self.niter))
        self._infu = tf.scan(next_u, self._itercount, initializer = tf.zeros([self.nunits,self.batch_size]))
        self.u = self._infu[-1]
        
        # for testing inference
        self._infacts = self.acts(self._infu)
        def mul_fn(someacts):
            return tf.matmul(tf.transpose(someacts), self.phi)
        self._infXhat = tf.map_fn(mul_fn, self._infacts)
        self._infresid = self.X - self._infXhat
        self._infmse = tf.reduce_sum(tf.square(self._infresid), axis=[1,2])/self.batch_size/self.stims.datasize
        
        self.final_acts = self.acts(self.u)
        self.Xhat = tf.matmul(tf.transpose(self.final_acts), self.phi)
        self.resid = self.X - self.Xhat
        self.mse = tf.reduce_sum(tf.square(self.resid))/self.batch_size/self.stims.datasize
        self.meanL1 = tf.reduce_sum(tf.abs(self.final_acts))/self.batch_size
        self.loss = 0.5*self.mse #+ self.lam*self.meanL1/self.stims.datasize
        
        learner = tf.train.GradientDescentOptimizer(self.learnrate)
        learn_step = tf.Variable(0,name='learn_step', trainable=False)
        self.learn_op = learner.minimize(self.loss, global_step=learn_step, var_list=[self.phi])
        
        self.renorm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi, dim=1))

        self.snr = tf.reduce_mean(tf.square(self.X))/self.mse
        snrconvert = tf.constant(np.log(10.0)/10.0, dtype=tf.float32)
        snr_ratio = self.snr/tf.exp(snrconvert*tf.constant(self.snr_goal,dtype=tf.float32))
        self.seek_snr = self.thresh.assign(self.thresh*tf.pow(snr_ratio,0.1))
        self.snr_db = 10.0*tf.log(self.snr)/np.log(10.0)
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.renorm_phi)
        
    def train_step(self):
        feed_dict = {self.X: self.stims.rand_stim(batch_size=self.batch_size).T}
        acts, _, loss_value, mse_value, meanL1_value = self.sess.run([self.final_acts, self.learn_op, self.loss,
             self.mse, self.meanL1], feed_dict=feed_dict)
        self.sess.run(self.renorm_phi)
    
        return acts, loss_value, mse_value, meanL1_value

    def snr(self, acts, feed_dict):
        """Returns the signal-noise ratio for the given data and coefficients."""
        data = feed_dict[self.X]
        sig = np.var(data,axis=1)
        noise = np.var(data - self.sess.run(self.Xhat, feed_dict=feed_dict), axis=1)
        return np.mean(sig/noise)

    def test_inference(self):
        feed_dict = {self.X: self.stims.rand_stim(batch_size=self.batch_size).T}
        acts, costs = self.sess.run([self.final_acts, self._infmse], feed_dict=feed_dict)
        plt.plot(costs, 'b')
        print("Final SNR: " + str(self.snr(acts, feed_dict)))
        return acts, costs
        
    def get_param_list(self):
        lrnrate = self.sess.run(self.learnrate)
        irate = self.sess.run(self.infrate)
        return {'nunits' : self.nunits,
        'batch_size' : self.batch_size,
        'paramfile' : self.paramfile,
        'lam' : self.lam,
        'niter' : self.niter,
        'infrate' : irate,
        'learnrate' : lrnrate}
        
    @property
    def lam(self):
        return self.sess.run(self.thresh)
        
    @lam.setter
    def lam(self, lam):
        self.sess.run(self.thresh.assign(lam))

    @property
    def niter(self):
        return self._niter

    @niter.setter
    def niter(self, value):
        self._niter = value
        self.build_graph(self.lam)

    def feed_rand_batch(self):
        raise AttributeError
