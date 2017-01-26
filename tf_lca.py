# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:53:03 2017

@author: Eric
"""

import numpy as np
import tensorflow as tf
import tf_sparsenet

# TODO: test everything

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
                 infrate = 200.0,
                 learnrate = 2.0):
        """
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
        niter       : (int) number of time steps in inference
        infrate     : (float) gradient descent rate for inference
        learnrate   : (float) gradient descent rate for learning
        """
        # save input parameters
        self.nunits = nunits
        self.batch_size = batch_size
        self.paramfile = paramfile
        self.moving_avg_rate = moving_avg_rate
        self.stimshape = stimshape or ((16,16) if datatype == 'image' else (25,256))
        self.lam = lam
        self.niter = niter
        self.infrate = infrate
        self.learnrate = learnrate
        
        # initialize model
        self._load_stims(data, datatype, self.stimshape, pca)
        self.build_graph()
        self.initialize_stats()
        
    def build_graph(self):
        self.infrate = tf.Variable(self.infrate, trainable=False)
        self.learnrate = tf.Variable(self.learnrate, trainable=False)
        
        self.phi = tf.Variable(tf.random_normal([self.nunits,self.stims.datasize]))
        self.u = tf.Variable(tf.zeros([self.nunits,self.batch_size]),
                             trainable=False,
                             name='internal variables')
        self.reset_u = self.u.assign(tf.zeros([self.nunits,self.batch_size]))
        
        self.acts = tf.select(tf.greater(tf.abs(self.u), self.lam),
                              self.u, tf.zeros([self.nunits, self.batch_size]),
                                name='activity')
        
        self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.stims.datasize])
        self.Xhat = tf.matmul(tf.transpose(self.acts), self.phi)
        self.resid = self.X - self.Xhat
        self.mse = tf.reduce_sum(tf.square(self.resid))/self.batch_size/self.stims.datasize
        self.meanL1 = tf.reduce_sum(tf.abs(self.acts))/self.batch_size
        self.loss = 0.5*self.mse + self.lam*self.meanL1/self.stims.datasize
        
        # LCA inference
        self.lca_drive = tf.matmul(self.phi, tf.transpose(self.X))
        self.lca_gram = (tf.matmul(self.phi, tf.transpose(self.phi)) - 
            tf.constant(np.identity(int(self.nunits)),dtype=np.float32))
        self.lca_compet = tf.matmul(self.lca_gram, self.acts)
        self.du = self.lca_drive - self.lca_compet - self.u
        self.inf_op = tf.group(self.u.assign_add(self.infrate * self.du)) # why group? copied from Dylan
        
        learner = tf.train.GradientDescentOptimizer(self.learnrate)
        learn_step = tf.Variable(0,name='learn_step', trainable=False)
        self.learn_op = learner.minimize(self.loss, global_step=learn_step, var_list=[self.phi])
        
        self.renorm_phi = tf.nn.l2_normalize(self.phi, dim=1)
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.renorm_phi)
        
    def train_step(self):
        feed_dict = {self.X: self.stims.rand_stim(batch_size=self.batch_size).T}
        self.sess.run(self.reset_u)
        for ii in range(self.niter):
            self.sess.run([self.inf_op], feed_dict=feed_dict)
        
        _, loss_value, mse_value, meanL1_value = self.sess.run([self.learn_op, self.loss, self.mse, self.meanL1], feed_dict=feed_dict)
        
        self.sess.run(self.renorm_phi)
    
        return loss_value, mse_value, meanL1_value
        
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