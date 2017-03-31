# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:53:03 2017

@author: Eric
"""

import numpy as np
import tensorflow as tf
import tf_sparsenet
# workaround for cluster issue
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Failed to import matplotlib, plotting unavailable.')


class LCALearner(tf_sparsenet.Sparsenet):

    def __init__(self,
                 data,
                 datatype="image",
                 pca=None,
                 nunits=200,
                 batch_size=100,
                 paramfile=None,
                 moving_avg_rate=0.01,
                 stimshape=None,
                 lam=0.15,
                 niter=200,
                 infrate=0.1,
                 learnrate=50.0,
                 snr_goal=None,
                 seek_snr_rate=0.1,
                 threshfunc='hard'):
        """
        Sparse dictionary learner using the
        L1 or L0 locally competitive algorithm
        from Rozell et al 2008 for inference.
        Parameters
        data        : [nsamples, ndim] numpy array of training data
        datatype    : (str) "image" or "spectro"
        pca         : PCA object with inverse_transform(), or None
        nunits      : (int) number of units in sparsenet model
        batch_size  : (int) number of samples in each batch for learning
        paramfile   : (str) filename for pickle file storing parameters
        moving_avg_rate: (float) rate for updating average statistics
        stimshape   : (array-like) original shape of each training datum
        lam         : (float) sparsity parameter; higher means more sparse
        niter       : (int) number of time steps in inference (not adjustable)
        infrate     : (float) gradient descent rate for inference
        learnrate   : (float) gradient descent rate for learning
                      loss gets divided by numinput, so make this larger
        snr_goal    : (float) snr in dB, lam adjusted dynamically to match
        seek_snr_rate : (float) rate parameter for adjusting lam as above
        threshfunc  : (str) specifies which thresholding function to use
        """
        # save input parameters
        self.nunits = nunits
        self.batch_size = batch_size
        self.paramfile = paramfile
        self.moving_avg_rate = moving_avg_rate
        self.stimshape = stimshape or ((16, 16) if datatype == 'image'
                                       else (25, 256))
        self.infrate = infrate / batch_size
        self._niter = niter
        self.learnrate = learnrate
        self.threshfunc = threshfunc
        self.snr_goal = snr_goal
        self.seek_snr_rate = seek_snr_rate

        # initialize model
        self._load_stims(data, datatype, self.stimshape, pca)
        self.build_graph(lam)
        self.initialize_stats()

    def acts(self, uu, ll):
        """Computes the activation function given the internal varaiable uu
        and the current threshold parameter ll."""
        if self.threshfunc.startswith('hard'):
            thresholded = tf.identity
        else:
            thresholded = lambda xx: tf.add(xx, -tf.sign(xx)*ll)

        if self.threshfunc.endswith('pos') or self.threshfunc.endswith('rec'):
            rect = tf.identity
        else:
            rect = tf.abs
    
        return tf.select(tf.greater(rect(uu), ll),
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

        def next_u(old_u_l, ii):
            old_u = old_u_l[0]
            ll = old_u_l[1]
            lca_compet = tf.matmul(self.lca_gram, self.acts(old_u, ll))
            du = self.lca_drive - lca_compet - old_u
            new_l = tf.constant(0.98)*ll
            new_l = tf.select(tf.greater(new_l,self.thresh), 
                                new_l,
                                self.thresh*np.ones(self.batch_size))
            return (old_u + self.infrate*du, new_l)

        self._itercount = tf.constant(np.arange(self.niter))
        init_u_l = (tf.zeros([self.nunits,self.batch_size]),
                     0.5*tf.reduce_max(tf.abs(self.lca_drive),axis=0))
        self._infu = tf.scan(next_u, self._itercount, initializer = init_u_l)[0]
        self.u = self._infu[-1]
        
        # for testing inference
        self._infacts = self.acts(self._infu, self.thresh)

        def mul_fn(someacts):
            return tf.matmul(tf.transpose(someacts), self.phi)
        self._infXhat = tf.map_fn(mul_fn, self._infacts)
        self._infresid = self.X - self._infXhat
        self._infmse = tf.reduce_sum(tf.square(self._infresid), axis=[1,2])/self.batch_size/self.stims.datasize
        
        self.final_acts = self.acts(self.u, self.thresh)
        self.Xhat = tf.matmul(tf.transpose(self.final_acts), self.phi)
        self.resid = self.X - self.Xhat
        self.mse = tf.reduce_sum(tf.square(self.resid))/self.batch_size/self.stims.datasize
        self.meanL1 = tf.reduce_sum(tf.abs(self.final_acts))/self.batch_size
        self.loss = 0.5*self.mse #+ self.lam*self.meanL1/self.stims.datasize
        
        #learner = tf.train.AdadeltaOptimizer(self.learnrate)
        learner = tf.train.GradientDescentOptimizer(self.learnrate)
        learn_step = tf.Variable(0,name='learn_step', trainable=False)
        self.learn_op = learner.minimize(self.loss, global_step=learn_step, var_list=[self.phi])
        
        self.renorm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi, dim=1))

        self.snr = tf.reduce_mean(tf.square(self.X))/self.mse
        if self.snr_goal is not None:
            snrconvert = tf.constant(np.log(10.0)/10.0, dtype=tf.float32)
            snr_ratio = self.snr/tf.exp(snrconvert*tf.constant(self.snr_goal,dtype=tf.float32))
            self.seek_snr = self.thresh.assign(self.thresh*tf.pow(snr_ratio,self.seek_snr_rate))
        self.snr_db = 10.0*tf.log(self.snr)/np.log(10.0)
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.renorm_phi)
        
    def train_step(self):
        feed_dict = {self.X: self.stims.rand_stim(batch_size=self.batch_size).T}
        if self.snr_goal is None:
            op_list = [self.final_acts, self.learn_op, self.loss, self.mse, self.meanL1]
            acts, _, loss_value, mse_value, meanL1_value = self.sess.run(op_list, feed_dict=feed_dict)
        else:
            op_list = [self.final_acts, self.learn_op, self.loss, self.mse, self.meanL1, self.seek_snr]
            acts, _, loss_value, mse_value, meanL1_value,_ = self.sess.run(op_list, feed_dict=feed_dict)

        self.sess.run(self.renorm_phi)

        return acts, loss_value, mse_value, meanL1_value

    def run(self, ntrials=10000):
        for tt in range(ntrials):
            self.store_stats(*self.train_step())
            if tt % 50 == 0:
                print(tt)
                if (tt % 1000 == 0 or tt+1 == ntrials) and tt != 0:
                    try:
                        print("Saving progress to " + self.paramfile)
                        self.save()
                    except (ValueError, TypeError) as er:
                        print('Failed to save parameters. ', er)

    def test_inference(self):
        feed_dict = {self.X: self.stims.rand_stim(batch_size=self.batch_size).T}
        acts, costs, snr = self.sess.run([self.final_acts,
                                          self._infmse,
                                          self.snr_db],
                                         feed_dict=feed_dict)
        plt.plot(costs, 'b')
        print("Final SNR: " + str(snr))
        return acts, costs

    def get_param_list(self):
        lrnrate = self.sess.run(self.learnrate)
        irate = self.sess.run(self.infrate)
        return {'nunits': self.nunits,
                'batch_size': self.batch_size,
                'paramfile': self.paramfile,
                'lam': self.lam,
                'niter': self.niter,
                'infrate': irate,
                'learnrate': lrnrate}

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
