# -*- coding: utf-8 -*-

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    print("Can't import matplotlib.")
import tensorflow as tf
import sparsenet


class Sparsenet(sparsenet.Sparsenet):
    """
    A sparse dictionary learner based on (Olshausen and Field, 1996).
    Uses a tensorflow backend.
    """

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
                 var_goal=0.04,
                 var_avg_rate=0.1,
                 gain_rate=0.01,
                 infrate=0.1,
                 learnrate=2.0):
        """
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
        niter       : (int) number of time steps in inference
        var_goal    : (float) target variance of activities
        var_avg_rate: (float) rate for updating moving avg activity variance
        gain_rate   : (float) rate for updating gains to fix activity variance
        infrate     : (float) gradient descent rate for inference
        learnrate   : (float) gradient descent rate for learning
        """
        # save input parameters
        self.nunits = nunits
        self.batch_size = batch_size
        self.paramfile = paramfile
        self.moving_avg_rate = moving_avg_rate
        self.stimshape = stimshape or ((16, 16) if datatype == 'image'
                                       else (25, 256))
        self.lam = lam
        self.niter = niter
        self.var_goal = var_goal
        self.var_avg_rate = var_avg_rate
        self.gain_rate = gain_rate
        self.infrate = infrate
        self.learnrate = learnrate

        # initialize model
        self._load_stims(data, datatype, self.stimshape, pca)
        self.Q = tf.random_normal([self.nunits, self.stims.datasize])
        self.ma_variances = tf.ones(self.nunits)
        self.gains = tf.ones(self.nunits)
        self.graph = self.build_graph()
        self.initialize_stats()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.config = tf.ConfigProto(gpu_options=gpu_options)
        self.config.gpu_options.allow_growth = True
        with tf.Session(graph=self.graph, config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            self.Q = sess.run(self.phi.assign(tf.nn.l2_normalize(self.phi,
                                                                 dim=1)))

    def initialize_stats(self):
        self.loss_history = np.array([])
        self.mse_history = np.array([])
        self.L1_history = np.array([])
        nunits = self.nunits
        self.L0acts = np.zeros(nunits)
        self.L1acts = np.zeros(nunits)
        self.L2acts = np.zeros(nunits)
        self.meanacts = np.zeros_like(self.L0acts)

    def store_stats(self, acts, loss_value, mse_value, meanL1_value):
        eta = self.moving_avg_rate
        self.loss_history = np.append(self.loss_history, loss_value)
        self.mse_history = np.append(self.mse_history, mse_value)
        self.L1_history = np.append(self.L1_history, meanL1_value/self.nunits)
        self.L2acts = (1-eta)*self.L2acts + eta*(acts**2).mean(1)
        self.L1acts = (1-eta)*self.L1acts + eta*np.abs(acts).mean(1)
        L0means = np.mean(acts != 0, axis=1)
        self.L0acts = (1-eta)*self.L0acts + eta*L0means
        means = acts.mean(1)
        self.meanacts = (1-eta)*self.meanacts + eta*means

    def build_graph(self):
        graph = tf.get_default_graph()

        self._infrate = tf.Variable(self.infrate, trainable=False)
        self._learnrate = tf.Variable(self.learnrate, trainable=False)

        self.phi = tf.Variable(self.Q)
        self.acts = tf.Variable(tf.zeros([self.nunits, self.batch_size]))
        self.reset_acts = self.acts.assign(tf.zeros([self.nunits,
                                                     self.batch_size]))

        self.x = tf.Variable(tf.zeros([self.batch_size, self.stims.datasize]),
                             trainable=False)
        self.xhat = tf.matmul(tf.transpose(self.acts), self.phi)
        self.resid = self.x - self.xhat
        self.mse = tf.reduce_sum(tf.square(self.resid))
        self.mse = self.mse/self.batch_size/self.stims.datasize
        self.meanL1 = tf.reduce_sum(tf.abs(self.acts))/self.batch_size
        self.loss = 0.5*self.mse + self.lam*self.meanL1/self.stims.datasize

        self.snr = tf.reduce_mean(tf.square(self.x))/self.mse
        self.snr_db = 10.0*tf.log(self.snr)/np.log(10.0)

        inffactor = self.batch_size*self.stims.datasize
        inferer = tf.train.GradientDescentOptimizer(self._infrate*inffactor)
        self.inf_op = inferer.minimize(self.loss, var_list=[self.acts])

        learner = tf.train.GradientDescentOptimizer(self._learnrate)
        self.learn_op = learner.minimize(self.loss, var_list=[self.phi])

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

        return graph

    def train_step(self, sess):
        sess.run(self.x.assign(self.get_batch()))
        sess.run(self.reset_acts)
        for ii in range(self.niter):
            sess.run([self.inf_op, self.loss])
        oplist = [self.learn_op, self.loss, self.mse, self.meanL1]
        _, loss_value, mse_value, meanL1_value = sess.run(oplist)

        sess.run(self.update_variance)
        sess.run(self.update_gains)
        sess.run(self.renorm_phi)

        return sess.run(self.acts), loss_value, mse_value, meanL1_value

    def initialize_vars(self, sess):
        """Initializes values of tf Variables."""
        sess.run(tf.global_variables_initializer())
        sess.run([self.phi.assign(self.Q),
                  self._infrate.assign(self.infrate),
                  self._learnrate.assign(self.learnrate),
                  self._ma_variances.assign(self.ma_variances),
                  self._gains.assign(self.gains)])

    def retrieve_vars(self, sess):
        """Retrieve values from tf graph."""
        stuff = sess.run([self.phi,
                          self._infrate,
                          self._learnrate,
                          self._ma_variances,
                          self._gains])
        (self.Q, self.infrate,
         self.learnrate, self.ma_variances, self.gains) = stuff

    def run(self, nbatches=1000):
        with tf.Session(config=self.config, graph=self.graph) as sess:
            self.initialize_vars(sess)
            for tt in range(nbatches):
                self.store_stats(*self.train_step(sess))
                if tt % 50 == 0:
                    print(tt)
                    self.retrieve_vars(sess)
                    if (tt % 1000 == 0 or tt+1 == nbatches) and tt != 0:
                        try:
                            print("Saving progress to " + self.paramfile)
                            self.save()
                        except (ValueError, TypeError) as er:
                            print('Failed to save parameters. ', er)
            self.retrieve_vars(sess)

    def show_dict(self, cmap='RdBu', subset=None, layout='sqrt', savestr=None):
        """Plot an array of tiled dictionary elements.
        The 0th element is in the top right."""
        if subset is not None:
            indices = np.random.choice(self.nunits, subset)
            Qs = self.Q[np.sort(indices)]
        else:
            Qs = self.Q
        array = self.stims.stimarray(Qs[::-1], layout=layout)
        plt.figure()
        arrayplot = plt.imshow(array, interpolation='nearest', cmap=cmap,
                               aspect='auto', origin='lower')
        plt.axis('off')
        plt.colorbar()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')
        return arrayplot

    def show_element(self, index, cmap='jet', labels=None, savestr=None):
        elem = self.stims.stim_for_display(self.Q[index])
        plt.figure()
        plt.imshow(elem.T, interpolation='nearest', cmap=cmap,
                   aspect='auto', origin='lower')
        if labels is None:
            plt.axis('off')
        else:
            plt.colorbar()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')

    def test_inference(self):
        costs = np.zeros(self.niter)
        with tf.Session(config=self.config, graph=self.graph) as sess:
            self.initialize_vars(sess)
            sess.run(self.x.assign(self.get_batch()))
            sess.run(self.reset_acts)
            for ii in range(self.niter):
                _, costs[ii] = sess.run([self.inf_op, self.loss])
            plt.plot(costs, 'b')
            print("Final SNR: " + str(sess.run(self.snr_db)))
            finalacts = sess.run(self.acts)
        return (finalacts, costs)

    def get_batch(self):
        return self.stims.rand_stim(batch_size=self.batch_size).T

    def progress_plot(self, window_size=1000, norm=1, start=0, end=-1):
        """Plots a moving average of the error and activity history with
        the given averaging window."""
        window = np.ones(int(window_size))/float(window_size)
        smoothederror = np.convolve(self.mse_history[start:end], window,
                                    'valid')
        smoothedactivity = np.convolve(self.L1_history[start:end], window,
                                       'valid')
        plt.plot(smoothederror, 'b', smoothedactivity, 'g')

    def adjust_rates(self, factor):
        self.infrate = factor*self.infrate
        self.learnrate = factor*self.learnrate

    def sort_dict(self, **kwargs):
        raise NotImplementedError

    def sort(self, usages, sorter, plot=False, savestr=None):
        self.Q = self.Q[sorter]
        self.L0acts = self.L0acts[sorter]
        self.L1acts = self.L1acts[sorter]
        self.L2acts = self.L2acts[sorter]
        self.meanacts = self.meanacts[sorter]
        if plot:
            plt.figure()
            plt.plot(usages[sorter])
            plt.title('L0 Usage')
            plt.xlabel('Dictionary index')
            plt.ylabel('Fraction of stimuli')
            if savestr is not None:
                plt.savefig(savestr, format='png', bbox_inches='tight')

    def get_param_list(self):
        return {'nunits': self.nunits,
                'batch_size': self.batch_size,
                'paramfile': self.paramfile,
                'lam': self.lam,
                'niter': self.niter,
                'var_goal': self.var_goal,
                'var_avg_rate': self.var_avg_rate,
                'gain_rate': self.gain_rate,
                'infrate': self.infrate,
                'learnrate': self.learnrate,
                'gains': self.gains,
                'ma_variances': self.ma_variances}

    def set_params(self, params):
        for key, val in params.items():
            try:
                getattr(self, key)
            except AttributeError:
                print('Unexpected parameter passed: ' + key)
                setattr(self, key, val)

    def get_histories(self):
        return {'loss': self.loss_history,
                'mse': self.mse_history,
                'L1': self.L1_history,
                'L0acts': self.L0acts,
                'L1acts': self.L1acts,
                'L2acts': self.L2acts,
                'meanacts': self.meanacts}

    def set_histories(self, histories):
        self.loss_history = histories['loss']
        self.mse_history = histories['mse']
        self.L1_history = histories['L1']
        self.L0acts = histories['L0acts']
        self.L1acts = histories['L1acts']
        self.L2acts = histories['L2acts']
        self.meanacts = histories['meanacts']
