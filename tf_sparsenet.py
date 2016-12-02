# -*- coding: utf-8 -*-

import DictLearner
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sparsenet

class Sparsenet(sparsenet.Sparsenet):
	"""A sparse dictionary learner based on (Olshausen and Field, 1996). Uses a tensorflow backend."""

	def __init__(self, data, datatype="image", pca=None, **kwargs):

		self.nunits = 200
		self.batch_size = 100
		self.paramfile = None
		self.moving_avg_rate = 0.01
		self.stimshape = (16,16) if datatype == 'image' else (25,256)

		self.lam = 0.15
		
		self.niter = 200
		self.var_goal = 0.1 if datatype == 'image' else 0.033
		self.var_avg_rate = 0.1
		self.gain_rate = 0.001

		self.infrate = 200.0
		self.learnrate = 2.0

		for key, val in kwargs.items():
			try:
				getattr(self,key)
			except AttributeError:
				print('Unexpected parameter passed:' + key)
			setattr(self, key, val)

		self._load_stims(data, datatype, self.stimshape, pca)
		self.initialize_graph()
		self.initialize_stats()

	def initialize_stats(self):
		self.loss_history = np.array([])
		self.mse_history = np.array([])
		self.L1_history = np.array([])

	def store_stats(self, loss_value, mse_value, meanL1_value):
		self.loss_history = np.append(self.loss_history, loss_value)
		self.mse_history = np.append(self.mse_history, mse_value)
		self.L1_history = np.append(self.L1_history, meanL1_value/self.nunits)

	def initialize_graph(self):
		self.infrate = tf.Variable(self.infrate, trainable=False)
		self.learnrate = tf.Variable(self.learnrate, trainable=False)

		self.phi = tf.Variable(tf.random_normal([self.nunits,self.stims.datasize]))
		self.acts = tf.Variable(tf.zeros([self.nunits,self.batch_size]))

		self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.stims.datasize])
		self.Xhat = tf.matmul(tf.transpose(self.acts), self.phi)
		self.resid = self.X - self.Xhat
		self.mse = tf.reduce_sum(tf.square(self.resid))/self.batch_size/self.stims.datasize
		self.meanL1 = tf.reduce_sum(tf.abs(self.acts))/self.batch_size
		self.loss = 0.5*self.mse + self.lam*self.meanL1/self.stims.datasize

		inferer = tf.train.GradientDescentOptimizer(self.infrate)
		inf_step = tf.Variable(0, name='inf_step', trainable=False)
		self.inf_op = inferer.minimize(self.loss, global_step=inf_step, var_list=[self.acts])

		learner = tf.train.GradientDescentOptimizer(self.learnrate)
		learn_step = tf.Variable(0,name='learn_step', trainable=False)
		self.learn_op = learner.minimize(self.loss, global_step=learn_step, var_list=[self.phi])

		self.ma_variances = tf.Variable(tf.ones(self.nunits), trainable=False)
		self.gains = tf.Variable(tf.ones(self.nunits), trainable=False)
		_, self.variances = tf.nn.moments(self.acts, axes=[1])
		self.update_variance = self.ma_variances.assign((1.-self.var_avg_rate)*self.ma_variances + self.var_avg_rate*self.variances)
		self.update_gains = self.gains.assign(self.gains*tf.pow(self.var_goal/self.ma_variances, self.gain_rate))
		self.renorm_phi = self.phi.assign((tf.expand_dims(self.gains,dim=1)*tf.nn.l2_normalize(self.phi, dim=1)))

		self.sess = tf.Session()

		self.sess.run(tf.initialize_all_variables())
		self.sess.run(self.phi.assign(tf.nn.l2_normalize(self.phi, dim=1)))

	def train_step(self):
		feed_dict = {self.X: self.stims.rand_stim(batch_size=self.batch_size).T}
		self.sess.run(self.acts.assign(tf.zeros([self.nunits, self.batch_size])))
		for ii in range(self.niter):
			self.sess.run([self.inf_op, self.loss], feed_dict=feed_dict)

		_, loss_value, mse_value, meanL1_value = self.sess.run([self.learn_op, self.loss, self.mse, self.meanL1], feed_dict=feed_dict)

		self.sess.run(self.update_variance)
		self.sess.run(self.update_gains)
		self.sess.run(self.renorm_phi)

		return loss_value, mse_value, meanL1_value

	def run(self, nbatches=1000):
		for tt in range(nbatches):
			self.store_stats(*self.train_step())
			if tt % 50 == 0:
				print (tt)
				if (tt % 1000 == 0 or tt+1 == nbatches) and tt!= 0:
					try:
						print ("Saving progress to " + self.paramfile)
						self.save()
					except (ValueError, TypeError) as er:
						print ('Failed to save parameters. ', er)

	def show_dict(self, cmap='RdBu', subset=None, layout='sqrt', savestr=None):
		"""Plot an array of tiled dictionary elements. The 0th element is in the top right."""
		Qs = self.sess.run(self.phi)
		if subset is not None:
			indices = np.random.choice(self.nunits, subset)
			Qs = Qs[np.sort(indices)]
		array = self.stims.stimarray(Qs[::-1], layout=layout)
		plt.figure()
		arrayplot = plt.imshow(array,interpolation='nearest', cmap=cmap, aspect='auto', origin='lower')
		plt.axis('off')
		plt.colorbar()
		if savestr is not None:
			plt.savefig(savestr, bbox_inches='tight')
		return arrayplot

	def show_element(self, index, cmap='jet', labels=None, savestr=None):
		elem = self.stims.stim_for_display(self.sess.run(self.phi)[index])
		plt.figure()
		plt.imshow(elem.T, interpolation='nearest',cmap=cmap, aspect='auto', origin='lower')
		if labels is None:
			plt.axis('off')
		else:
			plt.colorbar()
		if savestr is not None:
			plt.savefig(savestr, bbox_inches='tight')

	def test_inference(self):
		feed_dict = {self.X: self.stims.rand_stim(batch_size=self.batch_size).T}
		costs = np.zeros(self.niter)
		for ii in range(self.niter):
			_, costs[ii] = self.sess.run([self.inf_op, self.loss] , feed_dict=feed_dict)
		plt.plot(costs)
		print("Final SNR: " + str(self.snr(feed_dict)))

	def snr(self, feed_dict):
		"""Returns the signal-noise ratio for the given data and current coefficients."""
		data = feed_dict[self.X]
		sig = np.var(data,axis=0)
		noise = np.var(data - self.sess.run(self.Xhat, feed_dict=feed_dict), axis=0)
		return np.mean(sig/noise)

	def progress_plot(self, window_size=1000, norm=1, start=0, end=-1):
		"""Plots a moving average of the error and activity history with the given averaging window."""
		window = np.ones(int(window_size))/float(window_size)
		smoothederror = np.convolve(self.mse_history[start:end], window, 'valid')
		smoothedactivity = np.convolve(self.L1_history[start:end], window, 'valid')
		plt.plot(smoothederror, 'b', smoothedactivity, 'g')

	def set_infrate(self, infrate):
		self.sess.run(self.infrate.assign(infrate))

	def set_learnrate(self, learnrate):
		self.sess.run(self.learnrate.assign(learnrate))

	def adjust_rates(self, factor):
		self.set_infrate(self.sess.run(self.infrate))
		self.set_learnrate(self.sess.run(self.learnrate))

	def sort_dict(self, batch_size=None, plot = False, allstims = True, savestr=None):
		raise NotImplementedError

	def fast_sort(self, L1=False, plot=False, savestr=None):
		raise NotImplementedError

	def sort(self, usages, sorter, plot=False, savestr=None):
		raise NotImplementedError

	def get_param_list(self):
		return {'nunits' : self.nunits,
		'batch_size' : self.batch_size,
		'paramfile' : self.paramfile,
		'lam' : self.lam,
		'niter' : self.niter,
		'var_goal' : self.var_goal,
		'var_avg_rate' : self.var_avg_rate,
		'gain_rate' : self.gain_rate,
		'infrate' : self.infrate,
		'learnrate' : self.learnrate}

	def set_params(self, params):
		for key, val in params.items():
			if key == 'infrate':
				self.sess.run(self.infrate.assign(val))
			elif key == 'learnrate':
				self.sess.run(self.learnrate.assign(val))
			else:
				try:
					getattr(self,key)
				except AttributeError:
					print('Unexpected parameter passed: ' + key)
				setattr(self, key, val)

	def get_histories(self):
		return {'loss' : self.loss_history,
		'mse': self.mse_history,
		'L1' : self.L1_history}

	def set_histories(self, histories):
		self.loss_history = histories['loss']
		self.mse_history = histories['mse']
		self.L1_history = histories['L1']

	@property
	def Q(self):
		"""Q is just a different notation for the dictionary. Used for compatability with DictLearner methods."""
		return self.sess.run(self.phi)

	@Q.setter
	def Q(self, value):
		self.sess.run(self.phi.assign(value))