import tensorflow as tf
import tf_sparsenet
import numpy as np

class TopoSparsenet(tf_sparsenet.Sparsenet):
	
	def __init__(self, data, datatype="image", pca=None,  **kwargs):
		self.lam_g = 0.1
		self.epsilon = 0.0001 # to regularize derivative of square root
		self.sigma = 1.0
		self.dict_shape = (30,30)

		try:
			kwargs['lam']
			super().__init__(data, datatype = datatype, pca = pca, **kwargs)
		except KeyError:
			super().__init__(data, datatype = datatype, pca = pca, lam=0, **kwargs)

	def initialize_graph(self):
		self.nunits = int(np.prod(self.dict_shape))
		self.g = tf.constant(self.layer_two_weights(), dtype=tf.float32)

		self.infrate = tf.Variable(self.infrate, trainable=False)
		self.learnrate = tf.Variable(self.learnrate, trainable=False)

		self.phi = tf.Variable(tf.random_normal([self.nunits,self.stims.datasize]))
		self.acts = tf.Variable(tf.zeros([self.nunits,self.batch_size]))

		self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.stims.datasize])
		self.Xhat = tf.matmul(tf.transpose(self.acts), self.phi)
		self.resid = self.X - self.Xhat
		self.mse = tf.reduce_sum(tf.square(self.resid))/self.batch_size/self.stims.datasize
		self.meanL1 = tf.reduce_sum(tf.abs(self.acts))/self.batch_size
		self.layer2 = tf.reduce_sum(tf.sqrt(tf.matmul(self.g,tf.square(self.acts)) + self.epsilon))/self.batch_size
		self.loss = 0.5*self.mse + (self.lam*self.meanL1 + self.lam_g*self.layer2)/self.stims.datasize

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

	def distance(self, i, j):
		""" This function measures the distance between element i and j. The distance 
		here is the distance between element i and j once the row vector has been 
		reshaped into a square matrix, treating the dictionary as a torus globally."""

		rows, cols = self.dict_shape
		rowi = i // cols
		coli = i % cols
		rowj = j // cols
		colj = j % cols

		# global topology is a torus
		rowj = [rowj - rows, rowj, rowj + rows]
		colj = [colj - cols, colj, colj + cols]

		dist = []
		for r in rowj:
			for c in colj:
				dist.append((rowi - r)**2 + (coli - c)**2)

		return np.min(dist)

	def layer_two_weights(self):
		"""This is currently only working for the case when (# of layer 2
		units) = (# of layer 1 units) """

		g = np.zeros((self.nunits, self.nunits))

		sigsquared = self.sigma**2
		for i in range(self.nunits):
			for j in range(self.nunits):
				g[i, j] = np.exp(-self.distance(i, j)/(2 * sigsquared))

		return g

	def block_membership(self, i, j, width=5):
		"""This returns 1 if j is in the ith block, otherwise 0. Currently only
		works for square dictionaries."""
		# TODO: I think there's a bug here that makes the boundary conditions
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
		self.g = np.zeros_like(self.g)
		nunits = np.prod(self.dict_shape)
		for i in range(nunits):
			for j in range(nunits):
				self.g[i, j] = self.block_membership(i, j, width)

	def binarize_g(self, thresh=1/2, width=None):
		if width is not None:
			thresh = np.exp(-width**2/(2*self.sigma**2))
		self.g = np.array(self.g >= thresh, dtype=int)

	def show_dict(self, stimset=None, cmap='RdBu', subset=None, layout=None, savestr=None):
		layout = layout or self.dict_shape
		super().show_dict(stimset, cmap, subset, layout, savestr)

	def sort(self, *args, **kwargs):
		print("The topographic order is meaningful, don't sort it away!") 