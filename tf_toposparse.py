import tensorflow as tf
import tf_sparsenet

class TopoSparsenet(tf_sparsenet.Sparsenet):
	
	def __init__(self, datatype="image", pca=None, **kwargs):
		self.lam_g = 0.1

		# TODO: everything

		try:
			kwargs['lam']
			super().__init__(kwargs)
		except KeyError:
			super().__init__(lam=0, kwargs)