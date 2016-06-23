# -*- coding: utf-8 -*-

import numpy as np
from DictLearner import DictLearner
import scipy.sparse.linalg

"""The inference code was adapted from S. Zayd Enam's sparsenet implementation,
available on github."""

class FISTALearner(DictLearner):
    
    def __init__(self, data, learnrate, nunits, lam = 0.4, niter=100, **kwargs):
        self.lam = 0.4
        self.niter = niter
        super().__init__(data, learnrate, nunits, **kwargs)
    
    def infer(self, data, max_iterations=None, display=False):
      """ FISTA Inference for Lasso (l1) Problem 
      data: Batches of data (dim x batch)
      Phi: Dictionary (dictionary element x dim) (nparray or sparse array)
      lambdav: Sparsity penalty
      max_iterations: Maximum number of iterations
      """
      lambdav=self.lam
      def proxOp(x,t):
        """ L1 Proximal Operator """ 
        return np.fmax(x-t, 0) + np.fmin(x+t, 0)
    
      x = np.zeros((self.Q.shape[0], data.shape[1]))
      c = self.Q.dot(self.Q.T)
      b = -2*self.Q.dot(data)
    
      L = scipy.sparse.linalg.eigsh(2*c, 1, which='LM')[0]
      invL = 1/float(L)
    
      y = x
      t = 1
    
      max_iterations = max_iterations or self.niter
      for i in range(max_iterations):
        g = 2*c.dot(y) + b
        x2 = proxOp(y-invL*g,invL*lambdav)
        t2 = (1+np.sqrt(1+4*(t**2)))/2.0
        y = x2 + ((t-1)/t2)*(x2-x)
        x = x2
        t = t2
        if display == True:
          print ("L1 Objective " +  str(np.sum((data-self.Q.T.dot(x2))**2) + lambdav*np.sum(np.abs(x2))))
    
      return x2, 0, 0