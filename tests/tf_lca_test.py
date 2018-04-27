# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:14:20 2017

@author: Eric
"""

import numpy as np
import scipy.io as io
from tf_lca import LCALearner

class tf_lca_test():
    
    def setup(self):
        images = io.loadmat('Data/IMAGES.mat')['IMAGES']
        images /= images.std()
        self.net = LCALearner(images)
    
    def inference_test(self):
        """Check that inference runs and the cost goes down."""
        _, costs = self.net.test_inference()
        assert costs[-1] < costs[0]

    def learn_test(self):
        """Just check that a training step runs without errors."""
        self.net.train_step()