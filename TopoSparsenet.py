# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:23:14 2016

@author: Tep & Eric
"""


import DictLearner
import sparsenet
import numpy as np
import matplotlib.pyplot as plt

class TopoSparsenet(sparsenet.Sparsenet):
    
    def __init__(self, data, dict_shape=None, lamb=0.15, lamb_2=0.01,
                 sigma=1, **kwargs):
        self.lamb_2 = lamb_2
        self.dict_shape = dict_shape
        self.nunits = int(np.prod(self.dict_shape))
        self.sigma = sigma
        self.g = self.layer_two_weights()
        super().__init__(data, int(np.prod(self.dict_shape)), lamb=lamb, **kwargs)
        
    def infer(self, X, infplot=False):
        acts = np.zeros((self.nunits,self.batch_size))
        if infplot:
            error_hist = np.zeros(self.niter)
            first_hist = np.zeros(self.niter)
            second_hist = np.zeros(self.niter)
        phi_sq = self.Q.dot(self.Q.T)
        QX = self.Q.dot(X)
        for k in range(self.niter):    
            acts2 = self.g @ acts**2
            da2_da1 = 2 * self.g @ acts
            acts2_terms = self.dSda(acts2) * da2_da1
            da_dt = QX - phi_sq.dot(acts) - self.lamb*self.dSda(acts) - self.lamb_2*acts2_terms
            acts = acts+self.infrate*(da_dt)
            if infplot:
                error_hist[k] = np.mean((X.T-np.dot(acts.T,self.Q))**2) 
                first_hist[k] = np.mean(np.abs(acts))
                second_hist[k] = np.mean(acts2)
        if infplot:
            plt.figure()
            plt.plot(error_hist,'b')
            plt.plot(first_hist,'g')
            plt.plot(second_hist, 'r')
        return acts, None, None
        
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
        
        for i in range(self.nunits):
            for j in range(self.nunits):              
                g[i, j] = np.exp(-self.distance(i, j)/(2. * self.sigma**2))
                
        return g
        
    def show_dict(self, stimset=None, cmap='RdBu', subset=None, square=False, savestr=None):
        """Plot an array of tiled dictionary elements. The 0th element is in the top right."""
        stimset = stimset or self.stims
        if subset is not None:
            Qs = self.Q[subset]
        else:
            Qs = self.Q
        if cmap=='RdBu':
            Qs=-Qs
        array = stimset.stimarray(Qs[::-1], layout=self.dict_shape)
        plt.figure()        
        arrayplot = plt.imshow(array,interpolation='nearest', cmap=cmap, aspect='auto', origin='lower')
        plt.axis('off')
        plt.colorbar()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')
        return arrayplot
        
    def set_params(self, params):
        (self.learnrate, self.infrate, self.niter, self.lamb, self.lamb_2,
             self.measure, self.var_goal, self.gains, self.variances,
             self.var_eta, self.gain_rate) = params
             
    def get_param_list(self):
        return (self.learnrate, self.infrate, self.niter, self.lamb, self.lamb_2,
             self.measure, self.var_goal, self.gains, self.variances,
             self.var_eta, self.gain_rate)
        
    def sort(self, *args, **kwargs):
        print("The topographic order is meaningful, don't sort it away!")  