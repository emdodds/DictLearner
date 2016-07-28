# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:55:22 2016

@author: Eric
"""

import DictLearner
import numpy as np
import matplotlib.pyplot as plt

class Sparsenet(DictLearner.DictLearner):
    """A sparse dictionary learner based on (Olshausen and Field, 1996)."""
    
    def __init__(self, data, nunits, learnrate=0.01, measure='abs', infrate=0.01,
                 niter=200, lamb=0.15, var_goal=0.1, gain_rate=0.02,
                 var_eta=0.1, **kwargs):
        self.niter=niter
        self.lamb = lamb
        self.infrate=0.01
        self.measure = measure
        self.var_goal = var_goal
        self.gains = np.ones(nunits)
        self.variances = self.var_goal*np.ones(nunits)
        self.var_eta = var_eta
        self.gain_rate = gain_rate
        super().__init__(data, learnrate, nunits, **kwargs)
        
    def dSda(self, acts):
        if self.measure == 'log':
            # TODO: Why doesn't this work well? The activities in the denominator may need to be scaled?
            return acts*(1/(1+acts*acts))
        elif self.measure == 'abs':
            return np.sign(acts)
        elif self.measure == 'bell':
            return 2*acts*np.exp(-acts**2)
            
    
    def infer(self,X, plot=False):
        acts = np.zeros((self.nunits,self.batch_size))
        costY1 = np.zeros(self.niter)
                       
        for k in range(self.niter):    
            phi_sq = self.Q.dot(self.Q.T)
            QX = self.Q.dot(X)
            da_dt = QX - phi_sq.dot(acts) - self.lamb*self.dSda(acts)
            acts=(1-self.infrate)*acts+self.infrate*(da_dt)

            costY1[k]=np.mean((X.T-np.dot(acts.T,self.Q))**2) 
        if plot:
            plt.plot(costY1)
        return acts, None, None
    
    def learn(self, data, coeffs, normalize=True):
        mse = super().learn(data, coeffs, normalize)
        variances = np.diag(coeffs.dot(coeffs.T))/self.batch_size
        self.variances = (1-self.var_eta)*self.variances + self.var_eta*variances
        newgains = self.var_goal/self.variances
        self.gains = self.gains*newgains**self.gain_rate
        self.Q = self.gains[:,np.newaxis]*self.Q
        return mse