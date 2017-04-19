# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:55:22 2016

@author: Eric
"""

from DictLearner import DictLearner as BaseLearner
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Can't import matplotlib.")

class Sparsenet(BaseLearner):
    """A sparse dictionary learner based on (Olshausen and Field, 1996)."""
    
    def __init__(self, data, nunits, learnrate=0.01, measure='abs', infrate=0.01,
                 niter=200, lamb=0.15, var_goal=0.1, gain_rate=0.02,
                 var_eta=0.1, **kwargs):
        self.niter=niter
        self.lamb = lamb
        self.infrate=infrate
        self.measure = measure
        self.var_goal = var_goal
        self.gains = np.ones(nunits)
        self.variances = self.var_goal*np.ones(nunits)
        self.var_eta = var_eta
        self.gain_rate = gain_rate
        BaseLearner.__init__(self, data, learnrate, nunits, **kwargs)
        
    def dSda(self, acts):
        """Returns the derivative of the activity-measuring function."""
        if self.measure == 'log':
            # TODO: Why doesn't this work well? The activities in the denominator may need to be scaled?
            return acts*(1/(1+acts*acts))
        elif self.measure == 'abs':
            return np.sign(acts)
        elif self.measure == 'bell':
            return 2*acts*np.exp(-acts**2)
            
    
    def infer(self, X, infplot=False):
        acts = np.zeros((self.nunits,X.shape[1]))
        if infplot:
            costY1 = np.zeros(self.niter)
        phi_sq = self.Q.dot(self.Q.T)
        QX = self.Q.dot(X)
        for k in range(self.niter):    
            da_dt = QX - phi_sq.dot(acts) - self.lamb*self.dSda(acts)
            acts = acts+self.infrate*(da_dt)
            
            if infplot:
                costY1[k]=np.mean((X.T-np.dot(acts.T,self.Q))**2) 
        if infplot:
            plt.plot(costY1)
        return acts, None, None
    
    def learn(self, data, coeffs, normalize=True):
        mse = BaseLearner.learn(self, data, coeffs, normalize)
        variances = np.diag(coeffs.dot(coeffs.T))/self.batch_size
        self.variances = (1-self.var_eta)*self.variances + self.var_eta*variances
        newgains = self.var_goal/self.variances
        self.gains = self.gains*newgains**self.gain_rate
        self.Q = self.gains[:,np.newaxis]*self.Q
        return mse
        
    def sort(self, usages, sorter, plot=False, savestr=None):
        self.gains = self.gains[sorter]
        self.variances = self.variances[sorter]
        BaseLearner.sort(self, usages, sorter, plot, savestr)
        
    def set_params(self, params):
        (self.learnrate, self.infrate, self.niter, self.lamb,
             self.measure, self.var_goal, self.gains, self.variances,
             self.var_eta, self.gain_rate) = params
             
    def get_param_list(self):
        return (self.learnrate, self.infrate, self.niter, self.lamb,
             self.measure, self.var_goal, self.gains, self.variances,
             self.var_eta, self.gain_rate)