# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:01:18 2015

@author: Eric Dodds

Abstract dictionary learner.
Includes gradient descent on MSE energy function as a default learning method.
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt

class DictLearner(object):

    def __init__(self, eta):
        self.eta = eta # learning rate
        self.Q = self.rand_dict()
        self.errorhist = np.array([])
    
    def infer(self, data):
        raise NotImplementedError
    
    def learn(self, data, coeffs, theta=0):
        """Adjust dictionary elements according to gradient descent on the 
        mean-squared error energy function, optionally with an extra term to
        increase orthogonality between basis functions. This term is
        multiplied by the parameter theta.
        Returns the mean-squared error."""
        R = data.T - np.dot(coeffs, self.Q)
        self.Q = self.Q + self.eta*np.dot(coeffs.T,R)
        if theta != 0:
            self.Q = self.Q + theta*(self.Q - np.dot(self.Q,np.dot(self.Q.T,self.Q)))
        return np.mean(R**2)
            
    def run(self, ntrials = 1000, batch_size = None):
        batch_size = batch_size or self.stims.batch_size        
        errors = np.zeros(ntrials)
        for trial in range(ntrials):
            if trial % 50 == 0:
                print (trial)
            X = self.stims.rand_stim(batch_size=batch_size)
            coeffs = self.infer(X)
            errors[trial] = self.learn(X, coeffs, theta=0)   
        self.errorhist = np.concatenate((self.errorhist, errors))
        plt.figure()
        plt.plot(self.errorhist)
        plt.show()            
    
    def show_dict(self, stimset=None, cmap='gray'):
        """The StimSet object handles the plotting of the current dictionary."""
        stimset = stimset or self.stims
        array = stimset.stimarray(self.Q)        
        arrayplot = plt.imshow(array,interpolation='nearest', cmap=cmap, aspect='auto')
        plt.gca().invert_yaxis()
        plt.colorbar()
        return arrayplot
        
    def rand_dict(self):
        dataSize = np.prod(self.stims.data.shape[:-1])
        Q = np.random.randn(self.nunits, dataSize)
        normmatrix = np.diag(1/np.sqrt(np.sum(Q*Q,1))) 
        return np.dot(normmatrix,Q)
        
    def adjust_rates(self, factor):
        """Multiply the learning rate by the given factor."""
        self.eta = factor*self.eta
        
    def load_params(self, filename=None):
        filename = filename or self.picklefile
        with open(filename, 'rb') as f:
            self.Q = pickle.load(f)
        self.picklefile = filename
        
    def save_params(self, filename=None):
        filename = filename or self.picklefile
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)
            