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

    def __init__(self, eta, paramfile=None, theta=0):
        self.eta = eta # learning rate
        self.Q = self.rand_dict()
        self.errorhist = np.array([])
        self.paramfile = paramfile
        self.theta=theta
    
    def infer(self, data):
        raise NotImplementedError
    
    def learn(self, data, coeffs, normalize = True):
        """Adjust dictionary elements according to gradient descent on the 
        mean-squared error energy function, optionally with an extra term to
        increase orthogonality between basis functions. This term is
        multiplied by the parameter theta.
        Returns the mean-squared error."""
        R = data.T - np.dot(coeffs.T, self.Q)
        self.Q = self.Q + self.eta*np.dot(coeffs,R)
        if self.theta != 0:
            self.Q = self.Q + self.theta*(self.Q - np.dot(self.Q,np.dot(self.Q.T,self.Q)))
        if normalize:
            # force dictionary elements to be normalized
            normmatrix = np.diag(1./np.sqrt(np.sum(self.Q*self.Q,1))) 
            self.Q = normmatrix.dot(self.Q)
        return np.mean(R**2)
            
    def run(self, ntrials = 1000, batch_size = None, show=True, rate_decay=None, normalize = True):
        batch_size = batch_size or self.stims.batch_size
        errors = np.zeros(min(ntrials,1000))
        for trial in range(ntrials):
            if trial % 50 == 0:
                print (trial)
            X = self.stims.rand_stim(batch_size=batch_size)
            coeffs = self.infer(X)
            thiserror = self.learn(X, coeffs, normalize)
            errors[trial % 1000] = thiserror
            
            #temporary hack to stop LCA from killing itself
            #if(thiserror>.85):
            #    rate_decay = 1
            
            if (trial % 1000 == 0 or trial+1 == ntrials) and trial != 0:
                print ("Saving progress to " + self.paramfile)
                self.errorhist = np.concatenate((self.errorhist, errors))
                try: 
                    self.save_params()
                except ValueError as er:
                    print ('Failed to save parameters. ', er)
            if rate_decay is not None:
                self.adjust_rates(rate_decay)
        if show:
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
        datasize = self.stims.datasize
        Q = np.random.randn(self.nunits, datasize)
        normmatrix = np.diag(1./np.sqrt(np.sum(Q*Q,1))) 
        return np.dot(normmatrix,Q)
        
    def adjust_rates(self, factor):
        """Multiply the learning rate by the given factor."""
        self.eta = factor*self.eta
        self.theta = factor*self.theta
        
    def load_params(self, filename=None):
        filename = filename or self.paramfile
        if filename is None:
            raise ValueError("You need to input a filename.")
        self.paramfile = filename
        with open(filename, 'rb') as f:
            self.Q, self.errorhist = pickle.load(f)
        self.picklefile = filename
        
    def save_params(self, filename=None):
        filename = filename or self.paramfile
        if filename is None:
            raise ValueError("You need to input a filename.")
        self.paramfile = filename        
        with open(filename, 'wb') as f:
            pickle.dump([self.Q, self.errorhist], f)
               
