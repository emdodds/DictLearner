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
from scipy import ndimage

class DictLearner(object):

    def __init__(self, learnrate, paramfile=None, theta=0, moving_avg_rate=0.001):
        self.learnrate = learnrate
        self.Q = self.rand_dict()
        self.errorhist = np.array([])
        self.paramfile = paramfile
        self.theta=theta
        self.L0hist = np.array([])
        self.L1hist = np.array([])
        nunits = self.Q.shape[0]
        self.L0acts = np.zeros(nunits)
        self.L1acts = np.zeros(nunits)
        self.moving_avg_rate=moving_avg_rate
    
    def infer(self, data):
        raise NotImplementedError
    
    def learn(self, data, coeffs, normalize = True):
        """Adjust dictionary elements according to gradient descent on the 
        mean-squared error energy function, optionally with an extra term to
        increase orthogonality between basis functions. This term is
        multiplied by the parameter theta.
        Returns the mean-squared error."""
        R = data.T - np.dot(coeffs.T, self.Q)
        self.Q = self.Q + self.learnrate*np.dot(coeffs,R)
        if self.theta != 0:
            # Notice this is calculated using the Q after the mse learning rule
            thetaterm = (self.Q - np.dot(self.Q,np.dot(self.Q.T,self.Q)))
            self.Q = self.Q + self.theta*thetaterm
        if normalize:
            # force dictionary elements to be normalized
            normmatrix = np.diag(1./np.sqrt(np.sum(self.Q*self.Q,1))) 
            self.Q = normmatrix.dot(self.Q)
        return np.mean(R**2)
            
    def run(self, ntrials = 1000, batch_size = None, show=True, rate_decay=None, normalize = True):
        batch_size = batch_size or self.stims.batch_size
        errors = np.zeros(min(ntrials,1000))
        L0means = np.zeros_like(errors)
        L1means = np.zeros_like(errors)
        for trial in range(ntrials):
            if trial % 50 == 0:
                print (trial)
            X = self.stims.rand_stim(batch_size=batch_size)
            coeffs,_,_ = self.infer(X)
            thiserror = self.learn(X, coeffs, normalize)
            errors[trial % 1000] = thiserror
            L0means[trial % 1000] = np.mean(coeffs!=0)
            L1means[trial % 1000] = np.mean(np.abs(coeffs))
            
            if (trial % 1000 == 0 or trial+1 == ntrials) and trial != 0:
                print ("Saving progress to " + self.paramfile)
                self.errorhist = np.concatenate((self.errorhist, errors))
                self.L0hist = np.concatenate((self.L0hist, L0means))
                self.L1hist = np.concatenate((self.L1hist, L1means))
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
    
    def show_dict(self, stimset=None, cmap='jet', subset=None, savestr=None):
        """The StimSet object handles the plotting of the current dictionary."""
        stimset = stimset or self.stims
        if subset is not None:
            indices = np.random.choice(self.Q.shape[0], subset)
            Qs = self.Q[np.sort(indices)]
        else:
            Qs = self.Q
        array = stimset.stimarray(Qs[::-1])
        plt.figure()        
        arrayplot = plt.imshow(array,interpolation='nearest', cmap=cmap, aspect='auto', origin='lower')
        plt.axis('off')
        plt.colorbar()
        if savestr is not None:
            plt.savefig(savestr, bbox_inches='tight')
        return arrayplot
        
    def rand_dict(self):
        Q = np.random.randn(self.nunits, self.stims.datasize)
        return (np.diag(1/np.sqrt(np.sum(Q**2,1)))).dot(Q)
        
    def adjust_rates(self, factor):
        """Multiply the learning rate by the given factor."""
        self.learnrate = factor*self.learnrate
        self.theta = factor*self.theta

    def modulation_plot(self, usepeaks=False, **kwargs):
        modcentroids = np.zeros((self.Q.shape[0],2))
        for ii in range(self.Q.shape[0]):
            modspec = self.stims.modspec(self.Q[ii])
            if usepeaks:
                modcentroids[ii,0] = np.argmax(np.mean(modspec,axis=1))
                modcentroids[ii,1] = np.argmax(np.mean(modspec,axis=0))
            else:
                modcentroids[ii] = ndimage.measurements.center_of_mass(modspec)
        plt.scatter(modcentroids[:,0], modcentroids[:,1])
        plt.title('Center of mass of modulation power spectrum of each dictionary element')
        try:
            plt.xlabel(kwargs.xlabel)
        except:
            pass
        try:
            plt.ylabel(kwargs.ylabel)
        except:
            pass
        
        
    def load_params(self, filename=None):
        if filename is None:
            filename = self.paramfile
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
               
