# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:48:46 2015

@author: Eric Dodds
(Inference method adapted from code by Jesse Livezey)
Dictionary learner that uses LCA for inference and gradient descent for learning.
(Intended for static inputs)
"""
import numpy as np
import matplotlib.pyplot as plt
import StimSet
from DictLearner import DictLearner

class LCALearner(DictLearner):
    
    def __init__(self, data, nunits, learn_rate=None, theta = 0.022,
                 batch_size = 100, infrate=.003, #.0005 in Nicole's notes
                 niter=300, min_thresh=0.4, adapt=0.95,
                 softthresh = False, datatype = "image",
                 pca = None, stimshape = None, paramfile = None):
        self.batch_size = batch_size
        learn_rate = learn_rate or 1./(2*self.batch_size)
        if datatype == "image":
            stimshape = stimshape or (16,16)
            self.stims = StimSet.ImageSet(data, batch_size = self.batch_size, buffer=20, stimshape = stimshape)
        elif datatype == "spectro" and pca is not None:
            if stimshape == None:
                raise Exception("When using PC representations, you need to provide the shape of the original stimuli.")
            self.stims = StimSet.PCvecSet(data, stimshape, pca, self.batch_size)
        else:
            raise ValueError("Specified data type not currently supported.")
        self.nunits = nunits
        self.infrate = infrate
        self.niter = niter
        self.min_thresh = min_thresh
        self.adapt = adapt
        self.softthresh = softthresh
        super().__init__(learn_rate, paramfile = paramfile, theta=theta)
        
    
    def infer(self, X, infplots=False):
        """Infer sparse approximation to given data X using this LCALearner's 
        current dictionary. Returns coefficients of sparse approximation."""
        ndict = self.Q.shape[0]
        nstim = X.shape[-1]
        u = np.zeros((nstim, ndict))
        s = np.zeros_like(u)
        ci = np.zeros_like(u)
        
        # c is the overlap of dictionary elements with each other, minus identity (i.e., ignore self-overlap)
        c = self.Q.dot(self.Q.T) - np.eye(ndict)
        
        # b[i,j] is overlap of stimulus i with dictionary element j
        b = (self.Q.dot(X)).T

        thresh = np.absolute(b).mean(1) # initial thresholds
        
        if infplots:
            errors = np.zeros(self.niter)
            histories = np.zeros((ndict, self.niter))
        
        for kk in range(self.niter):
            # ci is the competition term in the dynamical equation
            ci = s.dot(c)
            u = self.infrate*(b-ci) + (1-self.infrate)*u
            if np.max(np.isnan(u)):
                raise ValueError("Internal variable blew up at iteration " + str(kk))
            if self.softthresh:
                s = np.sign(u)*np.maximum(0.,np.absolute(u)-thresh[:,np.newaxis]) 
            else:
                s[:] = u
                s[np.absolute(s) < thresh[:,np.newaxis]] = 0
                
            if infplots:
                histories[:,kk] = u[0,:]
                errors[kk] = np.mean((X.T - s.dot(self.Q))**2)
                
            thresh[thresh>self.min_thresh] = self.adapt*thresh[thresh>self.min_thresh]
            
        
        if infplots:
            plt.figure(3)
            plt.clf()
            plt.plot(errors)
            plt.figure(4)
            plt.clf()
            plt.plot(histories.T)
        return s.T
            
    def test_inference(self, niter=None):
        temp = self.niter
        self.niter = niter or self.niter
        X = self.stims.rand_stim()
        s = self.infer(X, infplots=True)
        self.niter = temp
        return s
                  
    def sort_dict(self, batch_size=None, plot = False):
        """Sorts the RFs in order by their usage on a batch. Default batch size
        is 10 times the stored batch size. Usage means 1 for each stimulus for
        which the element was used and 0 for the other stimuli, averaged over 
        stimuli."""
        batch_size = batch_size or 10*self.batch_size
        testX = self.stims.rand_stim(batch_size)
        means = np.mean(self.infer(testX) != 0, axis=1)
        sorter = np.argsort(means)
        self.Q = self.Q[sorter]
        if plot:
            plt.plot(means[sorter])
        return means[sorter]
            
    def rand_dict(self):
        datasize = self.stims.datasize
        Q = np.random.randn(self.nunits, datasize) - 0.5
        normmatrix = np.diag(1/np.sqrt(np.sum(Q*Q,1))) 
        return np.dot(normmatrix,Q)