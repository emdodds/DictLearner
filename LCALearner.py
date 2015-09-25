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
    
    def __init__(self, data, nunits, learn_rate=.01, theta = 0.022,
                 batch_size = 100, infrate=.003, niter=300, min_thresh=0.4, adapt=0.95,
                 softthresh = False, datatype = "image",
                 pca = None, stimshape = None, paramfile = None):
        self.batch_size = batch_size
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
        
    
    def infer(self, X, infplot=False):
        """Infer sparse approximation to given data X using this LCALearner's 
        current dictionary. Returns coefficients of sparse approximation."""
        ndict = self.Q.shape[0]
        nstim = X.shape[-1]
        u = np.zeros((nstim, ndict))
        s = np.zeros_like(u)
        ci = np.zeros((nstim, ndict))
        
        # c is the overlap of dictionary elements with each other, minus identity (i.e., ignore self-overlap)
        c = self.Q.dot(self.Q.T) - np.eye(ndict)
        
        # b[i,j] is overlap of stimulus i with dictionary element j
        b = X.T.dot(self.Q.T)
        thresh = np.absolute(b).mean(1) # initial thresholds

        
        if infplot:
            errors = np.zeros(self.niter)
        
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
                s[np.absolute(s) < thresh[:,np.newaxis]] = 0.   
            thresh[thresh>self.min_thresh] = self.adapt*thresh[thresh>self.min_thresh]
            
            if infplot:
                errors[kk] = np.mean((X.T - s.dot(self.Q))**2)
        
        if infplot:
            plt.figure(3)
            plt.clf()
            plt.plot(errors)
        return s.T
            
    def test_inference(self, niter=None):
        temp = self.niter
        self.niter = niter or self.niter
        X = self.stims.rand_stim()
        self.infer(X, infplot=True)
        self.niter = temp
                  
    def sort_dict(self, batch_size=None, plot = False):
        """Sorts the RFs in order by their activities on a batch. Default batch size
        is 10 times the stored batch size."""
        batch_size = batch_size or 10*self.batch_size
        testX = self.stims.rand_stim(batch_size)
        # the coefficients can be positive or negative; I use the RMS average to get a sense of how often they're used
        means = np.sqrt(np.mean(self.infer(testX)**2, axis=1))
        sorter = np.argsort(means)
        self.Q = self.Q[sorter]
        plt.plot(means[sorter])
        return means[sorter]
            
