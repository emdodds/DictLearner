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
    
    def __init__(self, data, nunits, learn_rate=.001, batch_size = None, 
                 infrate=.01, niter=150, softthresh = False, datatype = "image",
                 pca = None, stimshape = None):
        self.batch_size = batch_size or data.shape[0]
        if datatype == "image":
            self.stims = StimSet.ImageSet(data, self.batch_size, buffer=20, lpatch=16)
        elif datatype == "spectro" and pca is not None:
            if stimshape == None:
                raise Exception("When using PC representations, you need to provide the shape of the original stimuli.")
            self.stims = StimSet.PCvecSet(data, stimshape, pca, self.batch_size)
        else:
            raise ValueError("Specified data type not currently supported.")
        self.nunits = nunits
        self.infrate = infrate
        self.niter = niter
        self.softthresh = softthresh
        super().__init__(learn_rate)
        
    
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
        thresh = np.absolute(b).mean(1)
        
        if infplot:
            errors = np.zeros(self.niter)
        
        for kk in range(self.niter):
            # ci is the competition term in the dynamical equation
            ci = s.dot(c)
            u = self.infrate*(b-ci) + (1-self.infrate)*u
            if np.max(np.isnan(u)):
                print ("Internal variable blew up at iteration " + str(kk) + "Current values:")
                print (u)
                raise ValueError
            if self.softthresh:
                s = np.sign(u)*np.maximum(0.,np.absolute(u)-thresh[:,np.newaxis]) 
            else:
                s[:] = u
                s[np.absolute(s) < thresh[:,np.newaxis]] = 0.   
                
            if infplot:
                errors[kk] = np.mean((X - s.dot(self.Q))**2)
        
        if infplot:
            plt.figure(3)
            plt.clf()
            plt.plot(errors)
        return s
            
    def test_inference(self, niter=None):
        temp = self.niter
        self.niter = niter or self.niter
        X = self.stims.randStim()
        self.infer(X, infplot=True)
        self.niter = temp
                  
            
            