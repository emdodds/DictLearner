# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:00:46 2016

@author: Eric
"""

import numpy as np
import matplotlib.pyplot as plt
import LCALearner
import pickle

class HomeostaticLCA(LCALearner.LCALearner):
    """LCA with homeostatic activities, a la SAILnet. Each unit has its own 
    threshold (lambda) which is learned (homeorate) to keep the average activity
    around firingrate. min_thresh is the initial value for all lambdas. 
    Other arguments are as LCALearner, must be keyworded."""
    def __init__(self, *args, firingrate=0.05, homeorate=0.001, min_thresh=0.4, **kwargs):
        self.firingrate = firingrate
        self.homeorate=homeorate
        self.lams = np.ones(args[1])*min_thresh
        super().__init__(*args, **kwargs)
    
    def learn(self, data, coeffs, normalize = True):
        abscoeffs = np.abs(coeffs)
        meanabs = np.mean(abscoeffs,1)
        self.lams = self.lams + self.homeorate*(meanabs - self.firingrate)
        return super().learn(data, coeffs, normalize)
    
    def infer_cpu(self, X, infplot=False, tolerance=None, max_iter = None):
        """Infer sparse approximation to given data X using this LCALearner's 
        current dictionary. Returns coefficients of sparse approximation.
        Optionally plot reconstruction error vs iteration number.
        The instance variable niter determines for how many iterations to evaluate
        the dynamical equations. Repeat this many iterations until the mean-squared error
        is less than the given tolerance or until max_iter repeats."""
        tolerance = tolerance or self.tolerance
        max_iter = max_iter or self.max_iter
        ndict = self.Q.shape[0]
        thresh = self.lams
               
        nstim = X.shape[-1]
        u = np.zeros((nstim, ndict))
        s = np.zeros_like(u)
        ci = np.zeros_like(u)
        
        # c is the overlap of dictionary elements with each other, minus identity (i.e., ignore self-overlap)
        c = self.Q.dot(self.Q.T)
        for i in range(c.shape[0]):
            c[i,i] = 0
        
        # b[i,j] is overlap of stimulus i with dictionary element j
        b = (self.Q.dot(X)).T
        
        if infplot:
            errors = np.zeros(self.niter)
            allerrors = np.array([])
        
        error = tolerance+1
        outer_k = 0
        while(error>tolerance and ((max_iter is None) or outer_k<max_iter)):
            for kk in range(self.niter):
                # ci is the competition term in the dynamical equation
                ci[:] = s.dot(c)
                u[:] = self.infrate*(b-ci) + (1.-self.infrate)*u
                if np.max(np.isnan(u)):
                    raise ValueError("Internal variable blew up at iteration " + str(kk))
                if self.softthresh:
                    s[:] = np.sign(u)*np.maximum(0.,np.absolute(u)-thresh[:,np.newaxis]) 
                else:
                    s[:] = u
                    s[np.absolute(s) < thresh] = 0
                    
                if infplot:
                    errors[kk] = np.mean(self.compute_errors(s.T,X))
                
                
            error = np.mean((X.T - s.dot(self.Q))**2)
            outer_k = outer_k+1
            if infplot:
                allerrors = np.concatenate((allerrors,errors))
        
        if infplot:
            plt.figure(3)
            plt.clf()
            plt.plot(allerrors)
            return s.T, errors
        return s.T, u.T, thresh
        
    def load_params(self, filename=None):
        """Loads the parameters that were saved. For older files when I saved less, loads what I saved then."""
        self.paramfile = filename
        with open(filename, 'rb') as f:
            self.Q, params, histories = pickle.load(f)      
        self.errorhist, self.L0acts, self.L1acts = histories
        self.learnrate, self.theta, self.lams, self.infrate, self.niter, self.adapt, self.max_iter, self.tolerance = params
        self.picklefile = filename
        
    def save_params(self, filename=None, dialog=False):
        filename = filename or self.paramfile
        if filename is None:
            raise ValueError("You need to input a filename.")
        self.paramfile = filename        
        params = (self.learnrate, self.theta, self.lams, self.infrate, 
                  self.niter, self.adapt, self.max_iter, self.tolerance)
        histories = (self.errorhist,self.L0acts, self.L1acts)
        with open(filename, 'wb') as f:
            pickle.dump([self.Q, params, histories], f)
        
class HomeoPositiveLCA(HomeostaticLCA):
    """HomeostaticLCA with activities forced to be positive."""
    def infer_cpu(self, X, infplot=False, tolerance=None, max_iter = None):
        """Infer sparse approximation to given data X using this LCALearner's 
        current dictionary. Returns coefficients of sparse approximation.
        Optionally plot reconstruction error vs iteration number.
        The instance variable niter determines for how many iterations to evaluate
        the dynamical equations. Repeat this many iterations until the mean-squared error
        is less than the given tolerance or until max_iter repeats."""
        tolerance = tolerance or self.tolerance
        max_iter = max_iter or self.max_iter
        ndict = self.Q.shape[0]
        thresh = self.lams
               
        nstim = X.shape[-1]
        u = np.zeros((nstim, ndict))
        s = np.zeros_like(u)
        ci = np.zeros_like(u)
        
        # c is the overlap of dictionary elements with each other, minus identity (i.e., ignore self-overlap)
        c = self.Q.dot(self.Q.T)
        for i in range(c.shape[0]):
            c[i,i] = 0
        
        # b[i,j] is overlap of stimulus i with dictionary element j
        b = (self.Q.dot(X)).T
        
        if infplot:
            errors = np.zeros(self.niter)
            allerrors = np.array([])
        
        error = tolerance+1
        outer_k = 0
        while(error>tolerance and ((max_iter is None) or outer_k<max_iter)):
            for kk in range(self.niter):
                # ci is the competition term in the dynamical equation
                ci[:] = s.dot(c)
                u[:] = self.infrate*(b-ci) + (1.-self.infrate)*u
                if np.max(np.isnan(u)):
                    raise ValueError("Internal variable blew up at iteration " + str(kk))
                if self.softthresh:
                    s[:] = np.maximum(0.,u-thresh[:,np.newaxis]) 
                else:
                    s[:] = u
                    s[s < thresh] = 0
                    
                if infplot:
                    errors[kk] = np.mean(self.compute_errors(s.T,X))
                
                
            error = np.mean((X.T - s.dot(self.Q))**2)
            outer_k = outer_k+1
            if infplot:
                allerrors = np.concatenate((allerrors,errors))
        
        if infplot:
            plt.figure(3)
            plt.clf()
            plt.plot(allerrors)
            return s.T, errors
        return s.T, u.T, thresh
        
class PositiveLCA(LCALearner.LCALearner):
    """LCA with activities forced to be positive."""
    def infer_cpu(self, X, infplot=False, tolerance=None, max_iter = None):
        """Infer sparse approximation to given data X using this LCALearner's 
        current dictionary. Returns coefficients of sparse approximation.
        Optionally plot reconstruction error vs iteration number.
        The instance variable niter determines for how many iterations to evaluate
        the dynamical equations. Repeat this many iterations until the mean-squared error
        is less than the given tolerance or until max_iter repeats."""
        tolerance = tolerance or self.tolerance
        max_iter = max_iter or self.max_iter
        ndict = self.Q.shape[0]
               
        nstim = X.shape[-1]
        u = np.zeros((nstim, ndict))
        s = np.zeros_like(u)
        ci = np.zeros_like(u)
        
        # c is the overlap of dictionary elements with each other, minus identity (i.e., ignore self-overlap)
        c = self.Q.dot(self.Q.T)
        for i in range(c.shape[0]):
            c[i,i] = 0
        
        # b[i,j] is overlap of stimulus i with dictionary element j
        b = (self.Q.dot(X)).T

        # initialize threshold values, one for each stimulus, based on average response magnitude
        thresh = np.absolute(b).mean(1) 
        thresh = np.array([np.max([th, self.min_thresh]) for th in thresh])
        
        if infplot:
            errors = np.zeros(self.niter)
            allerrors = np.array([])
        
        error = tolerance+1
        outer_k = 0
        while(error>tolerance and ((max_iter is None) or outer_k<max_iter)):
            for kk in range(self.niter):
                # ci is the competition term in the dynamical equation
                ci[:] = s.dot(c)
                u[:] = self.infrate*(b-ci) + (1.-self.infrate)*u
                if np.max(np.isnan(u)):
                    raise ValueError("Internal variable blew up at iteration " + str(kk))
                if self.softthresh:
                    s[:] = np.maximum(0.,u-thresh[:,np.newaxis]) 
                else:
                    s[:] = u
                    s[s < thresh[:,np.newaxis]] = 0
                    
                if infplot:
                    errors[kk] = np.mean(self.compute_errors(s.T,X))
                    
                thresh = self.adapt*thresh
                thresh[thresh<self.min_thresh] = self.min_thresh
                
            error = np.mean((X.T - s.dot(self.Q))**2)
            outer_k = outer_k+1
            if infplot:
                allerrors = np.concatenate((allerrors,errors))
        
        if infplot:
            plt.figure(3)
            plt.clf()
            plt.plot(allerrors)
            return s.T, errors
        return s.T, u.T, thresh