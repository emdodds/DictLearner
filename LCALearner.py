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
                 niter=300, min_thresh=0.4, adapt=0.95, tolerance = .01, max_iter=4,
                 softthresh = False, datatype = "image",
                 pca = None, stimshape = None, paramfile = None):
        """
        An LCALearner is a dictionary learner (DictLearner) that uses a Locally Competitive Algorithm (LCA) for inference.
        By default the LCALearner optimizes for sparsity as measured by the L0 pseudo-norm of the activities of the units
        (i.e. the usages of the dictionary elements). 
        
        Args:
            data: data presented to LCALearner for estimating with LCA
            nunits: number of units in thresholding circuit = number dictionary elements
            learn_rate: rate for mean-squared error part of learning rule
            theta: rate for orthogonality constraint part of learning rule
            batch_size: number of data presented for inference per learning step
            infrate: rate for evolving the dynamical equation in inference (size of each step)
            niter: number of steps in inference (if tolerance is small, chunks of this many iterations are repeated until tolerance is satisfied)
            min_thresh: thresholds are reduced during inference no lower than this value
            adapt: factor by which thresholds are multipled after each inference step
            tolerance: inference ceases after mean-squared error falls below tolerance
            max_iter: maximum number of chunks of inference (each chunk having niter iterations)
            softthresh: if True, optimize for L1-sparsity
            datatype: image or spectro
            pca: pca object for inverse-transforming data if used in PC representation
            stimshape: original shape of data (e.g., before unrolling and PCA)
            paramfile: a pickle file with dictionary and error history is stored here        
        """
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
        self.tolerance = tolerance
        self.max_iter = max_iter
        super().__init__(learn_rate, paramfile = paramfile, theta=theta)
        
    
    def infer(self, X, infplots=False, tolerance=None, max_iter = None):
        """Infer sparse approximation to given data X using this LCALearner's 
        current dictionary. Returns coefficients of sparse approximation.
        Optionally plot reconstruction error vs iteration number.
        The instance variable niter determines for how many iterations to evaluate
        the dynamical equations. Repeat this many iterations until the mean-squared error
        is less than the given tolerance."""
        tolerance = tolerance or self.tolerance
        max_iter = max_iter or self.max_iter
        
        ndict = self.Q.shape[0]
        nstim = X.shape[-1]
        u = np.zeros((nstim, ndict))
        s = np.zeros_like(u)
        ci = np.zeros_like(u)
        
        # c is the overlap of dictionary elements with each other, minus identity (i.e., ignore self-overlap)
        c = self.Q.dot(self.Q.T) - np.eye(ndict)
        
        # b[i,j] is overlap of stimulus i with dictionary element j
        b = (self.Q.dot(X)).T

        # initialize threshold values, one for each stimulus, based on average response magnitude
        thresh = np.absolute(b).mean(1) 
        thresh = np.array([np.max([thr, self.min_thresh]) for thr in thresh])
        
        if infplots:
            errors = np.zeros((self.niter, nstim))
            histories = np.zeros((ndict, self.niter))
            shistories = np.zeros((ndict, self.niter))
            threshhist = np.zeros((nstim, self.niter))
        
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
                    s[np.absolute(s) < thresh[:,np.newaxis]] = 0
                    
                if infplots:
                    histories[:,kk] = u[0,:]
                    shistories[:,kk] = s[0,:]
                    errors[kk,:] = np.mean((X.T - s.dot(self.Q))**2,axis=1)
                    threshhist[:,kk] = thresh
                    
                thresh = self.adapt*thresh
                thresh[thresh<self.min_thresh] = self.min_thresh
                
            error = np.mean((X.T - s.dot(self.Q))**2)
            outer_k = outer_k+1
        
        if infplots:
            plt.figure(3)
            plt.clf()
            plt.plot(errors)
            plt.figure(4)
            plt.clf()
            hists = np.concatenate((histories,shistories),axis=0)
            plt.plot(hists.T)
            return s.T, errors, histories, shistories, threshhist
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
        """Return a random normalized dictionary."""
        datasize = self.stims.datasize
        Q = np.random.randn(self.nunits, datasize) - 0.5
        normmatrix = np.diag(1/np.sqrt(np.sum(Q*Q,1))) 
        return np.dot(normmatrix,Q)