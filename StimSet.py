# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:23:08 2015

@author: Eric Dodds
"""
import numpy as np
import matplotlib.pyplot as plt

class StimSet(object):
    def __init__(self, data, stimshape, batch_size=None):
        """Notice that stimshape and the length of a datum may be different, since the
        data may be represented in a reduced form."""
        self.data = data
        self.stimshape = stimshape
        self.stimsize = np.prod(stimshape)
        self.nstims = data.shape[0]
        self.batch_size = batch_size
        
    def rand_stim(self, batch_size=None):
        """Select random inputs. Return an array of batch_size columns,
        each of which is an input represented as a (column) vector. """
        batch_size = batch_size or self.batch_size
        veclength = np.prod(self.datasize)
        X = np.zeros((veclength, batch_size))
        for i in range(batch_size):
            which = np.random.randint(self.nstims)
            vec = self.data[which,...]
            if len(vec.shape) > 1:
                vec = vec.reshape(self.stimsize)
            X[:,i] = vec
        return X  
    
    @staticmethod
    def _stimarray(stims, stimshape, square=False):
        """Returns an array of the stimuli reshaped to 2d and tiled."""    
        length, height = stimshape
        assert length*height == stims.shape[1]
        buf = 1 # buffer pixels between stimuli
        nstim = stims.shape[0]
        
        # n and m are respectively the numbers of rows and columns of stimuli in the array
        if square:
            if np.floor(np.sqrt(nstim))**2 != nstim:
                n = int(np.ceil(np.sqrt(nstim/2.)))
                m = int(np.ceil(nstim/n))
            else:
                # M is a perfect square
                m = int(np.sqrt(nstim))
                n = m
        else:
            #if length != height, partly account for this so stimuli aren't so distorted. remove the extra square root to fully accommodate
            n = int(np.sqrt(nstim*np.sqrt(height/length)))
            m = int(np.ceil(nstim/n))
        
        array = 0.5*np.ones((buf+n*(length+buf), buf+m*(height+buf)))
        k = 0
        
        for i in range(m):
            for j in range(n):
                if k < nstim:
                    normfactor = np.max(np.abs(stims[k,:]))
                    hstart = buf+i*(height+buf)
                    lstart = buf+j*(length+buf)
                    thestim = stims[k,:].reshape(length,height)/normfactor
                    array[lstart:lstart+length, hstart:hstart+height] = thestim
                    
                k = k+1
                
        return array.T
    
    def stimarray(self, stims, stimshape=None, square=False):
        stimshape = stimshape or self.stimshape
        return StimSet._stimarray(stims, stimshape, square)
        
    def modspec(self, elem):
        """Compute the modulation power spectrum."""
        image = elem.reshape(self.stimshape)
        fourier =  np.fft.rfft2(image)
        mid = int(fourier.shape[0]/2)
        power = np.abs(fourier)**2
        avgmag = np.array([(power[ii] + power[-ii])/2 for ii in range(mid)])
        return avgmag
        
    def stim_for_display(self, stim):
        return stim.reshape(self.stimshape)
        
class ImageSet(StimSet):
    """Currently only compatible with square images (but arbitrary patches)."""
    
    def __init__(self, data, stimshape=(16,16), batch_size=None, buffer=20):
        self.buffer = buffer
        self.datasize = np.prod(stimshape) # size of a patch
        super().__init__(data, stimshape, batch_size)
    
    def rand_stim(self, stimshape=None, batch_size=None):
        """
        Select random patches from the image data. Returns data array of
        batch_size columns, each of which is an unrolled image patch of size
        lpatch**2.
        """
        batch_size = batch_size or self.batch_size or 100
        length, height = stimshape or self.stimshape
        imsize = self.data.shape[0]
        # extract subimages at random from images array to make data array X
        X = np.zeros((length*height, batch_size))
        for i in range(batch_size):
                row = self.buffer + int(np.ceil((imsize-length-2*self.buffer)*np.random.rand()))
                col = self.buffer + int(np.ceil((imsize-height-2*self.buffer)*np.random.rand()))
                animage = self.data[row:row+length,
                                      col:col+height,
                                      np.random.randint(self.data.shape[-1])]                     
                animage = animage.reshape(self.stimsize)
                # normalize image
                animage = animage - np.mean(animage)
                animage = animage/np.std(animage)
                X[:,i] = animage
        return X
        
class PCvecSet(StimSet):
    """Principal component vector representations of arbitrary data."""    
    
    def __init__(self, data, stimshape, pca, batch_size=None):
        self.pca = pca
        self.datasize = data.shape[1]
        super().__init__(data, stimshape, batch_size)
        
    def stimarray(self, stims, square=False):
        reconst = self.pca.inverse_transform(stims)
        return super().stimarray(reconst, self.stimshape, square)
        
    def modspec(self, elem):
        return super().modspec(self.pca.inverse_transform(elem))
        
    def stim_for_display(self, stim):
        return super().stim_for_display(self.pca.inverse_transform(stim))
        
class WaveformSet(StimSet):
    """1D signals, especially audio, of uniform length."""
    
    def tiledplot(self, stims):
        """Tiled plots of the given stumili. Zeroth index is over stimuli."""
        nstim = stims.shape[0]
        plotrows = int(np.sqrt(nstim))
        plotcols = int(np.ceil(nstim/plotrows))
        f, axes = plt.subplots(plotrows, plotcols, sharex=True, sharey=True)
        for ii in range(nstim):
            axes.flatten()[ii].plot(stims[ii])
        f.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.setp([a.get_yticklabels() for a in f.axes[:-1]], visible=False)
        
class WaveformPCSet(PCvecSet, WaveformSet):
    """Specifically for PCA reps of waveforms."""
    
    def tiledplot(self, stims):
        super().tiledplot(self.pca.inverse_transform(stims))