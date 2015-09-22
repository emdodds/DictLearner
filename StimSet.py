# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:23:08 2015

@author: Eric Dodds
"""
import numpy as np

class StimSet(object):
    def __init__(self, data, stimshape, batch_size=None):
        """Notice that stimshape and the length of a datum may be different, since the
        data may be represented in a reduced form."""
        self.data = data
        self.stimshape = stimshape
        self.stimsize = np.prod(stimshape)
        self.datasize = data.shape[1]
        self.nstims = data.shape[0]
        self.batch_size = batch_size
        
    def rand_stim(self, stimshape=None, batch_size=None):
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
            vec = vec - np.mean(vec)
            vec = vec/np.std(vec)
            X[:,i] = vec
        return X  
    
    def stimarray(self, stims, stimshape=None):
        stimshape = stimshape or self.stimshape
        length, height = stimshape
        assert length*height == stims.shape[1]
        buf = 1 # buffer pixels between stimuli
        M = stims.shape[0]
        
        # n and m are respectively the numbers of rows and columns of stimuli in the array
        if np.floor(np.sqrt(M))**2 != M:
            n = int(np.ceil(np.sqrt(M/2.)))
            m = int(np.ceil(M/n))
        else:
            # M is a perfect square
            m = int(np.sqrt(M))
            n = m
        
        array = 0.5*np.ones((buf+n*(length+buf), buf+m*(height+buf)))
        k = 0
        
        # TODO: make this less ugly
        # Right now it loops over every pixel in the array
        for i in range(m):
            for j in range(n):
                if k < M:
                    normfactor = np.max(np.abs(stims[k,:]))
                    hstart = buf+i*(height+buf)
                    lstart = buf+j*(length+buf)
                    thestim = stims[k,:].reshape(length,height)/normfactor
                    array[lstart:lstart+length, hstart:hstart+height] = thestim
#                    for li in range(height):
#                        for lj in range(length):
#                            array[buf+(i)*(height+buf)+li, buf+(j)*(length+buf)+lj] =  \
#                            stims[k,lj+length*li]/normfactor
                k = k+1
                
        return array
        
class ImageSet(StimSet):
    """Currently only compatible with square images (but arbitrary patches)."""
    
    def __init__(self, data, stimshape=(16,16), batch_size=None, buffer=20):
        self.buffer = buffer
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
    
    def __init__(self, data, stimshape, pca, batch_size=None):
        self.pca = pca
        super().__init__(data, stimshape, batch_size)
        
    def stimarray(self, stims):
        reconst = self.pca.inverse_transform(stims)
        return super().stimarray(reconst, self.stimshape)