# -*- coding: utf-8 -*-
"""

Note: This skeleton file can be safely removed if not needed!

"""

import numpy as np
import astropy.io.fits as fits 
import astropy.constants as const
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from lentils.operators import DiagonalOperator
from lentils.common import ImageSpace, VisibilitySpace 

class Dataset:

    def __init__(self, name='DataSet', data=None, sigma=None, mask=None, space=None, dtype=np.float64):
        self.name = name 
        self.space = space 
        self.data = data.astype(dtype)
        self.sigma = sigma.astype(np.float64) # we treat sigma as always real
        self.mask = mask.astype(np.bool_) 

    def __repr__(self):
        return '{} with space {}'.format(self.name, self.space) 

    @property
    def covariance_operator(self):
        diag = np.zeros_like(self.sigma)
        diag[self.mask] = self.sigma[self.mask]**-2 
        return DiagonalOperator(self.space, diag)

    @staticmethod
    def visibilities_from_uvfits(fitsfile, combine_stokes=True, mfs=True):

        # open the fits file 
        allhdu = fits.open(fitsfile, memmap=True)
        hdu = allhdu['PRIMARY']
        data = hdu.data
        header = hdu.header

        # get the header into a dict that we can access via the axis name
        axes = {}
        for ax in range(1,header['NAXIS']+1):
            nax = header['NAXIS%d'%ax]
            if nax == 0:
                continue
            axtmp = {'NAXIS': nax,}
            for label in ['CRVAL','CDELT','CRPIX','CROTA']:
                axtmp[label] = header['%s%d'%(label,ax)]
            axes[header['CTYPE%d'%ax]] = axtmp
        #for key in header:
            #print(key,':',header[key])
        #for key in axes:
            #print(key,':',axes[key])

        # shape the data
        num_rows = header['GCOUNT']
        num_spw = axes['IF']['NAXIS']
        num_channels = axes['FREQ']['NAXIS']
        num_stokes = axes['STOKES']['NAXIS']
        ref_freq = axes['FREQ']['CRVAL']
        ref_ch = axes['FREQ']['CRPIX']-1 # convert from 1-based indexing

        # get frequency data
        # TODO: check them against the C code
        fqhead = allhdu['AIPS FQ'].header
        fqdata = allhdu['AIPS FQ'].data
        #print(fqdata['IF FREQ'])
        #print(fqdata['CH WIDTH'])
        channels = (np.multiply.outer(fqdata['CH WIDTH'].reshape((num_spw)), (np.arange(num_channels)-ref_ch)) \
                + fqdata['IF FREQ'].reshape((num_spw,1)) + ref_freq).flatten()
        num_channels = num_channels*num_spw 
        assert channels.size == num_channels

        # read in data
        # TODO: original data is only 32-bit...
        uvcoords = np.array([data['UU'],data['VV'],data['WW']]).T.astype(np.float64, order='C')
        rawvisdata = data['DATA'][:,0,0,:,:,:,0] + 1j*data['DATA'][:,0,0,:,:,:,1]
        rawweights = data['DATA'][:,0,0,:,:,:,2]
        rawmask = (rawweights > 0.0) 
        rawsigma = np.zeros_like(rawweights)
        rawsigma[rawmask] = rawweights[rawmask]**-0.5

        # combine stokes parameters if desired
        # TODO: add a check of the stokes parameter names
        #combine_stokes=False
        combine_stokes=True
        if combine_stokes:

            rawmask = np.all(rawmask,axis=-1)
            #print('rawmask shape =', rawmask.shape)
             
            # TODO: check this math 
            rr = rawvisdata[:,:,:,0] 
            ll = rawvisdata[:,:,:,1] 
            rawvisdata = 0.5*(rr+ll)
            #print('rawvisdata shape =', rawvisdata.shape)
            srr = rawsigma[:,:,:,0] 
            sll = rawsigma[:,:,:,1] 
            rawsigma = np.zeros_like(srr)
            rawsigma[rawmask] = 0.5*np.sqrt(srr[rawmask]**2+sll[rawmask]**2)
            #print('rawsigma shape =', rawsigma.shape)
            num_stokes = 1
        else:
            rawvisdata = rawvisdata[:,:,:,0] 
            rawsigma = rawsigma[:,:,:,0] 
            rawmask = rawmask[:,:,:,0] 

        # shape and store
        uvspace = VisibilitySpace(name="VisibilitySpace from file {}".format(fitsfile), channels=channels, uvcoords=uvcoords)
        reordered_data = np.moveaxis(rawvisdata.reshape((uvspace.num_rows, uvspace.num_channels, uvspace.num_stokes)), [0,1,2], [2,0,1]).astype(np.complex128, order='C')
        reordered_sigma = np.moveaxis(rawsigma.reshape((uvspace.num_rows, uvspace.num_channels, uvspace.num_stokes)), [0,1,2], [2,0,1]).astype(np.float64, order='C')
        reordered_mask = np.moveaxis(rawmask.reshape((uvspace.num_rows, uvspace.num_channels, uvspace.num_stokes)), [0,1,2], [2,0,1]).astype(np.bool_, order='C')
        dataset = Dataset(name='Dataset from file {}'.format(fitsfile), \
                data=reordered_data, sigma=reordered_sigma, mask=reordered_mask, space=uvspace, dtype=np.complex128)

        allhdu.close()
        return dataset 


    @staticmethod
    def image_from_fits(datafits, maskfits=None, noise=None, bounds=[(-1.0,1.0),(-1.0,1.0)], dtype=np.float64):

        # TODO: make the fits reader smarter
        # get the raw data
        # assumes order (channel,y,x), so we take the transpose
        with fits.open(datafits) as f:
            data = f['PRIMARY'].data.T.copy()
        if data.ndim != 2:
            raise ValueError("data must be 2D for now...")

        # load mask if there is one
        mask = None
        if maskfits is not None:
            with fits.open(maskfits) as f:
                mask = f['PRIMARY'].data[:,:].T.copy()

        # load noise if given 
        sigma = None
        if noise is not None:
            if isinstance(noise, str):
                sigma = fits.open(noise)['PRIMARY'].data[:,:].T.copy()
            elif isinstance(noise, float):
                sigma = noise*np.ones_like(data)

        # create a space and a vector
        space = ImageSpace(shape=data.shape, bounds=bounds)
        dataset = Dataset(name='Dataset from file {}'.format(datafits), \
                data=data, sigma=sigma, mask=mask, space=space)
        return dataset


    @property
    def points(self):
        centers = self._dx*(0.5+np.mgrid[0:self.shape[1],0:self.shape[0]].T).astype(np.float64)
        centers[...,0] += self._bounds[-2][0]
        centers[...,1] += self._bounds[-1][0]
        return centers 


    @property
    def size(self):
        return np.product(self._shape)

    @property
    def shape(self):
        return self._shape[-2:]


    @property
    def num_channels(self):
        return self._shape[0]


