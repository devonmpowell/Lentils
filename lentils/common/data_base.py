# -*- coding: utf-8 -*-
"""

Note: This skeleton file can be safely removed if not needed!

"""

import numpy as np
import astropy.io.fits as fits 
import astropy.constants as const
from scipy.spatial import Delaunay
from lentils.operators import DiagonalOperator, ConvolutionOperator, NUFFTOperator, CompositeOperatorProduct
from lentils.common import ImageSpace, VisibilitySpace 
from .data_util import _load_uvfits

class Dataset:

    def __init__(self, name='DataSet', data=None, sigma=None, mask=None, space=None, dtype=np.float64):
        self.name = name 
        self.space = space 
        self.data = data.astype(dtype)
        self.sigma = sigma.astype(np.float64) # we treat sigma as always real
        self.mask = mask.astype(np.bool_) 
        self.dtype = dtype

    def __repr__(self):
        return '{} with space {}'.format(self.name, self.space) 

    @property
    def blurring_operator(self):
        raise NotImplementedError

    @property
    def blurred_covariance_operator(self):
        raise NotImplementedError

    @property
    def covariance_operator(self):
        try:
            return self._covariance_op
        except AttributeError:
            diag = np.zeros_like(self.sigma, dtype=self.space.dtype)
            diag[self.mask] = self.sigma[self.mask]**-2 
            self._covariance_op = DiagonalOperator(self.space, diag)
            return self._covariance_op

    @property
    def size(self):
        return np.product(self._shape)

    @property
    def shape(self):
        return self._shape[-2:]

    @property
    def num_channels(self):
        return self._shape[0]









class RadioDataset(Dataset):

    def __init__(self, file, image_space=None, combine_stokes=True, mfs=True):

        uvspace, data, sigma, mask = _load_uvfits(file, combine_stokes)

        super().__init__(name='Dataset from file {}'.format(file), \
                data=data, sigma=sigma, mask=mask, space=uvspace, dtype=np.complex128)

        if image_space is not None:
            if not isinstance(image_space, ImageSpace):
                raise TypeError("image_space must be of type ImageSpace.")
            self.image_space = image_space
            self.nufft_operator = NUFFTOperator(self.space, self.image_space)

    @property
    def blurring_operator(self):
        return self.nufft_operator

    @property
    def blurred_covariance_operator(self):
        try:
            return self._fcf
        except AttributeError:
            fft = self.nufft_operator.fft
            zpad = self.nufft_operator.zpad
            zpfft = fft*zpad
            db = self.dirty_beam
            db = np.roll(db,[db.shape[0]//2,db.shape[1]//2],axis=[0,1])
            dbfft = fft*db/np.product(db.shape)
            dbop = DiagonalOperator(fft.space_left, dbfft)
            self._fcf = CompositeOperatorProduct([zpfft.T, dbop, zpfft])
            return self._fcf

    @property
    def dirty_image(self):
        try:
            return self._dirty_image
        except AttributeError:
            self._dirty_image = self.nufft_operator.T * self.covariance_operator * self.data
            return self._dirty_image

    @property
    def dirty_beam(self):
        try:
            return self._dirty_beam
        except AttributeError:
            # TODO: use NUFFT padded space dimensions
            #padded = self.nufft_operator.padded_space
            #nx = padded._shape[0]
            #ny = padded._shape[1]
            #rx = 0.5*(padded._bounds[1,0]-padded._bounds[0,0]) 
            #ry = 0.5*(padded._bounds[1,1]-padded._bounds[0,1]) 
            nx = 2*self.image_space._shape[0]
            ny = 2*self.image_space._shape[1]
            rx = self.image_space._dx[0]*self.image_space._shape[0]
            ry = self.image_space._dx[1]*self.image_space._shape[1]
            self._space_beam = ImageSpace(shape=(nx,ny), bounds=[(-rx,rx), (-ry,ry)])
            self._nufft_beam = NUFFTOperator(self.space, self._space_beam)
            ones = self.space.new_vector(1.0+0.0j) 
            self._dirty_beam = self._nufft_beam.T * self.covariance_operator * ones
            return self._dirty_beam



class OpticalDataset(Dataset):

    def __init__(self, datafits, mask=None, noise=None, psf=None, psf_support=None, bounds=[(-1.0,1.0),(-1.0,1.0)], dtype=np.float64):

        # get the raw data
        # assumes order (channel,y,x), so we take the transpose
        with fits.open(datafits) as f:
            data = f['PRIMARY'].data.T.copy()
        if data.ndim != 2:
            raise ValueError("data must be 2D for now...")

        # load noise if given 
        sigma = None
        if noise is not None:
            if isinstance(noise, str):
                sigma = fits.open(noise)['PRIMARY'].data[:,:].T.copy()
            elif isinstance(noise, float):
                sigma = noise*np.ones_like(data)

        # create a space and a vector
        space = ImageSpace(shape=data.shape, bounds=bounds, mask=mask)
        super().__init__(name='Dataset from file {}'.format(datafits), \
                data=data, sigma=sigma, mask=space.mask, space=space)

        # load the psf operator
        if psf is not None:
            self.psf_operator = ConvolutionOperator(space, fitsfile=psf, kernelsize=psf_support, fft=False)
            self.bcb_operator = self.psf_operator.T * self.covariance_operator * self.psf_operator 


    @property
    def blurring_operator(self):
        return self.psf_operator

    @property
    def blurred_covariance_operator(self):
        return self.bcb_operator 


