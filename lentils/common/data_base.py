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

    def __init__(self, data=None, sigma=None, mask=None, space=None):
        self.space = space 
        self.dtype = space.dtype
        self.data = np.array(data, dtype=self.dtype, order='C')
        self.sigma = np.array(sigma, dtype=np.float64, order='C') # we treat sigma as always real
        self.mask = np.array(mask, dtype=np.bool_, order='C') # we treat sigma as always real

    def __repr__(self):
        return f'{self.__name__} with space {self.space}'

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

    #def __init__(self, file, image_space=None, combine_stokes=True, mfs=True):
    def __init__(self, uvfits=None, dirty_image_fits=None, dirty_beam_fits=None, image_space=None, combine_stokes=True, mfs=True):

        if uvfits is not None:
            uvspace, data, sigma, mask = _load_uvfits(uvfits, combine_stokes)
        else:
            uvspace = VisibilitySpace()
            data = [0.]
            sigma = [0.]
            mask = [0]

        super().__init__(data=data, sigma=sigma, mask=mask, space=uvspace)


        if image_space is not None:
            self.image_space = image_space
            self.nufft_operator = NUFFTOperator(self.space, self.image_space)

        if dirty_image_fits is not None and dirty_beam_fits is not None:

            with fits.open(dirty_image_fits) as f:
                hdu = f['PRIMARY']
                dirty_image = np.ascontiguousarray(hdu.data.T[:,:,0].astype(np.float64))

            with fits.open(dirty_beam_fits) as f:
                hdu = f['PRIMARY']
                dirty_beam = np.ascontiguousarray(hdu.data.T[:,:,0].astype(np.float64))

            if image_space is not None:

                dirty_image.shape = image_space.shape
                self._dirty_image = dirty_image
                self.beam_space = self._make_beam_space()
                dirty_beam.shape = self.beam_space.shape
                self._dirty_beam = dirty_beam

            else:
                # TODO: infer space shape and bounds from the header, if present
                #header = hdu.header
                #for item in header:
                    #print(item,header[item])
                pass

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
            pspace = self.nufft_operator.padded_space
            zpfft = fft*zpad
            db = self.dirty_beam
            db = np.roll(db,[pspace.nx//2,pspace.ny//2],axis=[-2,-1])
            dbfft = fft*db/np.product(db.shape)
            dbop = DiagonalOperator(fft.space_left, dbfft)
            self._fcf = CompositeOperatorProduct([zpfft.T, dbop, zpfft])
            return self._fcf

    @property
    def covariance_operator_dft(self):
        try:
            return self._covariance_op_dft
        except AttributeError:
            diag = np.zeros_like(self.sigma, dtype=self.space.dtype)
            diag[self.mask] = self.sigma[self.mask]**-2 
            self._covariance_op_dft = DiagonalOperator(self.space, diag, options='r2c')
            return self._covariance_op_dft

    @property
    def dirty_image(self):
        try:
            return self._dirty_image
        except AttributeError:
            self._dirty_image = self.nufft_operator.T * self.covariance_operator * self.data
            return self._dirty_image

    def _make_beam_space(self):
        # TODO: use NUFFT padded space dimensions?
        # May be necessary for BCB on odd-size grids
        #padded = self.nufft_operator.padded_space
        #nx = padded._shape[0]
        #ny = padded._shape[1]
        #rx = 0.5*(padded._bounds[1,0]-padded._bounds[0,0]) 
        #ry = 0.5*(padded._bounds[1,1]-padded._bounds[0,1]) 
        nx = 2*self.image_space.nx
        ny = 2*self.image_space.ny
        rx = self.image_space.dx*self.image_space.nx
        ry = self.image_space.dx*self.image_space.ny
        return ImageSpace(shape=(nx,ny), bounds=[(-rx,rx), (-ry,ry)])

    @property
    def dirty_beam(self):
        try:
            return self._dirty_beam
        except AttributeError:
            self.beam_space = self._make_beam_space()
            nufft_beam = NUFFTOperator(self.space, self.beam_space)
            ones = self.space.new_vector(1.0+0.0j) 
            self._dirty_beam = nufft_beam.T * self.covariance_operator * ones
            return self._dirty_beam



class OpticalDataset(Dataset):

    def __init__(self, datafits, mask=None, noise=None, psf=None, psf_support=None, bounds=[(-1.0,1.0),(-1.0,1.0)]):

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
        super().__init__(data=data, sigma=sigma, mask=space.mask, space=space)

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


