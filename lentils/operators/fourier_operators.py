

import numpy as np
from scipy import sparse, fft
from astropy.io import fits
from lentils.operators import Operator, CompositeOperatorProduct, DiagonalOperator
from lentils.common import VisibilitySpace, FourierSpace, ImageSpace
from lentils.backend import libnufft, c_nufft


class FFTOperator(Operator):

    def __init__(self, image_space, **superargs):
        space_fourier = FourierSpace(image_space)
        super().__init__(space_fourier, image_space)

    def _matrixfree_forward(self, vec):
        out = self._cast_output(fft.rfft2(vec, s=self.space_right.shape[-2:], norm='backward'))
        return out

    def _matrixfree_transpose(self, vec):
        out = self._cast_output(fft.irfft2(vec, s=self.space_left.shape[-2:], norm='forward'))
        return out


class ZeroPaddingOperator(Operator):

    def __init__(self, image_space, pad_factor, **superargs):
        
        # compute symmetric padding on both sides
        # rounding up for odd nx, ny
        self.pad_factor = int(pad_factor)
        padx = ((self.pad_factor-1)*image_space.nx+1)//2
        pady = ((self.pad_factor-1)*image_space.ny+1)//2
        shape = (image_space.nx+2*padx, image_space.ny+2*pady)
        dx, dy = image_space.dx, image_space.dy
        bounds = image_space.bounds.copy()
        bounds += np.array([-padx*dx, padx*dx, -pady*dy, pady*dy])
        padded_space = ImageSpace(shape=shape, bounds=bounds, 
                channels=image_space.channels, mask=None)
        super().__init__(padded_space, image_space)
        
    def _matrixfree_forward(self, vec):
        out = self.space_left.new_vector()
        libnufft.zero_pad(self.space_right, vec, self.space_left, out, 0) 
        return out

    def _matrixfree_transpose(self, vec):
        out = self.space_left.new_vector()
        libnufft.zero_pad(self.space_left, out, self.space_right, vec, 1) 
        return out

class ApodizationCorrectionOperator(DiagonalOperator):

    def __init__(self, image_space, padded_space, kb_beta, kernel_support, **superargs):
        
        # according to Beatty+2005
        self.kernel_support = int(kernel_support)
        self.kb_beta = float(kb_beta)
        cx = (np.arange(image_space.nx)-0.5*image_space.nx)/padded_space.nx
        cy = (np.arange(image_space.ny)-0.5*image_space.ny)/padded_space.ny
        argx = np.sqrt(self.kb_beta**2-(2*np.pi*self.kernel_support*cx)**2)
        argy = np.sqrt(self.kb_beta**2-(2*np.pi*self.kernel_support*cy)**2)
        sinhcx = np.sinh(argx)/argx
        sinhcy = np.sinh(argy)/argy
        apod_data = 1.0/np.outer(sinhcx, sinhcy)
        super().__init__(image_space, apod_data)


class GriddingOperator(Operator):

    def __init__(self, space_vis, space_fourier, kb_beta, kernel_support, **superargs):
        
        self.kernel_support = int(kernel_support)
        self.kb_beta = float(kb_beta)
        iumax = np.max(space_vis.channels)*np.max(np.abs(space_vis.uvw[:,0]))/space_fourier.du
        ivmax = np.max(space_vis.channels)*np.max(np.abs(space_vis.uvw[:,1]))/space_fourier.dv
        if iumax+self.kernel_support+1 > 0.5*space_fourier.nu or ivmax+self.kernel_support+1 > 0.5*space_fourier.nv: 
                raise ValueError("Visibility space contains points higher than the Nyquist frequency of the grid.")

        # finish up and pass along supers
        super().__init__(space_vis, space_fourier)

    def _matrixfree_forward(self, vec):
        out = self.space_left.new_vector()
        libnufft.grid_cpu(self.space_left, out, self.space_right, vec, self.kernel_support, self.kb_beta, 0) 
        return out

    def _matrixfree_transpose(self, vec):
        out = self.space_left.new_vector()
        libnufft.grid_cpu(self.space_right, vec, self.space_left, out, self.kernel_support, self.kb_beta, 1) 
        return out


class NUFFTOperator(CompositeOperatorProduct):

    def __init__(self, vis_space, image_space, pad_factor=2, kernel_support=4, combine_channels=True, combine_stokes=True, **superargs):

        # NUFFT-specific attributes
        self.image_space = image_space
        self.pad_factor = int(pad_factor)
        self.kernel_support = int(kernel_support)

        # Kaiser-Bessel optimal beta from Beatty+2005
        wsup = self.kernel_support
        padfac = self.pad_factor
        self.kb_beta = np.pi*np.sqrt((2.0*wsup/padfac)**2*(padfac-0.5)**2 - 0.8)
 
        # zero-padding
        self.zpad = ZeroPaddingOperator(self.image_space, self.pad_factor)
        self.padded_space = self.zpad.space_left

        # Apodization correction
        self.apod = ApodizationCorrectionOperator(self.image_space, 
                self.padded_space, self.kb_beta, self.kernel_support)
       
        # FFT
        self.fft = FFTOperator(self.padded_space)
        self.fourier_space = self.fft.space_left

        # Gridder
        self.gridder = GriddingOperator(vis_space, self.fourier_space, self.kb_beta, self.kernel_support)

        # Finish up
        super().__init__([self.gridder, self.fft, self.zpad, self.apod])



class DFTOperator(Operator):

    def __init__(self, vis_space, image_space, **superargs):
        
        # make the matrix
        num_rows = 2*vis_space.size # complex visibilities, so multiply rows by 2
        nnz_per_row = np.sum(image_space.mask)
        num_vals = num_rows*nnz_per_row
        row_inds = np.zeros(num_rows+1, dtype=np.int32) 
        cols = np.zeros(num_vals, dtype=np.int32) 
        vals = np.zeros(num_vals, dtype=np.float64) 
        libnufft.dft_matrix_csr(vis_space, image_space, row_inds, cols, vals)
        self._mat = sparse.csr_matrix((vals,cols,row_inds), shape=(num_rows,image_space.size))

        # Finish up
        super().__init__(vis_space, image_space)


class ConvolutionOperator(Operator):

    def __init__(self, image_space, fitsfile=None, kerneldata=None, kernelsize=None, fwhm=0.1, fft=False, **superargs):

        if not isinstance(image_space, ImageSpace): 
            raise ValueError("image_space must be of type ImageSpace")

        # load the kernel image
        if fitsfile is not None:
            with fits.open(fitsfile) as f:
                data = f['PRIMARY'].data[:,:].T
        elif kerneldata is not None:
            data = kerneldata
        else:
            raise NotImplementedError("Need to put in generic Gaussian kernel")

        # crop the kernel if desired
        if kernelsize is not None and kernelsize > 0:
            padi = (data.shape[0]-kernelsize)//2
            padj = (data.shape[1]-kernelsize)//2
            kernel = data[padi:padi+kernelsize,padj:padj+kernelsize]
        else:
            kernel = data
        self._kernel = kernel.astype(np.float64, order='C')
        self._kernelsize = self._kernel.shape

        # normalize the kernel
        self._kernel /= np.sum(self._kernel)

        # make a matrix instead of fft (for small kernels)
        if not fft:
            nnz_per_row = np.product(self._kernelsize)
            nrows = np.product(image_space.shape)
            row_inds = np.zeros(nrows+1, dtype=np.int32) 
            cols = np.zeros(nrows*nnz_per_row, dtype=np.int32) 
            vals = np.zeros(nrows*nnz_per_row, dtype=np.float64) 
            libnufft.convolution_matrix_csr(image_space,
                    self._kernel.shape[-2], self._kernel.shape[-1], self._kernel, 
                    row_inds, cols, vals)
            self._mat = sparse.csr_matrix((vals,cols,row_inds), shape=(nrows,nrows))

        # finish up and pass along supers
        super().__init__(image_space, image_space)

    # TODO: channel-dependent PSF?
    def _matrixfree_forward(self, vec):
        # TODO: normalization?
        norm = 1.0 # np.product(self.space_right._dx)
        return norm**-2 * convolve_fft(vec, self._kernel, boundary='fill', fill_value=0.0) 

    def _matrixfree_transpose(self, vec):
        # TODO: is this correct?
        return convolve_fft(vec, self._kernel[::-1,::-1].copy(order='C'), boundary='fill', fill_value=0.0)


