

import numpy as np
from scipy import sparse, fft
from lentils.operators import Operator, CompositeOperatorProduct, DiagonalOperator
from lentils.common import VisibilitySpace, FourierSpace, ImageSpace
from lentils.backend import libnufft, c_nufft


class FFTOperator(Operator):

    def __init__(self, image_space, nufft, **superargs):

        if not isinstance(image_space, ImageSpace): 
            raise TypeError("image_space must be of type ImageSpace")
        space_left = FourierSpace(image_space, nufft)
        super().__init__(space_left, image_space)

    def _matrixfree_forward(self, vec):
        out = self._cast_output(fft.rfft2(vec, s=self.space_right.shape, norm='backward'))
        return out

    def _matrixfree_transpose(self, vec):
        out = self._cast_output(fft.irfft2(vec, s=self.space_left.shape, norm='forward'))
        return out



class ZeroPaddingOperator(Operator):

    def __init__(self, image_space, nufft_op, **superargs):
        
        # TODO: provide more compiler options than the c struct
        if not isinstance(image_space, ImageSpace): 
            raise TypeError("image_space must be of type ImageSpace")
        if not isinstance(nufft_op, NUFFTOperator): 
            raise ValueError("nufft_op must be of type NUFFTOperator")

        # TODO: add a matrix operator option
        self._cpars = nufft_op._cpars

        # finish up and pass along supers
        super().__init__(nufft_op.padded_space, image_space)
        
    def _matrixfree_forward(self, vec):
        out = self.space_left.new_vector()
        libnufft.zero_pad(self._cpars, vec, out, 0)
        return out

    def _matrixfree_transpose(self, vec):
        out = self.space_left.new_vector()
        libnufft.zero_pad(self._cpars, vec, out, 1) 
        return out


class GriddingOperator(Operator):

    #def __init__(self, symbol, space_left, space_right, combine_channels=True, combine_stokes=True, **superargs):
    def __init__(self, space_vis, space_fourier, nufft, **superargs):
        
        if not isinstance(space_fourier, FourierSpace): 
            raise TypeError("space_fourier must be of type FourierSpace")
        if not isinstance(space_vis, VisibilitySpace): 
            raise TypeError("space_vis must be of type VisibilitySpace")
        self._cpars = nufft._cpars

        umax = np.max(space_vis.channels)*np.max(np.abs(space_vis.points[:,0]))
        vmax = np.max(space_vis.channels)*np.max(np.abs(space_vis.points[:,1]))
        if umax+self._cpars.du*(self._cpars.wsup+1) > 0.5*self._cpars.du*self._cpars.nx_pad \
            or vmax+self._cpars.dv*(self._cpars.wsup+1) > 0.5*self._cpars.dv*self._cpars.ny_pad: 
                raise ValueError("Visibility space contains points higher than the Nyquist frequency of the grid.")

        # finish up and pass along supers
        super().__init__(space_vis, space_fourier)

    def _matrixfree_forward(self, vec):
        out = self.space_left.new_vector()
        libnufft.grid_cpu(self._cpars, out, vec, 0) 
        return out

    def _matrixfree_transpose(self, vec):
        out = self.space_left.new_vector()
        libnufft.grid_cpu(self._cpars, vec, out, 1) 
        return out


class NUFFTOperator(CompositeOperatorProduct):

    # TODO: implement in C
    def __init__(self, vis_space, image_space, pad_factor=2, kernel_support=4, combine_channels=True, combine_stokes=True, **superargs):
        
        if not isinstance(image_space, ImageSpace): 
            raise TypeError("image_space must be of type ImageSpace")
        if not isinstance(vis_space, VisibilitySpace): 
            raise TypeError("vis_space must be of type VisibilitySpace")

        # initializ the c backend
        self._cpars = c_nufft()
        image_shape = image_space.shape
        image_bounds = image_space._bounds
        libnufft.init_nufft(
                self._cpars, image_shape[0], image_shape[1], image_bounds[0,0], 
                image_bounds[1,0], image_bounds[0,1], image_bounds[1,1],
                pad_factor, kernel_support, vis_space.num_rows, vis_space.num_channels, vis_space.num_stokes, 
                vis_space.points, vis_space.channels)

        # make the intermediate spaces
        # the following operator will need them
        self.image_space = image_space
        ######## TODO: move this into zero-pad operator, use cpars
        imshape_list = [n for n in self.image_space._shape]
        padshape_list = imshape_list
        padshape_list[-2:] = [self._cpars.nx_pad, self._cpars.ny_pad]
        self.padded_space = image_space.copy() 
        self.padded_space._shape = tuple(padshape_list)
        self.padded_space._bounds[0,0] -= self.image_space._dx[0]*self._cpars.padx
        self.padded_space._bounds[1,0] += self.image_space._dx[0]*self._cpars.padx
        self.padded_space._bounds[0,1] -= self.image_space._dx[1]*self._cpars.pady
        self.padded_space._bounds[1,1] += self.image_space._dx[1]*self._cpars.pady
        #######
        self.zpad = ZeroPaddingOperator(image_space, self)

        # Apodization correction
        apod_data = image_space.new_vector()
        libnufft.evaluate_apodization_correction(self._cpars, apod_data)
        self.apod = DiagonalOperator(image_space, apod_data)
        self.fft = FFTOperator(self.padded_space, self)
        self.grid = GriddingOperator(vis_space, self.fft.space_left, self)

        # Finish up
        super().__init__([self.grid, self.fft, self.zpad, self.apod])



class DFTOperator(Operator):

    def __init__(self, vis_space, image_space, **superargs):
        
        if not isinstance(image_space, ImageSpace): 
            raise TypeError("image_space must be of type ImageSpace")
        if not isinstance(vis_space, VisibilitySpace): 
            raise TypeError("vis_space must be of type VisibilitySpace")

        # initialize the c backend
        self._cpars = c_nufft()
        pad_factor=2
        kernel_support=4
        image_shape = image_space.shape
        image_bounds = image_space._bounds
        libnufft.init_nufft(
                self._cpars, image_shape[0], image_shape[1], image_bounds[0,0], 
                image_bounds[1,0], image_bounds[0,1], image_bounds[1,1],
                pad_factor, kernel_support, vis_space.num_rows, vis_space.num_channels, vis_space.num_stokes, 
                vis_space.points, vis_space.channels)

        # make the matrix
        num_rows = 2*vis_space.size # complex visibilities, so multiply rows by 2
        nnz_per_row = np.sum(image_space.mask)
        num_vals = num_rows*nnz_per_row
        row_inds = np.zeros(num_rows+1, dtype=np.int32) 
        cols = np.zeros(num_vals, dtype=np.int32) 
        vals = np.zeros(num_vals, dtype=np.float64) 
        libnufft.dft_matrix_csr(self._cpars, image_space.mask, row_inds, cols, vals)
        self._mat = sparse.csr_matrix((vals,cols,row_inds), shape=(num_rows,image_space.size))

        # Finish up
        super().__init__(vis_space, image_space)


