# -*- coding: utf-8 -*-
"""

Note: This skeleton file can be safely removed if not needed!

"""

import numpy as np
from astropy.convolution import convolve, convolve_fft
from scipy import sparse
from copy import copy
import matplotlib.pyplot as plt
from astropy.io import fits


from lentils.common import Space, VisibilitySpace, FourierSpace, ImageSpace, DelaunaySpace 
from lentils.models import LensModel
from lentils.backend import libtriangles, libnufft, c_nufft, libraster



def broadcastable(a, b):
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(a.shape[::-1], b.shape[::-1]))

def broadcast(func):

    def broadcast_func(self, vec_in):
        #print("broadcasting {}.{}".format(type(self).__name__, func.__name__))
        #print(" - shape left, right =", self.space_left._shape, self.space_right._shape)

        # iterate over axes not included in the operator kernel 
        #for nin, nout in zip(reversed(vec_in.shape), reversed(self.space_right._shape)):
            #print(nin, nout)

        # ValueError: operands could not be broadcast together 

        #from itertools import product
        #b = np.broadcast
        #for p in np.broadcast(self.space_left.shape,self.space_right.shape):
        #for p in np.broadcast(product(self.space_left.shape),product(self.space_right.shape)):
            #print("p", p)

        #for channel in num_channels:

        vec_out = func(self, vec_in)
        return vec_out

    return broadcast_func

# Next TODO:
# - Concatenating explicit matrices
# - Re-organize c library loading into its own sub-module
# - Re-organize operator sub-modules
# - Broadcasting to full data shape, including channels and stokes
# - Get rid of space names...


class Operator:

    def __init__(self, space_left, space_right):
        if not isinstance(space_left, Space):
            raise ValueError("space_left must be a Space")
        if not isinstance(space_right, Space):
            raise ValueError("space_right must be a Space")
        self.space_right = space_right
        self.space_left = space_left

    def _make_transpose(self):
        tmp = copy(self)
        tmp.space_left, tmp.space_right = tmp.space_right, tmp.space_left 
        try: 
            # transpose the matrix if it exists
            tmp._mat = tmp._mat.T
        except AttributeError:
            # otherwise just swap apply <--> apply_transpose
            tmp._matrixfree_forward, tmp._matrixfree_transpose = tmp._matrixfree_transpose, tmp._matrixfree_forward
        return tmp

    @property
    def matrix(self):        
        try: 
            return self._mat
        except AttributeError:
            return None

    @property
    def T(self):
        try:
            return self._transpose 
        except AttributeError:
            self._transpose = self._make_transpose()
            self._transpose._transpose = self
            return self._transpose

    def __mul__(self, other):
        return self.apply(other)

    def __add__(self, other):
        return CompositeOperatorSum([self, other])

    def apply(self, other):
        if isinstance(other, np.ndarray):
            if not broadcastable(self.space_right, other): 
                raise ValueError(f"Incompatible shapes! {self.space_right.shape}, {other.shape}")
            try: 
                # TODO: Broadcasting! Slice the arrays properly rather than just assume flatten works
                # TODO: make spaces aware of real -> complex mapping!
                return self._mat.dot(other.flatten()).reshape(self.space_left.shape)
            except AttributeError:
                return self._matrixfree_forward(other)
        elif isinstance(other, Operator):
            return CompositeOperatorProduct([self, other])
        raise TypeError("Operator.apply() can only be used on numpy.ndarray or Operator")

    def _matrixfree_forward(self, other):
        raise NotImplementedError

    def _matrixfree_transpose(self, other):
        raise NotImplementedError

    def __repr__(self):
        return f'<{type(self).__name__}: {self.space_right} -> {self.space_left}>'
        

class CompositeOperatorProduct(Operator):

    def __init__(self, oplist):

        # concatenate sub-operators
        self.subops = []
        for op in oplist:
            if not isinstance(op, Operator):
                raise TypeError("Composite operator needs a list of only Operators")
            if isinstance(op, CompositeOperatorProduct):
                self.subops += op.subops
            else:
                self.subops.append(op)

        # check shapes and make matrix if possible
        for op1, op2 in zip(self.subops[:-1], self.subops[1:]):
            if not broadcastable(op1.space_right, op2.space_left):
                raise ValueError(f"Incompatible shapes! {op1.space_right.shape}, {op2.space_left.shape}")

        super().__init__(oplist[0].space_left, oplist[-1].space_right)

    def _make_transpose(self):
        # TODO: what about copying matrices.... Could save some compute time
        return CompositeOperatorProduct([op.T for op in self.subops[::-1]])

    def _matrixfree_forward(self, vec):
        out = vec
        for op in self.subops[::-1]:
            out = op * out
        return out


class CompositeOperatorSum(Operator):

    def __init__(self, oplist):

        # concatenate sub-operators
        self.subops = []
        for op in oplist:
            if not isinstance(op, Operator):
                raise TypeError("Composite operator needs a list of only Operators")
            if isinstance(op, CompositeOperatorSum):
                self.subops += op.subops
            else:
                self.subops.append(op)

        # check shapes and make matrix if possible
        for op1, op2 in zip(self.subops[:-1], self.subops[1:]):
            if not broadcastable(op1.space_right, op2.space_right):
                raise ValueError(f"Incompatible shapes! {op1.space_right.shape}, {op2.space_right.shape}")
            if not broadcastable(op1.space_left, op2.space_left):
                raise ValueError(f"Incompatible shapes! {op1.space_left.shape}, {op2.space_left.shape}")

        super().__init__(oplist[0].space_left, oplist[-1].space_right)

    def _make_transpose(self):
        # TODO: what about copying matrices.... Could save some compute time
        return CompositeOperatorSum([op.T for op in self.subops])

    def _matrixfree_forward(self, vec):
        out = self.subops[0] * vec
        for op in self.subops[1:]:
            out += op * vec
        return out 


# base class for lens operator types
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
            libnufft.convolution_matrix_csr(
                    image_space.shape[0], image_space.shape[1], 
                    self._kernel.shape[-2], self._kernel.shape[-1],
                    self._kernel, row_inds, cols, vals)
            self._mat = sparse.csr_matrix((vals,cols,row_inds), shape=(nrows,nrows))

        # finish up and pass along supers
        super().__init__(image_space, image_space)

    # TODO: channel-dependent PSF?
    def _matrixfree_forward(self, vec):
        return convolve_fft(vec, self._kernel, boundary='fill', fill_value=0.0)

    def _matrixfree_transpose(self, vec):
        # TODO: is this correct?
        return convolve(vec, self._kernel[::-1,::-1].copy(order='C'), boundary='fill', fill_value=0.0)

class GridOperator(Operator):

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


class DiagonalOperator(Operator):

    def __init__(self, space, data, **superargs):

        # TODO: generalize this to higher than space.shape 
        # TODO: check data and space compatibility
        self._data = data.astype(np.float64) 
        self._shape = self._data.shape
        self._mat = sparse.spdiags(self._data.flatten(), 0, np.product(self._shape), np.product(self._shape))
        super().__init__(space, space)


class PriorCovarianceOperator(Operator):

    def __init__(self, space, type, strength, **superargs):

        # for triangle meshes
        if isinstance(space, DelaunaySpace): 

            if type != 'gradient':
                raise NotImplementedError("Only gradient regularization for now") 

            row_inds = np.zeros(2*space.num_tris+1, dtype=np.int32) 
            cols = np.zeros(3*2*space.num_tris, dtype=np.int32) 
            vals = np.zeros(3*2*space.num_tris, dtype=np.float64) 
            libtriangles.triangle_gradient_csr(
                    space.num_tris, space.tris, space.points, row_inds, cols, vals)
            self._op = sparse.csr_matrix((vals,cols,row_inds), shape=(2*space.num_tris,space.size))

        else:
            raise NotImplementedError("PriorCovarianceOperator only implemented for Delaunay mesh")

        self._lambda = strength
        self._mat = self._lambda*self._op.T.dot(self._op)
        self._transpose = self
        super().__init__(space, space)




class FFTOperator(Operator):

    def __init__(self, image_space, nufft, **superargs):

        if not isinstance(image_space, ImageSpace): 
            raise TypeError("image_space must be of type ImageSpace")
        space_left = FourierSpace(image_space, nufft)
        self._cpars = nufft._cpars

        # finish up and pass along supers
        super().__init__(space_left, image_space)

    def _matrixfree_forward(self, vec):
        out = self.space_left.new_vector()
        libnufft.fft2d(self._cpars, vec, out, 0)
        return out

    def _matrixfree_transpose(self, vec):
        out = self.space_left.new_vector()
        libnufft.fft2d(self._cpars, out, vec, 1)
        return out



# base class for lens operator types
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

        #for field_name, field_type in self._cpars._fields_:
            #print(field_name, getattr(self._cpars, field_name))

        # make the intermediate spaces
        # the following operator will need them
        self.image_space = image_space
        ######## TODO: move this into zero-pad operator
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
        self.grid = GridOperator(vis_space, self.fft.space_left, self)

        # Finish up
        super().__init__([self.grid, self.fft, self.zpad, self.apod])



class DFTOperator(Operator):

    # TODO: implement in C
    def __init__(self, vis_space, image_space, **superargs):
        
        if not isinstance(image_space, ImageSpace): 
            raise TypeError("image_space must be of type ImageSpace")
        if not isinstance(vis_space, VisibilitySpace): 
            raise TypeError("vis_space must be of type VisibilitySpace")

        # TODO: make the matrices operate on the native space shape
        self._mat = np.zeros((vis_space.num_rows, np.product(image_space.shape)), dtype=np.complex128)
        imcoords = image_space.points.reshape((-1,2))*(4.8481368111e-6)
        for i, k in enumerate(vis_space.points[:,:2]*vis_space.channels[0]):
            arg = -2*np.pi*(k[0]*imcoords[:,0]-k[1]*imcoords[:,1])
            self._mat[i] = np.exp(-1j*arg)
            #self._mat[2*i+0] = np.cos(arg)
            #self._mat[2*i+1] = np.sin(arg)

        # Finish up
        super().__init__(vis_space, image_space)




# base class for lens operator types
class LensOperator(Operator):
    def __init__(self, space_left, space_right, lensmodel):
        self._lensmodel = lensmodel
        super().__init__(space_left, space_right)
    def get_casted_points():
        raise NotImplementedError
    def compute_entries():
        raise NotImplementedError


class ManifoldLensOperator(LensOperator):

    def __init__(self, image_space, source_space, lensmodel, z_src, ncasted=1, mask=None, **superargs):

        if not isinstance(image_space, ImageSpace):
            raise TypeError("image_space must be ImageSpace")
        if not isinstance(source_space, ImageSpace):
            raise TypeError("source_space must be ImageSpace")
        if not isinstance(lensmodel, LensModel):
            raise TypeError("lensmodel must be LensModel")

        # load the image-plane mask for ray casting
        # this is different fromthe data mask
        if mask is not None:
            if mask.shape != image_space.shape:
                raise ValueError("mask shape must match the image space")
            image_mask = mask.astype(np.bool_)
        else:
            image_mask = np.ones(image_space.shape, dtype=np.bool_)

        print("Made the manifold lens op")

        # finish up and pass along supers
        super().__init__(image_space, source_space, lensmodel)

    
    def rasterize(self, triangles):

        source_space = self.space_right
        rastered = source_space.new_vector()
        b = source_space._bounds.flatten()
        libraster.rasterize_triangle(
                triangles, None, source_space.shape[0], source_space.shape[1], 
                b[0], b[1], b[2], b[3], rastered)
        return rastered 

    @broadcast
    def _matrixfree_forward(self, vec):
        out = self.space_left.new_vector()
        out_masked = out[self._image_mask]
        out_masked[self._casted_inds] = vec
        out_masked[self._uncasted_inds] = np.sum(self._uncasted_weights*vec[self._uncasted_tris], axis=-1)
        out[self._image_mask] = out_masked
        return out

    @broadcast
    def _matrixfree_transpose(self, vec):
        #print('Lensop adjoint shape =', vec.shape)
        #out = vec[self._image_mask][self._casted_inds]
        out = vec[self._image_mask][self._casted_inds]
        #out = vec[self._image_mask][self._casted_inds]
        #out_masked = out[self._image_mask]
        #out_masked[self._casted_inds] = vec
        #out_masked[self._uncasted_inds] = np.sum(self._uncasted_weights*vec[self._uncasted_tris], axis=-1)
        #out[self._image_mask] = out_masked
        return out




# Vegetti and Koopmans delaunay tessellation
class DelaunayLensOperator(LensOperator):

    def __init__(self, image_space, lensmodel, z_src, ncasted=1, mask=None, **superargs):

        if not isinstance(image_space, ImageSpace):
            raise TypeError("image_space must be ImageSpace")
        if not isinstance(lensmodel, LensModel):
            raise TypeError("lensmodel must be LensModel")

        # load the image-plane mask for ray casting
        # this is different from the data mask
        if mask is not None:
            if mask.shape != image_space.shape:
                raise ValueError("mask shape must match the image space")
            image_mask = mask.astype(np.bool_)
        else:
            image_mask = np.ones(image_space.shape, dtype=np.bool_)

        # make image-plane masks containing "casted" and "uncasted" points
        image_points = image_space.points
        casted_mask = np.zeros_like(image_mask)
        uncasted_mask = np.ones_like(image_mask)
        casted_mask[::ncasted,::ncasted] = True
        uncasted_mask[::ncasted,::ncasted] = False 
        casted_mask *= image_mask
        uncasted_mask *= image_mask

        # deflect the rays
        # Get Delaunay triangulation and triangle indices of uncasted rays
        self._casted_points = lensmodel.deflect(image_points[casted_mask], z_s=z_src)
        self._uncasted_points = lensmodel.deflect(image_points[uncasted_mask], z_s=z_src)
        source_space = DelaunaySpace(self._casted_points) 
        uncasted_tri_inds = source_space._tris.find_simplex(self._uncasted_points)

        # make the matrix
        num_casted = np.sum(casted_mask)
        num_uncasted = np.sum(uncasted_mask)
        num_rows = image_space.size
        num_vals = num_casted + 3*num_uncasted
        row_inds = np.zeros(num_rows+1, dtype=np.int32) 
        cols = np.zeros(num_vals, dtype=np.int32) 
        vals = np.zeros(num_vals, dtype=np.float64) 
        libtriangles.delaunay_lens_matrix_csr(
            image_space.shape[0], image_space.shape[1], ncasted, 
            image_mask, self._uncasted_points, uncasted_tri_inds, 
            source_space.points, source_space._tris.simplices,
            row_inds, cols, vals)
        self._mat = sparse.csr_matrix((vals,cols,row_inds), shape=(image_space.size,source_space.size))

        # finish up and pass along supers
        super().__init__(image_space, source_space, lensmodel)




