# -*- coding: utf-8 -*-
"""

Note: This skeleton file can be safely removed if not needed!

"""

import numpy as np
from astropy.convolution import convolve, convolve_fft
from scipy import sparse
from copy import copy


from lentils.common import Space, VisibilitySpace, FourierSpace, ImageSpace, DelaunaySpace 
from lentils.backend import libtriangles, libraster, libnufft



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
        if self.has_matrix:
            return self._mat
        else:
            return None

    @property
    def has_matrix(self):        
        return hasattr(self, '_mat')

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

    def _cast_output(self, vec):
        return np.ascontiguousarray(vec).view(self.space_left.dtype).reshape(self.space_left.shape)

    def apply(self, other):
        if isinstance(other, np.ndarray):
            if not broadcastable(self.space_right, other): 
                raise ValueError(f"Incompatible shapes! {self.space_right.shape}, {other.shape}")
            if self.has_matrix:
                # TODO: Broadcasting! Slice the arrays properly rather than just assume flatten works
                out = self._mat.dot(other.view(self._mat.dtype).flatten())
            else:
                out = self._matrixfree_forward(other)
            return self._cast_output(out)
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


class DiagonalOperator(Operator):

    def __init__(self, space, data, options=None, **superargs):

        if options == None:
            self._data = data
            matdim = np.product(space.shape)
        elif options == 'r2c':
            # makes a real matrix that represents complex entries
            # TODO: make this handle the complex part as well
            flatreal = data.flatten().real
            matdim = 2*flatreal.size
            self._data = np.zeros(matdim, dtype=np.float64)
            self._data[0::2] = flatreal
            self._data[1::2] = flatreal
        else:
            raise ValueError("options should be either None or r2c")

        self._mat = sparse.spdiags(self._data.flatten(), 0, matdim, matdim)
        super().__init__(space, space)


class PriorCovarianceOperator(Operator):

    def __init__(self, space, type, strength, **superargs):

        # for triangle meshes
        if isinstance(space, DelaunaySpace): 

            if type != 'gradient':
                raise NotImplementedError("Only gradient regularization for now") 

            row_inds = np.zeros(2*space.num_triangles+1, dtype=np.int32) 
            cols = np.zeros(3*2*space.num_triangles, dtype=np.int32) 
            vals = np.zeros(3*2*space.num_triangles, dtype=np.float64) 
            libtriangles.triangle_gradient_csr(space, row_inds, cols, vals)
            self._op = sparse.csr_matrix((vals,cols,row_inds), shape=(2*space.num_triangles,space.size))

        else:
            raise NotImplementedError("PriorCovarianceOperator only implemented for Delaunay mesh")

        self._lambda = strength
        self._mat = self._lambda*self._op.T.dot(self._op)
        self._transpose = self
        super().__init__(space, space)






