# -*- coding: utf-8 -*-
"""

Note: This skeleton file can be safely removed if not needed!

"""

import numpy as np
import astropy.io.fits as fits 
import astropy.constants as const
from scipy.spatial import Delaunay
from copy import deepcopy
import matplotlib.pyplot as plt

class Space:

    def __init__(self, name='Space', shape=(0,)):
        self._name = name 
        self._shape = shape
        pass;

    def new_vector(self, value=0.0):
        return np.full(self._shape, value, dtype=self.dtype)

    def copy(self):
        return deepcopy(self)

    @property
    def size(self):
        return np.product(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def num_channels(self):
        raise NotImplementedError

    @property
    def channels(self):
        raise NotImplementedError

    @property
    def points(self):
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}{self.shape}'


class VisibilitySpace(Space):

    def __init__(self, name='VisibilitySpace', channels=[0.], uvcoords=np.array([0.]), stokes=['I']):

        self.name = name
        self.dtype = np.complex128 
        self._channels = np.array(channels, dtype=np.float64).flatten() 
        num_channels = self._channels.shape[0] 
        self._uvw = np.array(uvcoords, dtype=np.float64).reshape((-1,3))
        num_rows = len(uvcoords) 
        num_stokes = len(stokes) 
        if num_stokes != 1 or stokes[0] != 'I':
            raise ValueError("Only stokes 'I' is allowed for now.,,")
        self._shape = (num_channels, num_stokes, num_rows)

    @property
    def num_channels(self):
        return self._shape[0]

    @property
    def channels(self):
        return self._channels

    @property
    def num_stokes(self):
        return self._shape[1]

    @property
    def num_rows(self):
        return self._shape[2]


    @property
    def shape(self):
        return (self._shape[2],)

    @property
    def points(self):
        return self._uvw



class FourierSpace(Space):

    # TODO: deal with the transpose convention here

    # TODO: multiple channels

    # TODO: don't be dependent on c pars...

    def __init__(self, image_space, nufft, name='FourierSpace'):

        if not isinstance(image_space, ImageSpace): 
            raise TypeError("image_space must be of type ImageSpace")
        self._shape = image_space.shape
        self._axis_names = image_space._axis_names
        self._axis_names[-2:] = ['u','v']
        self._ndim = len(self._shape)
        self.name = name
        self.dtype = np.complex128 
        self.nufft = nufft

        #self._dx = np.array([(bounds[ax][1]-bounds[ax][0])/self.shape[ax] for ax in range(self._ndim)])

    def plot(self, vec, ax=None):
        raise NotImplementedError("Need to implement a good way to visualize gridded uv space")

    @property
    def shape(self):
        return self._shape[-2:]


class ImageSpace(Space):

    # TODO: deal with the transpose convention here

    # TODO: multiple channels

    def __init__(self, name='ImageSpace', shape=(128,128), bounds=[(-1.0,1.0),(-1.0,1.0)], channels=[], axis_names=['x','y'], mask=None):

        super().__init__()
        self.name = name
        self.dtype = np.float64 
        self._bounds = np.array(bounds, dtype=np.float64).reshape((-1,2))
        self._shape = shape
        self._ndim = len(self._shape)

        # load mask if there is one
        if mask is not None:
            with fits.open(mask) as f:
                self.mask = f['PRIMARY'].data[:,:].T.astype(np.bool_, order='C')
        else:
            self.mask = np.ones(shape, dtype=np.bool_)

        assert self._ndim == self._bounds.shape[0]
        self._dx = np.array([(bounds[ax][1]-bounds[ax][0])/self.shape[ax] for ax in range(self._ndim)])
        assert self._ndim == len(axis_names) 
        self._axis_names = axis_names

    def plot(self, vec, ax=None):

        if ax is None:
            ax = plt
        ax.imshow(vec._data, origin='lower', extent=self._bounds.flatten(), interpolation='nearest')
        ax.show()


    @property
    def points(self):
        centers = np.zeros(self.shape+(2,))
        xc = self._bounds[-2][0] + self._dx[0]*np.arange(self.shape[0]) 
        yc = self._bounds[-1][0] + self._dx[1]*np.arange(self.shape[1]) 
        # TODO: use pixel centers
        #xc = self._bounds[-2][0] + self._dx[0]*(0.5+np.arange(self.shape[0])) 
        #yc = self._bounds[-1][0] + self._dx[1]*(0.5+np.arange(self.shape[1])) 
        centers[:,:,0] = xc[:,None]
        centers[:,:,1] = yc[None,:] 
        return centers 

    @property
    def shape(self):
        return self._shape[-2:]


    @property
    def num_channels(self):
        return self._shape[0]


class DelaunaySpace(Space):

    def __init__(self, points, name='DelaunaySpace', channels=[0]):

        super().__init__()
        self.name = name
        self.dtype = np.float64
        self._tris = Delaunay(points)
        self.edge = np.unique(self._tris.convex_hull.flatten())
        self._shape = self._tris.points.shape[-2:]

    @property
    def size(self):
        return self._tris.points.shape[-2]

    @property
    def num_channels(self):
        return self._shape[0]

    @property
    def shape(self):
        # TODO: consistent definitions of shape and size
        return (self._tris.points.shape[-2],)

    @property
    def points(self):
       return self._tris.points.astype(np.float64)

    @property
    def num_tris(self):
       return self._tris.simplices.shape[-2]

    @property
    def tris(self):
       return self._tris.simplices




