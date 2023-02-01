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
from lentils.backend import image_space_ctype, delaunay_space_ctype, fourier_space_ctype, visibility_space_ctype
from lentils.backend import c_int_p, c_double_p, c_bool_p

class Space:

    def __init__(self, shape=(0,), dtype=np.float64):
        self.shape = shape
        self.dtype = dtype

    def new_vector(self, value=0.0):
        return np.full(self.shape, value, dtype=self.dtype)

    def copy(self):
        return deepcopy(self)

    @property
    def size(self):
        return np.product(self.shape)

    @property
    def points(self):
        raise NotImplementedError

    def __repr__(self):
        return f'{type(self).__name__}{self.shape}'


class VisibilitySpace(Space, visibility_space_ctype):

    def __init__(self, channels=[0.], uvcoords=np.array([0.]), stokes=['I']):

        self.channels = np.array(channels, dtype=np.float64, order='C').flatten() 
        self._c_channels = self.channels.ctypes.data_as(c_double_p)
        self.num_channels = self.channels.shape[0] 
        self.uvw = np.array(uvcoords, dtype=np.float64, order='C').reshape((-1,3))
        self._c_uvw = self.uvw.ctypes.data_as(c_double_p)
        self.num_rows = len(uvcoords) 
        self.num_stokes = len(stokes) 
        if self.num_stokes != 1 or stokes[0] != 'I':
            raise ValueError("Only stokes 'I' is allowed for now.,,")
        super().__init__(shape=(self.num_channels, self.num_stokes, self.num_rows), dtype=np.complex128)

    @property
    def points(self):
        return self.uvw



class FourierSpace(Space, fourier_space_ctype):

    # TODO: deal with the transpose convention here

    # TODO: multiple channels

    def __init__(self, image_space, channels=[0.], stokes=['I']):

        self.nu, self.nv = image_space.nx, image_space.ny
        self.half_nv = self.nv//2+1
        arcsec_to_radians = 4.8481368111e-6
        lx = arcsec_to_radians*(image_space.xmax-image_space.xmin)
        ly = arcsec_to_radians*(image_space.ymax-image_space.ymin)
        self.du = 1.0/lx
        self.dv = 1.0/ly
        self.gcx = arcsec_to_radians*0.5*(image_space.xmax+image_space.xmin)
        self.gcy = arcsec_to_radians*0.5*(image_space.ymax+image_space.ymin)
        self.channels = np.array(channels, dtype=np.float64, order='C').flatten() 
        self._c_channels = self.channels.ctypes.data_as(c_double_p)
        self.num_channels = 1
        self.num_stokes = 1

        super().__init__(shape=(self.num_channels, self.num_stokes, self.nu, self.half_nv), dtype=np.complex128)


    def plot(self, vec, ax=None):
        raise NotImplementedError("Need to implement a good way to visualize gridded uv space")


class ImageSpace(Space, image_space_ctype):

    # TODO: deal with the transpose convention here

    # TODO: multiple channels

    def __init__(self, shape=(128,128), bounds=[(-1.0,1.0),(-1.0,1.0)], channels=[0.], stokes=['I'], mask=None):

        self.nx, self.ny = shape[0], shape[1]
        self.num_channels = 1
        self.num_stokes = 1
        self.bounds = np.array(bounds, dtype=np.float64, order='C').flatten()
        self.xmin, self.xmax = self.bounds[0], self.bounds[1]
        self.ymin, self.ymax = self.bounds[2], self.bounds[3]
        self.dx = (self.xmax-self.xmin)/self.nx
        self.dy = (self.ymax-self.ymin)/self.ny
        self.channels = np.array(channels, dtype=np.float64, order='C').flatten() 
        self._c_channels = self.channels.ctypes.data_as(c_double_p)

        # load mask if there is one
        if mask is not None:
            with fits.open(mask) as f:
                self.mask = f['PRIMARY'].data[:,:].T.astype(np.bool_, order='C')
        else:
            # TODO: Don't allocate a mask if we don't need it...
            self.mask = np.ones(shape, dtype=np.bool_)
        self._c_mask = self.mask.ctypes.data_as(c_bool_p)
        
        super().__init__(shape=(self.num_channels, self.num_stokes, self.nx, self.ny), dtype=np.float64)


    def plot(self, vec, ax=None):

        if ax is None:
            ax = plt
        ax.imshow(vec._data, origin='lower', extent=self._bounds.flatten(), interpolation='nearest')
        ax.show()


    @property
    def points(self):
        # TODO: use pixel centers?
        centers = np.zeros(self.shape[-2:]+(2,))
        xc = self.xmin + self.dx*np.arange(self.nx) 
        yc = self.ymin + self.dy*np.arange(self.ny) 
        centers[:,:,0] = xc[:,None]
        centers[:,:,1] = yc[None,:] 
        return centers 


class DelaunaySpace(Space, delaunay_space_ctype):

    def __init__(self, points, channels=[0.], stokes=['I']):

        self.num_channels = 1
        self.num_stokes = 1
        self.delaunay = Delaunay(points)
        self.edge = np.unique(self.delaunay.convex_hull.flatten())
        self.vertices = self.delaunay.points
        self._c_points = self.points.ctypes.data_as(c_double_p)
        self.num_points = self.points.shape[0]
        self.triangles = self.delaunay.simplices
        self._c_triangles = self.triangles.ctypes.data_as(c_int_p)
        self.num_triangles = self.triangles.shape[0]

        super().__init__(shape=(self.num_channels, self.num_stokes, self.num_points), dtype=np.float64)

    @property
    def points(self):
        return self.vertices





