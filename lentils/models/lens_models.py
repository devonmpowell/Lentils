# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import astropy.io.fits as fits 
import astropy.constants as const 
from ctypes import c_int, c_double, POINTER, Structure, cdll
import ctypes as cc
import glob
import os

import astropy.cosmology as cosmo
_cosmology = cosmo.Planck15

# TODO: make a dedicated sub-module for loading the C backend
c_int_p = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
c_double_p = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
c_null_dummy = np.zeros(0)

class _c_lensmodel(Structure):
    _fields_ = [('b', c_double), ('th', c_double), ('f', c_double), 
           ('x', c_double), ('y', c_double), ('rc', c_double), 
           ('qh', c_double), ('ss', c_double), ('sa', c_double), 
           ('z', c_double), ('d_l', c_double), ('d_s', c_double), 
           ('d_ls', c_double), ('sigma_c', c_double),
           ('sin_th', c_double), ('cos_th', c_double), ('sin_sa', c_double), ('cos_sa', c_double),]

modulepath = os.path.dirname(os.path.realpath(__file__))
libname = glob.glob('{}/deflect*.so'.format(modulepath))[0]
libdeflect = cdll.LoadLibrary(libname)
libdeflect.deflect_points.restype = None
libdeflect.deflect_points.argtypes = [_c_lensmodel, c_double_p,c_int,c_double_p,c_int,c_double_p]


# TODO: this needs to be split into more general classes
class LensModel:

    #def __init__(self, name='', description='', b=0.46, th=-16.0, f=0.80, x=-0.04, 
            #y=-0.13, rc=1.0e-5, qh=0.51, ss=-0.05, sa=9.0, z=0.88):
    def __init__(self, name='', description='', b=0.461902, th=-15.794212, f=0.804914, x=-0.038396, 
            y=-0.130468, rc=0.000571, qh=0.506730, ss=-0.047924, sa=9.135239, z=0.881000):

        self.name = name
        self.description = description
        self._c_lensmodel = _c_lensmodel()
        self._c_lensmodel.b  = b; 
        self._c_lensmodel.th = th;
        self._c_lensmodel.f  = f;
        self._c_lensmodel.x  = x; 
        self._c_lensmodel.y  = y; 
        self._c_lensmodel.rc = rc;
        self._c_lensmodel.qh = qh;
        self._c_lensmodel.ss = ss;
        self._c_lensmodel.sa = sa;
        self._c_lensmodel.z  = z;



    def deflect(self, points, z_s, deriv=False):

        z_l = self._c_lensmodel.z
        d_l = _cosmology.angular_diameter_distance(z_l)
        d_s = _cosmology.angular_diameter_distance(z_s)
        d_ls = _cosmology.angular_diameter_distance_z1z2(z_l, z_s)
        sigma_c = ((const.c*const.c*d_s)/(4*np.pi*const.G*d_ls*d_l))
        self._c_lensmodel.d_l = d_l.value; 
        self._c_lensmodel.d_s = d_s.value; 
        self._c_lensmodel.d_ls = d_ls.value; 
        #sigma_c = (sigma_c*(d_l*d_l)).to('10^10 M_sun / arcsec^2')
        sigma_c = (sigma_c).to('10^10 M_sun / kpc^2')
        #print("sigma_c =", sigma_c)
        self._c_lensmodel.sigma_c = sigma_c.value 


        # TODO: update c lensmodel here?
        npoints = points.reshape((-1,2)).shape[0] 
        deflected = np.zeros_like(points)

        if deriv:
            gradients = np.zeros((8,)+points.shape)
            libdeflect.deflect_points(self._c_lensmodel, points, npoints, deflected, 1, gradients)
            return deflected, gradients

        libdeflect.deflect_points(self._c_lensmodel, points, npoints, deflected, 0, c_null_dummy);
        return deflected




