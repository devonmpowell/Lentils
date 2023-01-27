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

# TODO: make Lensmodel *inherit* _c_lensmodel?
from lentils.backend import libdeflect, c_lensmodel, c_null_p



# TODO: this needs to be split into more general classes
class LensModel:

    def __init__(self, name='', description='', b=0.463544, th=-14.278754, f=0.799362, x=-0.046847, 
            y=-0.105357, rc=0.000571, qh=0.506730, ss=-0.046500, sa=7.921300, z=0.881000):

        self.name = name
        self.description = description
        self._c_lensmodel = c_lensmodel()
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
        print("sigma_c =", sigma_c)
        print("d_l =", d_l)
        self._c_lensmodel.sigma_c = sigma_c.value 


        # TODO: update c lensmodel here?
        npoints = points.reshape((-1,2)).shape[0] 
        deflected = np.zeros_like(points)

        if deriv:
            gradients = np.zeros((8,)+points.shape)
            libdeflect.deflect_points(self._c_lensmodel, points, npoints, deflected, 1, gradients)
            return deflected, gradients

        libdeflect.deflect_points(self._c_lensmodel, points, npoints, deflected, 0, c_null_p);
        return deflected




