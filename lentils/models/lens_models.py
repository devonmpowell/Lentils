# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import astropy.cosmology as cosmo
import astropy.units as units
import astropy.constants as const 
from lentils.backend import libdeflect, global_lens_model_ctype, generic_mass_model_ctype, c_void_p, c_null_p


class GlobalLensModel(global_lens_model_ctype):

    def __init__(self, components=None, cosmology=cosmo.Planck15):

        self.cosmology = cosmology
        self.components = []
        if components is not None:
            for comp in components:
                self.add_component(comp)


    def add_component(self, component):
        self.components.append(component)

    def setup_raytracing(self, z_s):

        # Collect all mass components
        # sort by z_l, filter to keep only z_l < z_s
        self.lenses = np.zeros(len(self.components), dtype=generic_mass_model_ctype)
        for i, comp in enumerate(self.components):
            self.lenses[i] = comp.get_cpars()
        self.lenses.sort(order='z_l')
        self.lenses = self.lenses[self.lenses['z_l'] < z_s]
        ncomp = len(self.lenses)

        # Compute angular diameter distances
        # assign c pars in more natural units for lensing 
        d_s = self.cosmology.angular_diameter_distance(z_s)
        d_l = self.cosmology.angular_diameter_distance(self.lenses['z_l'])
        d_ls = self.cosmology.angular_diameter_distance_z1z2(self.lenses['z_l'], z_s)
        sigma_c = ((const.c*const.c*d_s)/(4*np.pi*const.G*d_l*d_ls))
        self.lenses['z_s'] = z_s
        self.lenses['d_s'] = (d_s/units.radian).to('kpc/arcsec')
        self.lenses['d_l'] = (d_l/units.radian).to('kpc/arcsec')
        self.lenses['d_ls'] = (d_ls/units.radian).to('kpc/arcsec')
        self.lenses['sigma_c'] = (sigma_c*(d_l/units.radian)**2).to('10^10 M_sun / arcsec^2')

        # TODO: In the reconst code,
        # Lens 0 redshift = 0.350000, sigma_crit = 5.21746e+00 (10^10 M_sun arcsec^-2) 

        # TODO: compute betas 
        #for i, comp in enumerate(self.components):
            #pass
        #print(self.lenses)
        #print("mass model size =", generic_mass_model_ctype.itemsize)

        self.num_lenses = ncomp
        self._c_lenses = self.lenses.ctypes.data_as(c_void_p)



    def deflect(self, points, z_s, deriv=False):

        self.setup_raytracing(z_s)

        # call deflect backend
        npoints = points.reshape((-1,2)).shape[0] 
        deflected = np.zeros_like(points)
        if deriv:
            gradients = np.zeros((8,)+points.shape)
            libdeflect.deflect_points(self, points, npoints, deflected, 1, gradients)
            return deflected, gradients
        else:
            libdeflect.deflect_points(self, points, npoints, deflected, 0, c_null_p);
            return deflected


class MassModel:

    def __init__(self):
        pass

    def get_cpars(self):
        raise NotImplementedError

    def analytic_convergence(self, points):
        raise NotImplementedError


class PowerLawEllipsoid(MassModel):

    _my_ctype = 0

    def __init__(self, z=0.881000, b=0.463544, th=-14.278754, f=0.799362, x=-0.046847, 
            y=-0.105357, rc=0.000571, qh=0.506730):

        # set initial values
        self.z, self.x, self.y, self.f, self.th, self.b, self.qh, self.rc = z, x, y, f, th, b, qh, rc

    def get_cpars(self):
        '''Return a structured numpy array containing the C-formatted fields'''
        cpars = np.zeros(1, dtype=generic_mass_model_ctype)[0]
        cpars['type'] = self._my_ctype
        cpars['z_l'] = self.z
        cpars['fpars'][0] = self.x
        cpars['fpars'][1] = self.y
        cpars['fpars'][2] = self.f
        cpars['fpars'][3] = self.th
        cpars['fpars'][4] = np.sin((self.th+90)*np.pi/180)
        cpars['fpars'][5] = np.cos((self.th+90)*np.pi/180)
        cpars['fpars'][6] = self.b
        cpars['fpars'][7] = self.qh
        cpars['fpars'][8] = self.rc
        return cpars


class ExternalPotential(MassModel):

    _my_ctype = 1

    def __init__(self, z=0.881000,  x=-0.046847, y=-0.105357, ss=-0.046500, sa=7.921300,
            gks=0.0, gka=0.0, gss=0.0, gsa=0.0):

        self.z, self.x, self.y, self.ss, self.sa = z, x, y, ss, sa
        self.gks, self.gka, self.gss, self.gsa = gks, gka, gss, gsa

    def get_cpars(self):
        '''Return a structured numpy array containing the C-formatted fields'''
        cpars = np.zeros(1, dtype=generic_mass_model_ctype)[0]
        cpars['type'] = self._my_ctype
        cpars['z_l'] = self.z
        cpars['fpars'][0] = self.x
        cpars['fpars'][1] = self.y
        cpars['fpars'][2] = self.ss
        cpars['fpars'][3] = self.sa
        cpars['fpars'][4] = np.sin((self.sa+90)*np.pi/180)
        cpars['fpars'][5] = np.cos((self.sa+90)*np.pi/180)
        cpars['fpars'][6] = self.gks
        cpars['fpars'][7] = self.gka
        cpars['fpars'][8] = np.sin((self.gka+90)*np.pi/180)
        cpars['fpars'][9] = np.cos((self.gka+90)*np.pi/180)
        cpars['fpars'][10] = self.gss
        cpars['fpars'][11] = self.gsa
        cpars['fpars'][12] = np.sin((self.gsa+90)*np.pi/180)
        cpars['fpars'][13] = np.cos((self.gsa+90)*np.pi/180)
        return cpars


class InternalMultipoles(MassModel):

    _my_ctype = 2

    def __init__(self, z=0.881000,  x=-0.046847, y=-0.105357, qh=0.5,
            order=4, coefficients=[[0.0,0.0],[0.0,0.0]]):

        self.z, self.x, self.y, self.qh, self.order = z, x, y, qh, int(order)
        self.coefficients = np.array(coefficients, dtype=np.float64).reshape((order-2,2))

    def get_cpars(self):
        '''Return a structured numpy array containing the C-formatted fields'''
        cpars = np.zeros(1, dtype=generic_mass_model_ctype)[0]
        cpars['type'] = self._my_ctype
        cpars['z_l'] = self.z
        cpars['fpars'][0] = self.x
        cpars['fpars'][1] = self.y
        cpars['fpars'][2] = self.qh
        for i, c in enumerate(self.coefficients.flatten()):
            cpars['fpars'][3+i] = c
        cpars['ipars'][0] = self.order
        return cpars



