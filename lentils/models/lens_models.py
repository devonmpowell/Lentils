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

    #def __getattr__(self, name):
        #if name in self._exposed_cpars:
            #return self._cpars[0][name]
        #else:
            #return super().__getattr__(name)

    #def __setattr__(self, name, val):
        #if name in self._exposed_cpars:
            #self._cpars[name] = val
            #if name in self._special_setters:
                #self._special_setters[name](val)
        #else:
            #super().__setattr__(name, val)



class PowerLawEllipsoid(MassModel):

    _my_ctype = 0

    def __init__(self, b=0.463544, th=-14.278754, f=0.799362, x=-0.046847, 
            y=-0.105357, rc=0.000571, qh=0.506730, z=0.881000):

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

    def __init__(self,  x=-0.046847, y=-0.105357, ss=-0.046500, sa=7.921300, z=0.881000):

        self.z, self.x, self.y, self.ss, self.sa = z, x, y, ss, sa

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
        return cpars


