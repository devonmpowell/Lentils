# -*- coding: utf-8 -*-

# This file imports the libraries and sets up ctypes function signatures for all C backends
# Do not mess with it unless you know what you are doing!

from ctypes import c_int, c_double, c_char, c_bool, POINTER, Structure
from numpy import ctypeslib
import numpy as np
from os.path import dirname, realpath

# path to this backend directory
modulepath = dirname(realpath(__file__))

# array types
arg_c_int_p = ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
arg_c_double_p = ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
arg_c_complex_p = ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS')
arg_c_bool_p = ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS')
c_null_p = np.zeros(0) # a hack, c_double_p will not accept 'None' as 'NULL'

c_int_p = POINTER(c_int) 
c_double_p = POINTER(c_double)
c_bool_p = POINTER(c_bool)

# struct types for array containers

class image_space_ctype(Structure):
    _fields_ = [('nx', c_int), ('ny', c_int),('num_channels', c_int), ('num_stokes', c_int),
            ('xmin', c_double), ('xmax', c_double), ('ymin', c_double), ('ymax', c_double),
            ('dx', c_double), ('dy', c_double),('_c_channels', c_double_p),
            ('_c_mask', c_bool_p)]

class delaunay_space_ctype(Structure):
    _fields_ = [('num_points', c_int),('num_triangles', c_int),('num_channels', c_int), ('num_stokes', c_int),
            ('_c_channels', c_double_p),('_c_points', c_double_p),('_c_triangles', c_int_p)]

class fourier_space_ctype(Structure):
    _fields_ = [('nu', c_int), ('nv', c_int), ('half_nv', c_int),('num_channels', c_int), ('num_stokes', c_int),
            ('du', c_double), ('dv', c_double),('gcx', c_double), ('gcy', c_double),
            ('_c_channels', c_double_p)]

class visibility_space_ctype(Structure):
    _fields_ = [('num_rows', c_int),('num_channels', c_int), ('num_stokes', c_int),
            ('_c_uvw', c_double_p), ('_c_channels', c_double_p)]

        
############################################################
# deflect.c: lens model deflection angles
############################################################

generic_mass_model_ctype = np.dtype([('type','u8'),('z_l','f8'),('z_s','f8'),('d_l','f8'),('d_s','f8'),('d_ls','f8'),
                                    ('sigma_c','f8'),('beta','f8'),('fpars','f8',32),('flags','b',32)])
c_lensmodel_p = ctypeslib.ndpointer(dtype=generic_mass_model_ctype, flags='C_CONTIGUOUS')

libdeflect = ctypeslib.load_library('deflect', modulepath)
libdeflect.deflect_points.restype = None
libdeflect.deflect_points.argtypes = [c_lensmodel_p, c_int, arg_c_double_p, c_int,arg_c_double_p,c_int,arg_c_double_p]


############################################################
# triangles.c: triangle geometry 
############################################################

libtriangles = ctypeslib.load_library('triangles', modulepath)
libtriangles.triangle_gradient_csr.restype = None
libtriangles.triangle_gradient_csr.argtypes = [delaunay_space_ctype, 
        arg_c_int_p, arg_c_int_p, arg_c_double_p]
libtriangles.delaunay_lens_matrix_csr.restype = None
libtriangles.delaunay_lens_matrix_csr.argtypes = [image_space_ctype, delaunay_space_ctype, 
        c_int, arg_c_double_p, arg_c_int_p, arg_c_int_p, arg_c_int_p, arg_c_double_p]


############################################################
# nufft.c: non-uniform FFT 
############################################################

class c_nufft(Structure):
    _fields_ = [('nx_im', c_int), ('ny_im', c_int),
            ('padx', c_int), ('pady', c_int),
            ('nx_pad', c_int), ('ny_pad', c_int),
            ('half_ny_pad', c_int), ('pad_factor', c_double),
            ('xmin', c_double), ('xmax', c_double), ('ymin', c_double), ('ymax', c_double),
            ('dx', c_double), ('dy', c_double),
            ('du', c_double), ('dv', c_double),
            ('nchannels', c_int),('nstokes', c_int),('nrows', c_int),
            ('uv', arg_c_double_p), ('channels', arg_c_double_p),
            ('wsup', c_double), ('kb_beta', c_double),]

libnufft = ctypeslib.load_library('nufft', modulepath)
libnufft.zero_pad.restype = None
libnufft.zero_pad.argtypes = [image_space_ctype, arg_c_double_p, 
        image_space_ctype, arg_c_double_p, c_int]
#libtriangles.zpad_matrix_csr.restype = None
#libtriangles.zpad_matrix_csr.argtypes = [image_space_ctype, image_space_ctype, 
        #arg_c_int_p, arg_c_int_p, arg_c_double_p]
libnufft.grid_cpu.restype = None
libnufft.grid_cpu.argtypes = [visibility_space_ctype, arg_c_complex_p, 
        fourier_space_ctype, arg_c_complex_p, c_int, c_double, c_int]

libnufft.dft_matrix_csr.restype = None
libnufft.dft_matrix_csr.argtypes = [visibility_space_ctype, image_space_ctype, 
        arg_c_int_p, arg_c_int_p, arg_c_double_p]

libnufft.convolution_matrix_csr.restype = None
libnufft.convolution_matrix_csr.argtypes = [image_space_ctype, c_int, c_int, 
        arg_c_double_p, arg_c_int_p, arg_c_int_p, arg_c_double_p]


############################################################
# raster.c: computing exact pixel-triangle intersections
############################################################

libraster = ctypeslib.load_library('raster', modulepath)
libraster.rasterize_triangle.restype = None
libraster.rasterize_triangle.argtypes = [arg_c_double_p, arg_c_double_p, c_int, c_int, c_double, c_double, c_double, c_double, arg_c_double_p]


#libraster.manifold_lens_matrix_csr.restype = None
#libraster.manifold_lens_matrix_csr.argtypes = [image_space_ctype, image_space_ctype, c_int, 
        #arg_c_int_p, arg_c_int_p, arg_c_double_p]




