# -*- coding: utf-8 -*-

# This file imports the libraries and sets up ctypes function signatures for all C backends
# Do not mess with it unless you know what you are doing!

from ctypes import c_int, c_double, c_char, POINTER, Structure
from numpy import ctypeslib
import numpy as np
from os.path import dirname, realpath

# path to this backend directory
modulepath = dirname(realpath(__file__))

# array types
c_int_p = ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
c_double_p = ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
c_complex_p = ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS')
c_bool_p = ctypeslib.ndpointer(dtype=np.bool, flags='C_CONTIGUOUS')
c_null_p = np.zeros(0) # a hack, c_double_p will not accept 'None' as 'NULL'


############################################################
# deflect.c: lens model deflection angles
############################################################

class c_lensmodel(Structure):
    _fields_ = [('b', c_double), ('th', c_double), ('f', c_double), 
           ('x', c_double), ('y', c_double), ('rc', c_double), 
           ('qh', c_double), ('ss', c_double), ('sa', c_double), 
           ('z', c_double), ('d_l', c_double), ('d_s', c_double), 
           ('d_ls', c_double), ('sigma_c', c_double),
           ('sin_th', c_double), ('cos_th', c_double), ('sin_sa', c_double), ('cos_sa', c_double),]

libdeflect = ctypeslib.load_library('deflect', modulepath)
libdeflect.deflect_points.restype = None
libdeflect.deflect_points.argtypes = [c_lensmodel, c_double_p,c_int,c_double_p,c_int,c_double_p]


############################################################
# triangles.c: triangle geometry 
############################################################

libtriangles = ctypeslib.load_library('triangles', modulepath)
libtriangles.triangle_gradient_csr.restype = None
libtriangles.triangle_gradient_csr.argtypes = [c_int, c_int_p, c_double_p, c_int_p, c_int_p, c_double_p]
libtriangles.delaunay_lens_matrix_csr.restype = None
libtriangles.delaunay_lens_matrix_csr.argtypes = [c_int, c_int, c_int, c_bool_p, 
        c_double_p, c_int_p, c_double_p, c_int_p, c_int_p, c_int_p, c_double_p]


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
            ('uv', c_double_p), ('channels', c_double_p),
            ('wsup', c_double), ('kb_beta', c_double),]

libnufft = ctypeslib.load_library('nufft', modulepath)
libnufft.init_nufft.restype = None
libnufft.init_nufft.argtypes = [POINTER(c_nufft),
        c_int, c_int, c_double, c_double, c_double, c_double,
        c_double, c_int, c_int, c_int, c_int, c_double_p, c_double_p]
libnufft.evaluate_apodization_correction.restype = None
libnufft.evaluate_apodization_correction.argtypes = [POINTER(c_nufft), c_double_p]
libnufft.zero_pad.restype = None
libnufft.zero_pad.argtypes = [POINTER(c_nufft), c_double_p, c_double_p, c_int]
libnufft.grid_cpu.restype = None
libnufft.grid_cpu.argtypes = [POINTER(c_nufft), c_complex_p, c_complex_p, c_int]
libnufft.dft_matrix_csr.restype = None
libnufft.dft_matrix_csr.argtypes = [POINTER(c_nufft), c_bool_p, c_int_p, c_int_p, c_double_p]
libnufft.convolution_matrix_csr.restype = None
libnufft.convolution_matrix_csr.argtypes = [c_int, c_int, c_int, c_int, 
        c_double_p, c_int_p, c_int_p, c_double_p]


############################################################
# raster.c: computing exact pixel-triangle intersections
############################################################

libraster = ctypeslib.load_library('raster', modulepath)
libraster.rasterize_triangle.restype = None
libraster.rasterize_triangle.argtypes = [c_double_p, c_double_p, c_int, c_int, c_double, c_double, c_double, c_double, c_double_p]




