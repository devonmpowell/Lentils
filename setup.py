# -*- coding: utf-8 -*-
"""
    Setup file for lentils.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import VersionConflict, require
from setuptools import setup, Extension

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)

# TODO: parse gpu and omp options!

if __name__ == "__main__":

    deflect = Extension('models.deflect_backend', sources = ['lentils/models/deflect_backend.c'], include_dirs = ['lentils/common', 'lentils/models'])
    triangles = Extension('operators.triangles_backend', sources = ['lentils/operators/triangles_backend.c'], include_dirs = ['lentils/common', 'lentils/operators'])
    raster = Extension('operators.raster_backend', sources = ['lentils/operators/raster_backend.c'], include_dirs = ['lentils/common', 'lentils/operators'])
    nufft = Extension('operators.nufft_backend', sources = ['lentils/operators/nufft_backend.c'],
            libraries=['gsl', 'gslcblas', 'm', 'gomp', 'fftw3', 'fftw3_omp','fftw3_threads'],
            runtime_library_dirs=['/usr/lib/x86_64-linux-gnu'],
            include_dirs=['/usr/include', 'lentils/common', 'lentils/operators'],
            library_dirs=['/usr/lib/x86_64-linux-gnu'],extra_compile_args=['-fopenmp'])


    setup(use_pyscaffold=True, ext_modules = [deflect, triangles, nufft, raster])
