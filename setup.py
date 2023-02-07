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

    deflect = Extension('backend.deflect', sources = ['lentils/backend/deflect.c'], include_dirs = ['lentils/backend'])
    triangles = Extension('backend.triangles', sources = ['lentils/backend/triangles.c'], include_dirs = ['lentils/backend'])
    raster = Extension('backend.raster', sources = ['lentils/backend/raster.c','lentils/backend/deflect.c'], 
            include_dirs = ['lentils/backend'],extra_compile_args=['-Wall'])
    nufft = Extension('backend.nufft', sources = ['lentils/backend/nufft.c'],
            libraries=['gsl', 'gslcblas', 'm', 'gomp', 'fftw3', 'fftw3_omp','fftw3_threads'],
            runtime_library_dirs=['/usr/lib/x86_64-linux-gnu'],
            include_dirs=['/usr/include', 'lentils/backend'],
            library_dirs=['/usr/lib/x86_64-linux-gnu'],extra_compile_args=['-fopenmp'])


    setup(use_pyscaffold=True, ext_modules = [deflect, triangles, nufft, raster])
