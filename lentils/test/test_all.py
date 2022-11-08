import unittest
from unittest import TestCase, main
import numpy as np
import sys


from lentils.common import * 
from lentils.operators import * 
from lentils.models import * 

from astropy.table import Table
import astropy.io.fits as fits 
import matplotlib.pyplot as plt

import scipy.sparse as sparse
import scipy.sparse.linalg as linalg 

import pkgutil
import copy
from timeit import default_timer as timer

# set tolerances for floating-point checks
errtol = 1.0e-6 

imargs = {'origin': 'lower', 'interpolation': 'nearest', 'cmap': plt.cm.Spectral}

from os.path import dirname, realpath

# path to this test directory
testpath = dirname(realpath(__file__))


class LensTests(TestCase):


    def test_delaunay_lens_operator(self):

        return

        # load image data just for mask and noise
        imdata = Dataset.image_from_fits(f'{testpath}/data_optical_2d/data.fits',
                noise=0.0304896, maskfits=f'{testpath}/data_optical_2d/mask.fits', 
                bounds=[(-0.72,0.72),(-0.67,0.67)])
        image_space = imdata.space 
        print("data max =", np.max(imdata.data))
        print("data shape =", imdata.data.shape)
        plt.imshow(imdata.data.T, extent=image_space._bounds.flatten(), **imargs)
        plt.show()

        # make a lens model
        lensmodel = LensModel() # default optical test for now
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=2.059, ncasted=1, mask=imdata.mask)
        src_space = lensop.space_right

        # TODO: replace this bogus test source with reference data from the old code
        points = src_space.points
        testsrc = np.exp(-1.0/(2.0*0.02**2)*np.sum((points)**2, axis=-1))
        cr = np.array([0.03,0.04])
        testsrc += 2.0*np.exp(-1.0/(2.0*0.01**2)*np.sum((points-cr)**2, axis=-1))
        plt.tripcolor(points[:,0],points[:,1], src_space.tris, testsrc, shading='gouraud')
        #plt.set_xlims(*image_space._bounds[0])
        #plt.set_ylims(*image_space._bounds[1])
        plt.show()

        # apply operators and noise
        lensed = lensop.apply(testsrc)
        plt.imshow(lensed.T, extent=image_space._bounds.flatten(), **imargs)
        plt.show()

        psfop = ConvolutionOperator(image_space, fitsfile='{}/psf.fits'.format(testdir), kernelsize=21)
        blurred = psfop.apply(lensed)
        plt.imshow(blurred.T, extent=image_space._bounds.flatten(), **imargs)
        plt.show()

        noised = blurred + np.random.normal(scale=imdata.sigma) 
        plt.imshow(noised.T, extent=image_space._bounds.flatten(), **imargs)
        plt.show()

    def test_gradients(self):

        return

        # define the image plane
        image_space = ImageSpace(shape=(31,31)) 
        points = image_space.points
        xc = points[:,:,0] + 0.5*image_space._dx[0]
        yc = points[:,:,1] + 0.5*image_space._dx[1]
        bounds = image_space._bounds.flatten()

        # make a lens model
        zsrc = 2.0
        pars0 = {'b': 0.4, 'qh': 0.45, 'f': 0.9, 'x': -0.01, 'y': 0.05, 'th': -20.0, 'ss': 0.5, 'sa': 103.0, 'z': 0.2, 'rc': 0.0}
        idx = {'b': 0, 'qh': 1, 'f': 2, 'x': 3, 'y': 4 , 'th': 5, 'ss': 6, 'sa': 7}#, 'z': -1, 'rc': -1}
        lensmodel = LensModel(**pars0)

        # deflection and analytic gradient
        start = timer()
        deflected, gradient = lensmodel.deflect(points, z_s=zsrc, deriv=True)
        end = timer()
        print("Elapsed time, with gradients = %f ms" % (1000*(end-start)))
        #alpha0 = deflected-points
        #plt.imshow((np.sum(alpha0**2,axis=-1)**0.5).T, extent=image_space._bounds.flatten(), **imargs)
        #plt.quiver(points[:,:,0],points[:,:,1], alpha0[:,:,0], alpha0[:,:,1], angles='xy', scale_units='xy', scale=8)
        #plt.show()

        ttot = 0.0
        for par in idx:

            # gradient with finite diff
            dd = 1.0e-10;
            parsh = copy.deepcopy(pars0)
            parsh[par] += dd
            lensmodel = LensModel(**parsh)
            start = timer()
            defh = lensmodel.deflect(points, z_s=zsrc)
            gradh = (deflected-defh)/dd
            end = timer()
            ttot += 1000*(end-start)

            # plot analytic gradient
            grad0 = gradient[idx[par]]
            mag = np.sum(grad0**2,axis=-1)**0.5

            plt.figure()
            im = plt.imshow(mag.T, extent=bounds, **imargs)
            plt.colorbar(im)
            plt.quiver(xc, yc, grad0[:,:,0], grad0[:,:,1])
            plt.title(r'Analytic $\partial \vec{\alpha} / \partial (%s)$'%par)

            # plot finite diff
            #plt.figure()
            #mag = np.sum(gradh**2,axis=-1)**0.5
            #im = plt.imshow(mag.T, extent=bounds, **imargs)
            #plt.colorbar(im)
            #plt.quiver(xc, yc, gradh[:,:,0], gradh[:,:,1]) 
            #plt.title(r'Finite diff $\partial \vec{\alpha} / \partial (%s)$'%par)

            # residuals
            #plt.figure()
            #err = np.sum((gradh-grad0)**2,axis=-1)**0.5 / np.max(grad0)
            #im = plt.imshow(err.T, extent=bounds, **imargs)
            #plt.colorbar(im)
            #plt.title('Magnitude of gradient error')

        plt.show()

        print("Elapsed time, no gradients = %f ms" % ttot)









class NUFFTTests(TestCase):

    def test_gridder_and_apodization(self):

        # make a fake uv coordinate set at different scales in units of du
        uv_per_scale = 4 
        all_uv = []
        for scale in np.linspace(0.0,60.0,20):
            all_uv.append([np.random.normal(scale=1e5*scale,size=(uv_per_scale,3))])
        all_uv = np.array(all_uv, dtype=np.float64).reshape((-1,3))
        uv_space = VisibilitySpace(name='RandomVisTest', channels=[0.5], uvcoords=all_uv)

        # TODO: test for odd-sized and non-square grids!
        image_space = ImageSpace(shape=(256,256), bounds=[(-0.5,0.7),(-0.7,0.5)])

        # put some fake data through the pipeline
        vdata = np.random.normal(size=(uv_space.shape[0],2)).view(np.complex128)
        nufft = NUFFTOperator(uv_space, image_space)
        dft = DFTOperator(uv_space, image_space)
        nufftvec = nufft.T * vdata
        dftvec = dft.T * vdata
        resid_max = np.max(np.abs(dftvec-nufftvec))/np.max(np.abs(dftvec))
        self.assertLess(resid_max, errtol) 

        # Test the forward operation as well
        nufftvis = nufft * dftvec
        dftvis = dft * dftvec
        resid_max = np.max(np.abs(dftvis-nufftvis))/np.max(np.abs(dftvis))
        self.assertLess(resid_max, errtol) 


    def test_dirty_beam(self):

        # set up proper pixel sizes relative to reference fits file
        # We use a truncated space on only the center region of the beam 
        # to save time with the direct FT
        nx = 32 
        cx = 1024
        dx = 1.5/cx
        extent = 0.5*dx*nx
        image_space = ImageSpace(shape=(nx,nx), bounds=[(-extent, extent), (-extent,extent)])

        # load the radio data and compute the dirty image with NUFFT
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits',
                image_space=image_space)
        apvec = uvdata.dirty_beam

        # check that the sum of weights is equal to the beam max
        cvec = uvdata.covariance_operator.T * uvdata.space.new_vector(1.0)
        wtsum = np.sum(np.abs(cvec))
        self.assertAlmostEqual(1.0, np.max(apvec)/wtsum, delta=errtol) 

        # check against DFT computation
        dft = DFTOperator(uvdata.space, uvdata._space_beam)
        dftvec = dft.T * cvec
        resid = (dftvec-apvec)/wtsum
        resid_max = np.max(np.abs(resid))
        self.assertLess(resid_max, errtol) 

        # check against reference data
        with fits.open(f'{testpath}/data_radio_2d/reference/dirty_beam_reference_fft.fits') as f:
            reference = f['PRIMARY'].data.T[cx-nx:cx+nx,cx-nx:cx+nx,0]
        resid = (reference-apvec)/wtsum
        resid_max = np.max(np.abs(resid))
        self.assertLess(resid_max, errtol) 


    def test_dirty_image(self):

        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.15, 0.35,), (-0.65, 0.85)], 
                mask=f'{testpath}/data_radio_2d/input/mask_1024_zoom.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits',
                image_space=image_space)
        dirty_image = uvdata.dirty_image
        refmax = np.max(np.abs(dirty_image))

        # check against DFT computation
        dft = DFTOperator(uvdata.space, image_space)
        dftvec = dft.T * uvdata.covariance_operator.T * uvdata.data
        resid = (dftvec-dirty_image)/refmax
        resid_max = np.max(np.abs(resid*image_space.mask))
        self.assertLess(resid_max, errtol) 

        # compare to a reference FFT dataset 
        with fits.open(f'{testpath}/data_radio_2d/reference/dirty_image_reference_fft.fits') as f:
            reference = f['PRIMARY'].data.T[:,:,0]
        resid = (reference-dirty_image)/refmax
        resid_max = np.max(np.abs(resid))
        self.assertLess(resid_max, errtol) 

        # compare to reference DFT data set
        with fits.open(f'{testpath}/data_radio_2d/reference/dirty_image_reference_dft.fits') as f:
            reference = f['PRIMARY'].data.T[:,:,0]
        resid = (reference-dirty_image)/refmax
        resid_max = np.max(np.abs(resid*image_space.mask))
        self.assertLess(resid_max, errtol) 


class SolverTests(TestCase):

    def test_cg_radio(self):

        return

        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.15, 0.35,), (-0.65, 0.85)])
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits',
                image_space=image_space, image_mask=f'{testpath}/data_radio_2d/input/mask_1024_zoom.fits')

        # MAP lens model for J0751 PL only 
        lensmodel = LensModel(b=0.462437, th=19.162881, f=0.899243, x=-0.441519, y=0.175067, 
                rc=0.0, qh=0.449087, ss=0.092515, sa=73.698888, z=0.35)
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=3.2, ncasted=3, mask=uvdata.image_mask)
        src_space = lensop.space_right
        points = src_space.points

        # source prior
        lams = 5.0e14
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)

        # Set up operators
        lhs = lensop.T * uvdata.blurred_covariance_operator * lensop + reg_op
        rhs = lensop.T * uvdata.dirty_image

        # preconditioner
        # TODO: cheaper setup for weight sum
        wtsum = np.sum(np.abs(uvdata.covariance_operator.T * uvdata.space.new_vector(1.0)))
        lu = linalg.splu(reg_op._mat + wtsum * lensop._mat.T @ lensop._mat)
        def logdet(dcmp):
            return np.sum(np.log(dcmp.U.diagonal().astype(np.complex128))) + np.sum(np.log(dcmp.L.diagonal().astype(np.complex128)))
        print("Logdet (PC) =", logdet(lu))

        # solve
        #i = 0
        def lhs_fun(vec):
            #global i
            #print("Apply LHS", i)

            print(np.min(vec), np.max(vec))
            plt.tripcolor(points[:,0],points[:,1], src_space.tris, vec, shading='gouraud')
            plt.title('Iter')
            plt.show()
            #i += 1
            return lhs.apply(vec)


        lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
        #lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs_fun)
        pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
        x, info = linalg.cg(lhs_op, rhs, tol=1.0e-10, atol=1.0e-14, M=pc)


        plt.tripcolor(points[:,0],points[:,1], src_space.tris, x, shading='gouraud')
        print("Solution max =", np.max(x))
        plt.title('Solution')
        plt.show()


        resid = x - testsrc
        print('Source resid = ', np.sum(resid**2)**0.5 / np.sum(rhs**2)**0.5)
        #plt.tripcolor(points[:,0],points[:,1], src_space.tris, resid, shading='gouraud')
        #plt.title('Mock Resid')
        #plt.show()


        resid = lhs*x-rhs
        print('CG resid = ', np.sum(resid**2)**0.5 / np.sum(rhs**2)**0.5)
        #plt.tripcolor(points[:,0],points[:,1], src_space.tris, resid, shading='gouraud')
        #plt.title('CG Resid')
        #plt.show()

        # Direct solve of the full system
        # TODO: lhs._mat !
        lu = linalg.splu(reg_op._mat + lensop._mat.T @ psfop._mat.T @ covop._mat @ psfop._mat @ lensop._mat)
        print("Logdet (direct) =", logdet(lu))

        # solve
        x = lu.solve(rhs)

        plt.tripcolor(points[:,0],points[:,1], src_space.tris, x, shading='gouraud')
        print("Solution max =", np.max(x))
        plt.title('Solution (direct)')
        plt.show()



    def test_direct_radio(self):

        #return

        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.15, 0.35,), (-0.65, 0.85)], 
                            mask=f'{testpath}/data_radio_2d/input/mask_1024_zoom.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits',
                image_space=image_space, image_mask=f'{testpath}/data_radio_2d/input/mask_1024_zoom.fits')

        # MAP lens model for J0751 PL only 
        lensmodel = LensModel(b=0.462437, th=19.162881, f=0.899243, x=-0.441519, y=0.175067, 
                rc=0.0, qh=0.449087, ss=0.092515, sa=73.698888, z=0.35)
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=3.2, ncasted=3, mask=uvdata.image_mask)
        src_space = lensop.space_right
        points = src_space.points

        # source prior
        lams = 5.0e14
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)

        # check against DFT computation
        dft = DFTOperator(uvdata.space, image_space)
        rhs = lensop.T * dft.T * uvdata.covariance_operator * uvdata.data 

        # TODO: concatenating explicit mats automatically
        response = dft._mat * lensop._mat
        lhs = response.T * uvdata.covariance_operator._mat * response + reg_op._mat

        lu = linalg.splu(lhs)
        def logdet(dcmp):
            return np.sum(np.log(dcmp.U.diagonal().astype(np.complex128))) + np.sum(np.log(dcmp.L.diagonal().astype(np.complex128)))
        print("Logdet (PC) =", logdet(lu))

        x = lu.solve(rhs)


        plt.tripcolor(points[:,0],points[:,1], src_space.tris, x, shading='gouraud')
        print("Solution max =", np.max(x))
        plt.title('Solution')
        plt.show()


        resid = x - testsrc
        print('Source resid = ', np.sum(resid**2)**0.5 / np.sum(rhs**2)**0.5)
        #plt.tripcolor(points[:,0],points[:,1], src_space.tris, resid, shading='gouraud')
        #plt.title('Mock Resid')
        #plt.show()


        resid = lhs*x-rhs
        print('CG resid = ', np.sum(resid**2)**0.5 / np.sum(rhs**2)**0.5)
        #plt.tripcolor(points[:,0],points[:,1], src_space.tris, resid, shading='gouraud')
        #plt.title('CG Resid')
        #plt.show()

        # Direct solve of the full system
        # TODO: lhs._mat !
        lu = linalg.splu(reg_op._mat + lensop._mat.T @ psfop._mat.T @ covop._mat @ psfop._mat @ lensop._mat)
        print("Logdet (direct) =", logdet(lu))

        # solve
        x = lu.solve(rhs)

        plt.tripcolor(points[:,0],points[:,1], src_space.tris, x, shading='gouraud')
        print("Solution max =", np.max(x))
        plt.title('Solution (direct)')
        plt.show()




    def test_cg(self):

        return

        # load image data 
        # it creats the data covariance and psf operators automatically
        imdata = OpticalDataset(f'{testpath}/data_optical_2d/input/data.fits',
                noise=0.0304896, maskfits=f'{testpath}/data_optical_2d/input/mask.fits', 
                psf=f'{testpath}/data_optical_2d/input/psf.fits', psf_support=21,
                bounds=[(-0.72,0.72),(-0.67,0.67)])
        image_space = imdata.space 
        psfop = imdata.blurring_operator
        covop = imdata.covariance_operator
        plt.imshow((imdata.data*imdata.mask).T, extent=image_space._bounds.flatten(), **imargs)
        plt.show()


        # make a lens model
        lensmodel = LensModel() # default optical test for now
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=2.059, ncasted=3, mask=imdata.mask)
        src_space = lensop.space_right
        points = src_space.points


        # make some mock data... 
        testsrc = np.zeros(src_space.shape) 
        cr = np.array([-0.03,0.0])
        testsrc += 2.0*np.exp(-1.0/(2.0*0.05**2)*np.sum((points-cr)**2, axis=-1))
        cr = np.array([0.03,-0.05])
        testsrc += 2.0*np.exp(-1.0/(2.0*0.02**2)*np.sum((points-cr)**2, axis=-1))
        mockdata = psfop * lensop * testsrc + np.random.normal(scale=imdata.sigma) 

        # source prior
        lams = 5.0e2
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)

        # Set up operators
        response = psfop * lensop
        lhs = response.T * covop * response + reg_op
        rhs = response.T * covop * mockdata 

        # preconditioner
        # TODO: cholesky rather than LU?
        # TODO: Incomplete cholesky or LU? 
        lu = linalg.splu(reg_op._mat + lensop._mat.T @ covop._mat @ lensop._mat)
        def logdet(dcmp):
            return np.sum(np.log(dcmp.U.diagonal().astype(np.complex128))) + np.sum(np.log(dcmp.L.diagonal().astype(np.complex128)))
        print("Logdet (PC) =", logdet(lu))

        # solve
        lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
        pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
        x, info = linalg.cg(lhs_op, rhs, tol=1.0e-10, atol=1.0e-14, M=pc)


        plt.tripcolor(points[:,0],points[:,1], src_space.tris, x, shading='gouraud')
        print("Solution max =", np.max(x))
        plt.title('Solution')
        plt.show()


        resid = x - testsrc
        print('Source resid = ', np.sum(resid**2)**0.5 / np.sum(rhs**2)**0.5)
        #plt.tripcolor(points[:,0],points[:,1], src_space.tris, resid, shading='gouraud')
        #plt.title('Mock Resid')
        #plt.show()


        resid = lhs*x-rhs
        print('CG resid = ', np.sum(resid**2)**0.5 / np.sum(rhs**2)**0.5)
        #plt.tripcolor(points[:,0],points[:,1], src_space.tris, resid, shading='gouraud')
        #plt.title('CG Resid')
        #plt.show()

        # Direct solve of the full system
        # TODO: lhs._mat !
        lu = linalg.splu(reg_op._mat + lensop._mat.T @ psfop._mat.T @ covop._mat @ psfop._mat @ lensop._mat)
        print("Logdet (direct) =", logdet(lu))

        # solve
        x = lu.solve(rhs)

        plt.tripcolor(points[:,0],points[:,1], src_space.tris, x, shading='gouraud')
        print("Solution max =", np.max(x))
        plt.title('Solution (direct)')
        plt.show()










####### runs all tests by default #######
if __name__ == '__main__':
    main(verbosity=2)
