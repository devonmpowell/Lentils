import unittest
from unittest import TestCase, main
import numpy as np
import sys


from lentils.common import * 
from lentils.operators import * 
from lentils.models import * 

from astropy.table import Table
import matplotlib.pyplot as plt

import scipy.sparse as sparse
import scipy.sparse.linalg as linalg 

import pkgutil
import copy
from timeit import default_timer as timer

# set tolerances for floating-point checks
errtol = 1.0e-6 

imargs = {'origin': 'lower', 'interpolation': 'nearest', 'cmap': plt.cm.Spectral}

class LensTests(TestCase):


    def test_delaunay_lens_operator(self):

        return

        # load image data just for mask and noise
        testdir = 'tests/optical_2d_quick/input'
        imdata = Dataset.image_from_fits('{}/data.fits'.format(testdir), 
                noise=0.0304896, maskfits='{}/mask.fits'.format(testdir), bounds=[(-0.72,0.72),(-0.67,0.67)])
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
        image_space = ImageSpace(shape=(256,256), bounds=[(-0.5,0.7),(-0.7,0.5)])
        nufft = NUFFTOperator(uv_space, image_space)

        # put some fake data through the pipeline
        vdata = np.random.normal(size=(uv_space.shape[0],2)).view(np.complex128)
        apvec = nufft.T * vdata[:,0]  # TODO: take care of array axes!
        dft = DFTOperator(uv_space, image_space)
        dftvec = dft.T * vdata[:,0]  # TODO: take care of array axes!
        dftvec = np.real(dftvec) # TODO: make real/complex types smarter
        resid_max = np.max(np.abs(dftvec-apvec))/np.max(np.abs(dftvec))
        self.assertLess(resid_max, errtol) 


    def test_dirty_beam(self):

        # test the dirty beam of a mock dataset against the DFT computation, and a reference file
        uvdata = Dataset.visibilities_from_uvfits('tests/radio_2d_fft_quick/input/j0751_small_snr1x.uvfits')
        image_space = ImageSpace(shape=(64,64), bounds=[(-.035, .035), (-.035,.035)])
        nufft = NUFFTOperator(uvdata.space, image_space)
        vdata = np.ones(uvdata.space.shape, dtype=np.complex128)
        cvec = uvdata.covariance_operator.T * vdata
        wtsum = np.sum(cvec)
        apvec = nufft.T * cvec
        self.assertAlmostEqual(1.0, np.max(apvec)/wtsum, delta=errtol) 

        dft = DFTOperator(uvdata.space, image_space)
        dftvec = dft.T * cvec
        dftvec = np.real(dftvec) # TODO: make real/complex types smarter
        resid_max = np.max(np.abs(dftvec-apvec))/np.max(np.abs(dftvec))
        self.assertLess(resid_max, errtol) 

        # TODO: check against reference beam, but need proper pixel sizes and ranges
        #with fits.open('tests/radio_2d_fft_quick/input/dirty_beam_test.fits') as f:
            #reference = f['PRIMARY'].data.T[:,:,0]
        #print("reference shape=", reference.shape)
        #print('reference min,max =', np.min(reference), np.max(reference))
        #plt.imshow(reference.T, extent=[-1.5,1.5,-1.5,1.5], **imargs)
        #plt.show()



    def test_dirty_image(self):

        uvdata = Dataset.visibilities_from_uvfits('tests/radio_2d_fft_quick/input/j0751_small_snr1x.uvfits')
        image_space = ImageSpace(shape=(2048,2048), bounds=[(-1.0, 0.2,), (-0.5, 0.7)])
        nufft = NUFFTOperator(uvdata.space, image_space)
        cvec = uvdata.covariance_operator.T * uvdata.data[0,0,:] # TODO: take care of broadcastability 
        apvec = nufft.T * cvec

        # TODO: compare with a reference dataset here!
        plt.imshow(apvec.T, extent=image_space._bounds.flatten(), **imargs)
        plt.show()



class SolverTests(TestCase):

    def test_cg(self):

        #return True

        # load image data just for mask and noise
        testdir = 'tests/optical_2d_quick/input'
        imdata = Dataset.image_from_fits(f'{testdir}/data.fits', 
                noise=0.0304896, maskfits=f'{testdir}/mask.fits', 
                bounds=[(-0.72,0.72),(-0.67,0.67)])
        image_space = imdata.space 
        covop = imdata.covariance_operator
        #print("data max =", np.max(imdata.data))
        #print("data shape =", imdata.data.shape)
        #plt.imshow((imdata.data*imdata.mask).T, extent=image_space._bounds.flatten(), **imargs)
        #plt.show()


        # make a lens model
        lensmodel = LensModel() # default optical test for now
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=2.059, ncasted=3, mask=imdata.mask)
        src_space = lensop.space_right
        points = src_space.points


        # a mock source
        testsrc = np.zeros(src_space.shape) 
        cr = np.array([-0.03,0.0])
        testsrc += 2.0*np.exp(-1.0/(2.0*0.05**2)*np.sum((points-cr)**2, axis=-1))
        cr = np.array([0.03,-0.05])
        testsrc += 2.0*np.exp(-1.0/(2.0*0.02**2)*np.sum((points-cr)**2, axis=-1))
        #plt.tripcolor(points[:,0],points[:,1], src_space.tris, testsrc, shading='gouraud')
        #plt.title('Mock source')
        #plt.show()

        # apply operators and noise
        lensed = lensop * testsrc
        #print('lensed max =', np.max(np.abs(lensed)))
        #plt.imshow(lensed.T, extent=image_space._bounds.flatten(), **imargs)
        #plt.show()


        psfop = ConvolutionOperator(image_space, fitsfile=f'{testdir}/psf.fits', kernelsize=21, fft=False)
        blurred = psfop * lensed
        mockdata = blurred + np.random.normal(scale=imdata.sigma) 
        #plt.imshow(mockdata.T, extent=image_space._bounds.flatten(), **imargs)
        #plt.show()

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
