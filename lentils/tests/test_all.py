import unittest
from unittest import TestCase, main

from .test_utils import testpath, errtol, max_relative_error 


from lentils.common import * 
from lentils.operators import * 
from lentils.models import * 

from astropy.table import Table
import matplotlib.pyplot as plt

import scipy.sparse as sparse
import scipy.sparse.linalg as linalg 

import copy
from timeit import default_timer as timer


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









class SolverTests(TestCase):

    def test_cg_radio(self):

        #image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.15, 0.35,), (-0.65, 0.85)], 
                #mask=f'{testpath}/data_radio_2d/input/mask_1024_zoom.fits')
        #uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits', image_space=image_space)
        #lensmodel = LensModel(b=0.402005, th=48.987322, f=0.796415, x=-0.445178, y=0.178450, 
                #rc=1.0e-4, qh=0.504365, ss=0.070171, sa=75.522635, z=0.35)


        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.0, 0.2,), (-0.5, 0.7)], 
                mask=f'{testpath}/data_radio_2d/new_mock/mask_connected_thresh5_pad3.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/new_mock/j0751_lwmp_nopos_nosub_0.000000_tiny.uvfits', image_space=image_space)
        lensmodel = LensModel(b=0.462437, th=19.162881, f=0.899243, x=-0.441519, y=0.175067, 
                rc=0.0, qh=0.449087, ss=0.092515, sa=73.698888, z=0.35)


        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=3.2, ncasted=6)
        src_space = lensop.space_right
        points = src_space.points

        # source prior
        lams = 5.0e10
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

        # Run CG solve 
        lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
        pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
        sol, info = linalg.cg(lhs_op, rhs, tol=1.0e-10, atol=1.0e-14, M=pc)

        resid_max = np.max(np.abs(lhs*sol-rhs))/np.max(np.abs(rhs))
        self.assertLess(resid_max, errtol) 
        print('CG resid = ', resid_max) 

        # check against DFT computation
        dft = DFTOperator(uvdata.space, image_space)
        rhs = lensop.T * dft.T * uvdata.covariance_operator * uvdata.data 

        # TODO: concatenating explicit mats automatically
        response = dft._mat * lensop._mat
        lhs = response.T * uvdata.covariance_operator._mat * response + reg_op._mat

        lu = linalg.splu(lhs)
        print("Logdet (PC) =", logdet(lu))
        sol_dft = lu.solve(rhs)

        plt.tripcolor(points[:,0],points[:,1], src_space.tris, sol_dft, shading='gouraud')
        print("Solution max =", np.max(sol_dft))
        plt.title('Solution (direct)')
        plt.show()

        resid = sol - sol_dft 
        resid_max = np.max(np.abs(resid))/np.max(np.abs(sol_dft))
        self.assertLess(resid_max, errtol) 
        print('NUFFT vs DFT resid = ', resid_max) 

        plt.tripcolor(points[:,0],points[:,1], src_space.tris, sol, shading='gouraud')
        print("Solution max =", np.max(sol))
        plt.title('Solution')
        plt.show()



    def test_direct_radio(self):

        return

        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.15, 0.35,), (-0.65, 0.85)])
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits',
                image_space=image_space, image_mask=f'{testpath}/data_radio_2d/new_mock/mask_connected_thresh5_pad3.fits')

        # MAP lens model for J0751 PL only 
        lensmodel = LensModel(b=0.462437, th=19.162881, f=0.899243, x=-0.441519, y=0.175067, 
                rc=0.0, qh=0.449087, ss=0.092515, sa=73.698888, z=0.35)



        #image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.0, 0.2,), (-0.5, 0.7)],
                #mask=f'{testpath}/data_radio_2d/new_mock/mask_connected_thresh5_pad3.fits')
        #uvdata = RadioDataset(f'{testpath}/data_radio_2d/new_mock/j0751_lwmp_nopos_nosub_0.000000_tiny.uvfits',
                #image_space=image_space) #, image_mask=f'{testpath}/data_radio_2d/new_mock/mask_connected_thresh5_pad3.fits')

        # Ground-truth lens model from mock data
        #lensmodel = LensModel(b=0.462437, th=19.162881, f=0.899243, x=-0.441519, y=0.175067, 
                #rc=0.0, qh=0.449087, ss=0.092515, sa=73.698888, z=0.35)
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=3.2, ncasted=3, mask=uvdata.image_mask)
        src_space = lensop.space_right
        points = src_space.points

        # source prior
        lams = 5.0e10
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
        lu = linalg.splu(reg_op._mat + lensop._mat.T @ dft._mat.T @ covop._mat @ dft._mat @ lensop._mat)
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
                noise=0.0304896, mask=f'{testpath}/data_optical_2d/input/mask.fits', 
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

        # source prior
        lams = 5.0e2
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)

        # Set up operators
        response = psfop * lensop
        lhs = response.T * covop * response + reg_op
        rhs = response.T * covop * imdata.data 

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
