import unittest
from unittest import TestCase, main
import numpy as np

from .test_utils import testpath, errtol, max_relative_error
from .test_utils import imargs



from lentils.common import * 
from lentils.operators import DelaunayLensOperator, ManifoldLensOperator, ConvolutionOperator, PriorCovarianceOperator 
from lentils.models import * 

from astropy.table import Table
import matplotlib.pyplot as plt

import scipy.sparse.linalg as linalg 

import copy
from timeit import default_timer as timer


class LensTests(TestCase):



    def test_mock_radio(self):


        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.0, 0.2,), (-0.5, 0.7)], 
                mask=f'{testpath}/data_radio_2d/new_mock/mask_connected_thresh5_pad3.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/new_mock/j0751_lwmp_nopos_nosub_0.000000_tiny.uvfits', image_space=image_space)


        # make a lens model
        lensmodel = GlobalLensModel()
        lensmodel.add_component(PowerLawEllipsoid(b=0.462437, th=19.162881, f=0.899243, x=-0.441519, y=0.175067, 
                rc=0.0, qh=0.449087, z=0.35))
        lensmodel.add_component(ExternalPotential(ss=0.092515, sa=73.698888, z=0.35))


        srdims = (256,256)
        src_space = ImageSpace(shape=srdims, bounds=[-0.510000, -0.330000, 0.020000, 0.200000])
        print(src_space.dx)

        lensop = ManifoldLensOperator(image_space, src_space, lensmodel, z_src=3.2)

        # Set up operators
        lams = 1.0e12 * 0.000703125**2
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)

        lams = 1.0e8
        reg_op_tikh = PriorCovarianceOperator(src_space, type='tikhonov', strength=lams)



        lhs = lensop.T * uvdata.blurred_covariance_operator * lensop + reg_op + reg_op_tikh
        rhs = lensop.T * uvdata.dirty_image

        # preconditioner and CG solve
        # TODO: cheaper setup for weight sum
        wtsum = np.sum(np.abs(uvdata.covariance_operator.T * uvdata.space.new_vector(1.0)))
        lu = linalg.splu(reg_op._mat + reg_op_tikh._mat + wtsum * lensop._mat.T @ lensop._mat)
        #def logdet(dcmp):
            #usum = np.sum(np.log(dcmp.U.diagonal().astype(np.complex128))) 
            #lsum = np.sum(np.log(dcmp.L.diagonal().astype(np.complex128)))
            #print("lsum, usum =", lsum, usum)
            #return lsum+usum 
        #print("Logdet (PC) =", logdet(lu))
        lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
        pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
        sol_cg, info = linalg.cg(lhs_op, rhs.flatten(), tol=1.0e-10, atol=1.0e-14, M=pc)
        err_max = max_relative_error(lhs*sol_cg, rhs)
        self.assertLess(err_max, errtol) 

        sol = sol_cg.reshape(src_space.shape)
        plt.imshow(sol[0,0].T, extent=src_space.bounds, **imargs)
        plt.title("Solution CG")
        plt.show()




    def test_manifold_lens_operator(self):

        # load image data 
        # it creats the data covariance and psf operators automatically
        imdata = OpticalDataset(f'{testpath}/data_optical_2d/input/data.fits',
                noise=0.0304896, mask=f'{testpath}/data_optical_2d/input/mask.fits', 
                psf=f'{testpath}/data_optical_2d/input/psf.fits', psf_support=21,
                bounds=[(-0.72,0.72),(-0.67,0.67)])
        image_space = imdata.space 

        # make a lens model
        lensmodel = GlobalLensModel()
        ple = PowerLawEllipsoid()
        lensmodel.add_component(ple)
        lensmodel.add_component(ExternalPotential())

        
        rpts = image_space.points
        rpts[...,0] -= ple.x
        rpts[...,1] -= ple.y
        rr = np.sum(rpts**2, axis=-1)**0.5
        image_space.mask[:,:] = (rr > 0.15)*(rr < 0.8) 
        #image_space.mask[:,:] = True 

        srdims = (256,256)
        #srdims = (16,16)
        src_space = ImageSpace(shape=srdims, bounds=[-0.3,0.3,-0.3,0.3])


        #'''
        #testsrc = src_space.new_vector()
        #testsrc[0,0,:,:] = np.indices(srdims).sum(axis=0) % 2

        # TODO: replace this bogus test source with reference data from the old code
        points = src_space.points
        testsrc = np.exp(-1.0/(2.0*0.05**2)*np.sum((points)**2, axis=-1))
        cr = np.array([0.10,0.10])
        testsrc += 1.5*np.exp(-1.0/(2.0*0.03**2)*np.sum((points-cr)**2, axis=-1))
        testsrc = testsrc.reshape(src_space.shape)

        plt.imshow(testsrc[0,0,:,:].T, extent=src_space.bounds, **imargs)
        plt.title("Test source")
        plt.show()
        #'''

        #lams = 0.01 
        #reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)
        lams = 1.0e-6 
        reg_op = PriorCovarianceOperator(src_space, type='curvature', strength=lams)
        lams = 1.0e1
        reg_op_tikh = PriorCovarianceOperator(src_space, type='tikhonov', strength=lams)

        lensop = ManifoldLensOperator(image_space, src_space, lensmodel, z_src=2.059)

        psfop = imdata.blurring_operator 


        lensed = lensop*testsrc

        plt.imshow(lensed[0,0,:].T, extent=image_space.bounds, **imargs)
        plt.title("Lensed")
        plt.show()

        psfop = imdata.blurring_operator 
        blurred = psfop*lensed
        plt.imshow(blurred[0,0,:].T, extent=image_space.bounds, **imargs)
        plt.show()

        noised = blurred + np.random.normal(scale=imdata.sigma) 
        plt.imshow(noised[0,0,:].T, extent=image_space.bounds, **imargs)
        plt.show()


        covop = imdata.covariance_operator
        response = psfop * lensop
        #rhs = response.T * covop * imdata.data 
        rhs = response.T * covop * noised

        '''
        # Direct solve of the full system
        # TODO: lhs._mat !
        lu = linalg.splu(reg_op._mat + lensop._mat.T @ psfop._mat.T @ covop._mat @ psfop._mat @ lensop._mat)
        #print("Logdet (direct) =", chol.logdet())
        sol = lu.solve(rhs.flatten()).reshape(src_space.shape)
 
        plt.imshow(sol[0,0].T, extent=src_space.bounds, **imargs)
        plt.title("Solution")
        plt.show()

        lensed = lensop * sol
        plt.imshow(lensed[0,0,:].T, extent=image_space.bounds, **imargs)
        plt.title("Lensed")
        plt.show()

        sol_direct = sol
        '''

        # Set up operators
        #lhs = response.T * covop * response + reg_op
        lhs = response.T * covop * response + reg_op+reg_op_tikh

        # preconditioner and solve
        #lu = linalg.splu(reg_op._mat + lensop._mat.T @ covop._mat @ lensop._mat)
        lu = linalg.splu(reg_op._mat + reg_op_tikh._mat + lensop._mat.T @ covop._mat @ lensop._mat)
        #print("Logdet (PC) =", cholpc.logdet())
        lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
        pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
        sol, info = linalg.cg(lhs_op, rhs.flatten(), tol=1.0e-10, atol=1.0e-14, M=pc)
        err_max = max_relative_error(lhs*sol, rhs)
        self.assertLess(err_max, errtol) 
        sol = sol.reshape(src_space.shape)
 
        #err_max = max_relative_error(sol, sol_direct)
        #print("err max =", err_max)
        #self.assertLess(err_max, errtol) 
 
        plt.imshow(sol[0,0].T, extent=src_space.bounds, **imargs)
        plt.title("Solution CG")
        plt.show()

        lensed = lensop * sol
        plt.imshow(lensed[0,0,:].T, extent=image_space.bounds, **imargs)
        plt.title("Lensed")
        plt.show()




    def test_lensmodel(self):

        return

        lensmodel = GlobalLensModel()
        lensmodel.add_component(PowerLawEllipsoid())
        lensmodel.add_component(ExternalPotential())
        #lensmodel.add_component(SPEMD(z=0.15))
        #lensmodel.add_component(SPEMD(z=0.25))
        #lensmodel.add_component(SPEMD(z=4.05))
        #lensmodel.add_component(SPEMD(z=0.05))

        points = np.random.normal(size=(4,2))

        lensmodel.deflect(points, 2.059)


        #lensmodel.add_component(SIE(...))


    def test_delaunay_lens_operator(self):

        return

        # load image data 
        # it creats the data covariance and psf operators automatically
        imdata = OpticalDataset(f'{testpath}/data_optical_2d/input/data.fits',
                noise=0.0304896, mask=f'{testpath}/data_optical_2d/input/mask.fits', 
                psf=f'{testpath}/data_optical_2d/input/psf.fits', psf_support=21,
                bounds=[(-0.72,0.72),(-0.67,0.67)])
        image_space = imdata.space 
        print("data max =", np.max(imdata.data))
        print("data shape =", imdata.data.shape)
        plt.imshow(imdata.data.T, extent=image_space.bounds, **imargs)
        plt.show()

        # make a lens model
        #lensmodel = LensModel() # default optical test for now

        lensmodel = GlobalLensModel()
        lensmodel.add_component(PowerLawEllipsoid(z=0.35))
        lensmodel.add_component(ExternalPotential(z=0.35))
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=2.059, ncasted=1)
        src_space = lensop.space_right

        # TODO: replace this bogus test source with reference data from the old code
        points = src_space.points
        testsrc = np.exp(-1.0/(2.0*0.02**2)*np.sum((points)**2, axis=-1))
        cr = np.array([0.03,0.04])
        testsrc += 2.0*np.exp(-1.0/(2.0*0.01**2)*np.sum((points-cr)**2, axis=-1))
        plt.tripcolor(points[:,0],points[:,1], src_space.triangles, testsrc, shading='gouraud')
        plt.show()

        # apply operators and noise
        lensed = lensop.apply(testsrc)
        plt.imshow(lensed[0,0,:].T, extent=image_space.bounds, **imargs)
        plt.show()

        psfop = ConvolutionOperator(image_space, fitsfile=f'{testpath}/data_optical_2d/input/psf.fits', kernelsize=21)
        blurred = psfop.apply(lensed)
        plt.imshow(blurred[0,0,:].T, extent=image_space.bounds, **imargs)
        plt.show()

        noised = blurred + np.random.normal(scale=imdata.sigma) 
        plt.imshow(noised[0,0,:].T, extent=image_space.bounds, **imargs)
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










####### runs all tests by default #######
if __name__ == '__main__':
    main(verbosity=2)
