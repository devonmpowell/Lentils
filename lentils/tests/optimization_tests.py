
from unittest import TestCase, main
import numpy as np
import astropy.io.fits as fits 
from .test_utils import testpath, errtol, max_relative_error 
from lentils.common import VisibilitySpace, ImageSpace, OpticalDataset, RadioDataset 
from lentils.operators import NUFFTOperator, DFTOperator, DelaunayLensOperator, PriorCovarianceOperator, DiagonalOperator
from lentils.models import GlobalLensModel, PowerLawEllipsoid, ExternalPotential

from .test_utils import plt, imargs, tripargs
#from sksparse.cholmod import cholesky
import scipy.sparse.linalg as linalg 

def cg_callback(x):
    print('.', end='', flush=True)



class OptimizationTests(TestCase):

    def test_lognormal_optical(self):

        # load image data 
        # it creats the data covariance and psf operators automatically
        imdata = OpticalDataset(f'{testpath}/data_optical_2d/input/data.fits',
                noise=0.0304896, mask=f'{testpath}/data_optical_2d/input/mask.fits', 
                psf=f'{testpath}/data_optical_2d/input/psf.fits', psf_support=21,
                bounds=[(-0.72,0.72),(-0.67,0.67)])
        image_space = imdata.space 
        psfop = imdata.blurring_operator
        covop = imdata.covariance_operator

        # make a lens model
        lensmodel = GlobalLensModel()
        lensmodel.add_component(PowerLawEllipsoid(b=0.463544, th=-14.278754, f=0.799362, x=-0.046847, 
            y=-0.105357, rc=0.000571, qh=0.506730, z=0.881000))
        lensmodel.add_component(ExternalPotential(x=-0.046847, y=-0.105357, ss=-0.046500, sa=7.921300, z=0.881000))
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=2.059, ncasted=1)
        src_space = lensop.space_right
        points = src_space.points

        # source prior
        lams = 10.0
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)
        #reg_op = PriorCovarianceOperator(src_space, type='tikhonov', strength=lams)

        # set up 
        zlast = src_space.new_vector(-1.0)
        slast = np.exp(zlast)
        jz = DiagonalOperator(src_space, slast) # ds/dz = exp(z) = s
        ddata = psfop*lensop*slast - imdata.data

        for iter in range(10):

            # Set up operators
            response = psfop * lensop * jz
            lhs = response.T * covop * response + reg_op
            rhs = -(response.T * covop * ddata + reg_op * zlast)

            # preconditioner and solve
            lu = linalg.splu(reg_op._mat + jz._mat.T @ lensop._mat.T @ covop._mat @ lensop._mat @ jz._mat)
            lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
            pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
            sol_cg, info = linalg.cg(lhs_op, rhs.flatten(), tol=1.0e-10, atol=1.0e-14, M=pc, callback=cg_callback)
            print("Solution max =", np.max(np.abs(sol_cg)))

            # update and prep for next iter
            zlast += sol_cg.reshape(src_space.shape)
            slast = np.exp(zlast)
            jz = DiagonalOperator(src_space, slast) # ds/dz = exp(z) = s
            ls = lensop*slast
            ddata = psfop*ls - imdata.data
    
        im = plt.tripcolor(points[:,0],points[:,1], src_space.triangles, slast[0,0], vmin=0, **tripargs)
        plt.colorbar(im)
        plt.title('Solution')
        plt.show()


        plt.imshow(ls[0,0].T, extent=image_space.bounds, vmin=0, **imargs)
        plt.title('Ls')
        plt.show()

        plt.imshow(ddata[0,0].T, extent=image_space.bounds, **imargs)
        plt.title('Residuals')
        plt.show()


    def test_lognormal_radio(self):


        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.0, 0.2,), (-0.5, 0.7)], 
                mask=f'{testpath}/data_radio_2d/new_mock/mask_connected_thresh5_pad3.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/new_mock/j0751_lwmp_nopos_nosub_0.000000_tiny.uvfits', image_space=image_space)
        bcb = uvdata.blurred_covariance_operator
        dirty_im = uvdata.dirty_image
        wtsum = np.sum(np.abs(uvdata.covariance_operator.T * uvdata.space.new_vector(1.0)))

        # make a lens model
        lensmodel = GlobalLensModel()
        lensmodel.add_component(PowerLawEllipsoid(b=0.462437, th=19.162881, f=0.899243, x=-0.441519, y=0.175067, 
                rc=0.0, qh=0.449087, z=0.35))
        lensmodel.add_component(ExternalPotential(ss=0.092515, sa=73.698888, z=0.35))
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=2.059, ncasted=1)
        src_space = lensop.space_right
        points = src_space.points


        ############## First do a linear solve to get it close ###########

        # Set up operators
        lams = 1.0e11 
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)
        #lams = 1.0e8
        #reg_op_tikh = PriorCovarianceOperator(src_space, type='tikhonov', strength=lams)

        lhs = lensop.T * uvdata.blurred_covariance_operator * lensop + reg_op
        rhs = lensop.T * uvdata.dirty_image

        # preconditioner and CG solve
        # TODO: cheaper setup for weight sum
        lu = linalg.splu(reg_op._mat + wtsum * lensop._mat.T @ lensop._mat)
        lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
        pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
        sol_cg, info = linalg.cg(lhs_op, rhs.flatten(), tol=1.0e-10, atol=1.0e-14, M=pc, callback=cg_callback)

        print("Sone with linear solve")
 
        im = plt.tripcolor(points[:,0],points[:,1], src_space.triangles, sol_cg, **tripargs)
        plt.colorbar(im)
        plt.title('Solution')
        plt.show()

        slo = 1.0e-2*np.max(sol_cg)

        sol_cg[sol_cg < slo] = slo
        zlast = np.log(sol_cg).reshape(src_space.shape)


        ##################################################################

        # source prior
        lams = 1000.0 
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)
        #reg_op = PriorCovarianceOperator(src_space, type='tikhonov', strength=lams)


        # set up 
        #zlast = src_space.new_vector(-10.0)
        slast = np.exp(zlast)
        jz = DiagonalOperator(src_space, slast) # ds/dz = exp(z) = s
        ls = lensop*slast
        ddata = bcb * ls - dirty_im

        for iter in range(10):

            # Set up operators
            lhs = jz.T * lensop.T * bcb * lensop * jz + reg_op
            rhs = -(jz.T * lensop.T * ddata + reg_op * zlast)

            # preconditioner and solve
            wtsum = np.sum(np.abs(uvdata.covariance_operator.T * uvdata.space.new_vector(1.0)))
            lu = linalg.splu(reg_op._mat + wtsum * jz._mat.T @ lensop._mat.T @ lensop._mat @ jz._mat)
            lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
            pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
            sol_cg, info = linalg.cg(lhs_op, rhs.flatten(), tol=1.0e-10, atol=1.0e-14, M=pc, callback=cg_callback)
            print("Solution max =", np.max(np.abs(sol_cg)))

            # update and prep for next iter
            zlast += sol_cg.reshape(src_space.shape)
            slast = np.exp(zlast)
            jz = DiagonalOperator(src_space, slast) # ds/dz = exp(z) = s
            ls = lensop*slast
            #ddata = psfop*ls - imdata.data
            ddata = bcb * ls - dirty_im
    
        im = plt.tripcolor(points[:,0],points[:,1], src_space.triangles, slast[0,0], **tripargs)
        plt.colorbar(im)
        plt.title('Solution')
        plt.show()


        plt.imshow(ls[0,0].T, extent=image_space.bounds, **imargs)
        plt.title('Ls')
        plt.show()

        plt.imshow(ddata[0,0].T, extent=image_space.bounds, **imargs)
        plt.title('Residuals')
        plt.show()





