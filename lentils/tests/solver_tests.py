
from unittest import TestCase, main
import numpy as np
import astropy.io.fits as fits 
from .test_utils import testpath, errtol, max_relative_error 
from lentils.common import VisibilitySpace, ImageSpace, OpticalDataset, RadioDataset 
from lentils.operators import NUFFTOperator, DFTOperator, DelaunayLensOperator, PriorCovarianceOperator
from lentils.models import LensModel

#from .test_utils import plt, imargs
#from sksparse.cholmod import cholesky
import scipy.sparse.linalg as linalg 

class SolverTests(TestCase):

    def test_cg_optical(self):

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
        lensmodel = LensModel() # default optical test for now
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=2.059, ncasted=3, mask=imdata.mask)
        src_space = lensop.space_right
        points = src_space.points

        # source prior
        lams = 109.832868 
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)

        # Set up operators
        response = psfop * lensop
        lhs = response.T * covop * response + reg_op
        rhs = response.T * covop * imdata.data 

        # preconditioner and solve
        lu = linalg.splu(reg_op._mat + lensop._mat.T @ covop._mat @ lensop._mat)
        #print("Logdet (PC) =", cholpc.logdet())
        lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
        pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
        sol_cg, info = linalg.cg(lhs_op, rhs, tol=1.0e-10, atol=1.0e-14, M=pc)
        err_max = max_relative_error(lhs*sol_cg, rhs)
        self.assertLess(err_max, errtol) 

        # Direct solve of the full system
        # TODO: lhs._mat !
        lu = linalg.splu(reg_op._mat + lensop._mat.T @ psfop._mat.T @ covop._mat @ psfop._mat @ lensop._mat)
        #print("Logdet (direct) =", chol.logdet())
        sol_direct = lu.solve(rhs)
        err_max = max_relative_error(sol_cg, sol_direct)
        self.assertLess(err_max, errtol) 

        # compare to a reference FFT dataset 
        # TODO: why does this differ from the reference source??
        reference = np.unique(np.loadtxt(f'{testpath}/data_optical_2d/reference/best_reference_source.data', skiprows=1, usecols=3))
        err_max = max_relative_error(np.sort(sol_cg), np.sort(reference))
        print("Err max (relative to reference file) =", err_max)
        #self.assertLess(err_max, errtol) 

        #plt.tripcolor(points[:,0],points[:,1], src_space.tris, sol_cg, shading='gouraud')
        #print("Solution max =", np.max(sol_cg))
        #plt.title('Solution')
        #plt.show()


    def test_cg_radio(self):

        # load mock data
        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.15, 0.35,), (-0.65, 0.85)], 
                mask=f'{testpath}/data_radio_2d/input/mask_1024_zoom.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits', image_space=image_space)
        lensmodel = LensModel(b=0.402005, th=48.987322, f=0.796415, x=-0.445178, y=0.178450, 
                rc=1.0e-4, qh=0.504365, ss=0.070171, sa=75.522635, z=0.35)
        lensop = DelaunayLensOperator(image_space, lensmodel, z_src=3.2, ncasted=5)
        src_space = lensop.space_right
        points = src_space.points

        # Set up operators
        lams = 2.36e11
        reg_op = PriorCovarianceOperator(src_space, type='gradient', strength=lams)
        lhs = lensop.T * uvdata.blurred_covariance_operator * lensop + reg_op
        rhs = lensop.T * uvdata.dirty_image

        # preconditioner and CG solve
        # TODO: cheaper setup for weight sum
        wtsum = np.sum(np.abs(uvdata.covariance_operator.T * uvdata.space.new_vector(1.0)))
        lu = linalg.splu(reg_op._mat + wtsum * lensop._mat.T @ lensop._mat)
        #def logdet(dcmp):
            #usum = np.sum(np.log(dcmp.U.diagonal().astype(np.complex128))) 
            #lsum = np.sum(np.log(dcmp.L.diagonal().astype(np.complex128)))
            #print("lsum, usum =", lsum, usum)
            #return lsum+usum 
        #print("Logdet (PC) =", logdet(lu))
        lhs_op = linalg.LinearOperator((src_space.size,src_space.size), matvec=lhs.apply)
        pc = linalg.LinearOperator((src_space.size, src_space.size), lu.solve)
        sol, info = linalg.cg(lhs_op, rhs, tol=1.0e-10, atol=1.0e-14, M=pc)
        err_max = max_relative_error(lhs*sol, rhs)
        self.assertLess(err_max, errtol) 

        # check against DFT computation
        # TODO: concatenating explicit mats automatically
        dft = DFTOperator(uvdata.space, image_space)
        rhs = lensop.T * dft.T * uvdata.covariance_operator_dft * uvdata.data 
        response = dft._mat * lensop._mat
        lhs = response.T * uvdata.covariance_operator_dft._mat * response + reg_op._mat
        lu = linalg.splu(lhs)
        #print("Logdet (Exact) =", logdet(lu))
        sol_dft = lu.solve(rhs)
        err_max = max_relative_error(sol, sol_dft)
        self.assertLess(err_max, errtol) 

        # TODO: check these also against reference solutions?




