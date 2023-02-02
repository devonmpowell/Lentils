
from unittest import TestCase, main
import numpy as np
import astropy.io.fits as fits 
from .test_utils import testpath, errtol, max_relative_error 
from lentils.common import VisibilitySpace, ImageSpace, RadioDataset 
from lentils.operators import NUFFTOperator, DFTOperator, ConvolutionOperator


class NUFFTTests(TestCase):

    def test_convolution(self):
        
        imspace = ImageSpace(shape=[127,129])
        delta = imspace.new_vector()
        delta[0,0,50,40] = 1.0
        delta[0,0,100,50] = 1.0
        delta[0,0,70,100] = 0.5
        delta[0,0,60,1] = 1.0
        delta[0,0,125,50] = 1.0
        delta[0,0,64,63] = 0.8
        delta[0,0,2,50] = 1.5
        delta[0,0,80,126] = 1.0

        for ksize in [32, 33]:

            conv_mat = ConvolutionOperator(imspace, fitsfile=f'{testpath}/data_misc/arrow{ksize}.fits', fft=False)
            conv_fft = ConvolutionOperator(imspace, fitsfile=f'{testpath}/data_misc/arrow{ksize}.fits', fft=True)
    
            # forward convolution
            out_mat = conv_mat*delta
            out_fft = conv_fft*delta
            err_max = max_relative_error(out_mat, out_fft)
            self.assertLess(err_max, errtol) 
    
            # transpose direction
            out_mat_t = conv_mat.T*out_mat
            out_fft_t = conv_fft.T*out_mat
            err_max = max_relative_error(out_mat_t, out_fft_t)
            self.assertLess(err_max, errtol) 


    def test_gridder_and_apodization(self):

        # make a fake uv coordinate set at different scales in units of du
        uv_per_scale = 4 
        all_uv = []
        for scale in np.linspace(0.0,55.0,20):
            all_uv.append([np.random.normal(scale=1e5*scale,size=(uv_per_scale,3))])
        all_uv = np.array(all_uv, dtype=np.float64).reshape((-1,3))
        uv_space = VisibilitySpace(channels=[0.5], uvcoords=all_uv)
        image_space = ImageSpace(shape=(236,257), bounds=[(-0.5,0.7),(-0.7,0.5)])

        # put some fake data through the pipeline
        vdata = np.random.normal(size=(uv_space.num_rows,2)).view(np.complex128)
        nufft = NUFFTOperator(uv_space, image_space)
        dft = DFTOperator(uv_space, image_space)
        nufftvec = nufft.T * vdata
        dftvec = dft.T * vdata
        err_max = max_relative_error(dftvec, nufftvec)
        self.assertLess(err_max, errtol) 

        # Test the forward operation as well
        nufftvis = nufft * dftvec
        dftvis = dft * dftvec
        err_max = max_relative_error(dftvis, nufftvis)
        self.assertLess(err_max, errtol) 


    def test_blurred_covariance(self):

        # TODO: test this for non-square and odd grids as well.

        # load data and a model image
        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.15, 0.35,), (-0.65, 0.85)], 
                mask=f'{testpath}/data_radio_2d/input/mask_1024_zoom.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits',
                image_space=image_space)
        with fits.open(f'{testpath}/data_radio_2d/reference/best_Ls_reference_dft.fits') as f:
            imdata = np.ascontiguousarray(f['PRIMARY'].data.T[:,:,0].astype(np.float64))

        # do the blurred covariance three different ways
        nufft = uvdata.nufft_operator
        cov = uvdata.covariance_operator
        bcb = uvdata.blurred_covariance_operator 
        dft = DFTOperator(uvdata.space, image_space)
        result_explicit = nufft.T * cov * nufft * imdata
        result_convolved = bcb * imdata
        result_dft = dft.T * cov * dft * imdata

        # check results against each other
        err_max = max_relative_error(result_convolved, result_explicit)
        self.assertLess(err_max, errtol) 
        err_max = max_relative_error(result_convolved*image_space.mask, result_dft)
        self.assertLess(err_max, errtol) 
        err_max = max_relative_error(result_explicit*image_space.mask, result_dft)
        self.assertLess(err_max, errtol) 


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

        # check that the sum of weights is equal to the beam max
        cvec = uvdata.covariance_operator.T * uvdata.space.new_vector(1.0+0.0j)
        wtsum = np.sum(cvec)
        self.assertAlmostEqual(1.0, np.max(uvdata.dirty_beam)/wtsum, delta=errtol) 

        # check against DFT computation
        dft = DFTOperator(uvdata.space, uvdata._space_beam)
        dft_dirty_beam = dft.T * cvec
        err_max = max_relative_error(uvdata.dirty_beam, dft_dirty_beam)
        self.assertLess(err_max, errtol) 

        # check against reference data
        with fits.open(f'{testpath}/data_radio_2d/reference/dirty_beam_reference_fft.fits') as f:
            reference = f['PRIMARY'].data.T[cx-nx:cx+nx,cx-nx:cx+nx,0]
        err_max = max_relative_error(uvdata.dirty_beam, reference)
        self.assertLess(err_max, errtol) 


    def test_dirty_image(self):

        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.15, 0.35,), (-0.65, 0.85)], 
                mask=f'{testpath}/data_radio_2d/input/mask_1024_zoom.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/input/j0751_small_snr1x.uvfits',
                image_space=image_space)

        # check against DFT computation
        dft = DFTOperator(uvdata.space, image_space)
        dft_dirty_image = dft.T * uvdata.covariance_operator.T * uvdata.data
        err_max = max_relative_error(uvdata.dirty_image*image_space.mask, dft_dirty_image)
        self.assertLess(err_max, errtol) 

        # compare to a reference FFT dataset 
        with fits.open(f'{testpath}/data_radio_2d/reference/dirty_image_reference_fft.fits') as f:
            reference = f['PRIMARY'].data.T[:,:,0]
        err_max = max_relative_error(uvdata.dirty_image, reference)
        self.assertLess(err_max, errtol) 

        # compare to reference DFT data set
        with fits.open(f'{testpath}/data_radio_2d/reference/dirty_image_reference_dft.fits') as f:
            reference = f['PRIMARY'].data.T[:,:,0]
        err_max = max_relative_error(uvdata.dirty_image*image_space.mask, reference)
        self.assertLess(err_max, errtol) 




# run tests in this file
if __name__ == '__main__':
    main(verbosity=2)

