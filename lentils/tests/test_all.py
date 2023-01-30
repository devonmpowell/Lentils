import unittest
from unittest import TestCase, main
import numpy as np

from .test_utils import testpath, errtol, max_relative_error
from .test_utils import imargs



from lentils.common import * 
from lentils.operators import * 
from lentils.models import * 

from astropy.table import Table
import matplotlib.pyplot as plt

import copy
from timeit import default_timer as timer


class LensTests(TestCase):


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


        # load image data 
        # it creats the data covariance and psf operators automatically
        imdata = OpticalDataset(f'{testpath}/data_optical_2d/input/data.fits',
                noise=0.0304896, mask=f'{testpath}/data_optical_2d/input/mask.fits', 
                psf=f'{testpath}/data_optical_2d/input/psf.fits', psf_support=21,
                bounds=[(-0.72,0.72),(-0.67,0.67)])
        image_space = imdata.space 
        print("data max =", np.max(imdata.data))
        print("data shape =", imdata.data.shape)
        plt.imshow(imdata.data.T, extent=image_space._bounds.flatten(), **imargs)
        plt.show()

        # make a lens model
        #lensmodel = LensModel() # default optical test for now

        lensmodel = GlobalLensModel()
        lensmodel.add_component(PowerLawEllipsoid(z=0.35))
        lensmodel.add_component(ExternalPotential(z=0.35))
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

        psfop = ConvolutionOperator(image_space, fitsfile=f'{testpath}/data_optical_2d/input/psf.fits', kernelsize=21)
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



    def test_mock_radio(self):

        return

        image_space = ImageSpace(shape=(1024,1024), bounds=[(-1.0, 0.2,), (-0.5, 0.7)], 
                mask=f'{testpath}/data_radio_2d/new_mock/mask_connected_thresh5_pad3.fits')
        uvdata = RadioDataset(f'{testpath}/data_radio_2d/new_mock/j0751_lwmp_nopos_nosub_0.000000_tiny.uvfits', image_space=image_space)
        lensmodel = LensModel(b=0.462437, th=19.162881, f=0.899243, x=-0.441519, y=0.175067, 
                rc=0.0, qh=0.449087, ss=0.092515, sa=73.698888, z=0.35)












####### runs all tests by default #######
if __name__ == '__main__':
    main(verbosity=2)
