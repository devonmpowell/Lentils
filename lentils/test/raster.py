import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import operatorstack
from operatorstack.dataspace import * 
from operatorstack.operator import * 
from operatorstack.lensmodel import * 
from operatorstack.dataset import * 

# set tolerances for floating-point checks
errtol = 1.0e-6 
imargs = {'origin': 'lower', 'interpolation': 'nearest', 'cmap': plt.cm.Spectral}

fig, axes = plt.subplots(1,3,figsize=(12,4))


# load image data just for mask and noise
testdir = 'tests/optical_2d_quick/input'
imdata = Dataset.image_from_fits('{}/data.fits'.format(testdir), 
        noise=0.0304896, maskfits='{}/mask.fits'.format(testdir), bounds=[(-0.72,0.72),(-0.67,0.67)])
image_space = imdata.space 
src_space = ImageSpace(name='SourceSpace', shape=(17,18))
print("data max =", np.max(imdata.data))
print("data shape =", imdata.data.shape)
#ax.imshow(imdata.data.T, extent=image_space._bounds.flatten(), **imargs)
#ax.show()

# make a lens model
lensmodel = LensModel() # default optical test for now
lensop = ManifoldLensOperator('L', image_space, src_space, lensmodel, z_src=2.059, ncasted=1, mask=imdata.mask)

tri0 = np.array([[-0.5, 0.1],[0.1, 0.5],[0.4, -0.4]])
r0 = lensop.rasterize(tri0)

trih = tri0.copy() 
trih[0,1] += errtol
r1 = lensop.rasterize(trih)

nztot = np.sum((r0 != 0.0))
print("Total nonzero =", nztot)


# apply operators and noise
ax = axes[0]
#ax.imshow((r1.T-r0.T)/errtol, extent=src_space._bounds.flatten(), **imargs)
#ax.set_title('Finite diff')
ax.imshow(r0.T, extent=src_space._bounds.flatten(), **imargs)

p = Polygon(tri0, fill=False, ec='gray')
ax.add_patch(p)
for v in range(3):
    ax.scatter(tri0[v][0], tri0[v][1], s=10, c='k')
    ax.text(tri0[v][0], tri0[v][1]+0.05, r'$\mathbf{x}_%d=(%.2f,%.2f)$'%(v, tri0[v][0], tri0[v][1]), ha='center')
plt.show()



sys.exit(0)

# TODO: replace this bogus test source with reference data from the old code
points = src_space.points
testsrc = np.exp(-1.0/(2.0*0.02**2)*np.sum((points)**2, axis=-1))
cr = np.array([0.03,0.04])
testsrc += 2.0*np.exp(-1.0/(2.0*0.01**2)*np.sum((points-cr)**2, axis=-1))
ax.tripcolor(points[:,0],points[:,1], src_space.tris, testsrc, shading='gouraud')
#ax.set_xlims(*image_space._bounds[0])
#ax.set_ylims(*image_space._bounds[1])
ax.show()





# apply operators and noise
lensed = lensop.apply(testsrc)
ax.imshow(lensed.T, extent=image_space._bounds.flatten(), **imargs)
ax.show()

psfop = ConvolutionOperator('B', image_space, fitsfile='{}/psf.fits'.format(testdir), kernelsize=21)
blurred = psfop.apply(lensed)
ax.imshow(blurred.T, extent=image_space._bounds.flatten(), **imargs)
ax.show()

noised = blurred + np.random.normal(scale=imdata.sigma) 
ax.imshow(noised.T, extent=image_space._bounds.flatten(), **imargs)
plt.show()

