
import numpy as np
import astropy.io.fits as fits 
from lentils.common import VisibilitySpace 

def _load_uvfits(file, combine_stokes):

    # open the fits file 
    allhdu = fits.open(file, memmap=True)
    hdu = allhdu['PRIMARY']
    data = hdu.data
    header = hdu.header

    # get the header into a dict that we can access via the axis name
    axes = {}
    for ax in range(1,header['NAXIS']+1):
        nax = header['NAXIS%d'%ax]
        if nax == 0:
            continue
        axtmp = {'NAXIS': nax,}
        for label in ['CRVAL','CDELT','CRPIX','CROTA']:
            axtmp[label] = header['%s%d'%(label,ax)]
        axes[header['CTYPE%d'%ax]] = axtmp

    # shape the data
    num_rows = header['GCOUNT']
    num_spw = axes['IF']['NAXIS']
    num_channels = axes['FREQ']['NAXIS']
    num_stokes = axes['STOKES']['NAXIS']
    ref_freq = axes['FREQ']['CRVAL']
    ref_ch = axes['FREQ']['CRPIX']-1 # convert from 1-based indexing

    # get frequency data
    # TODO: check them against the C code
    fqhead = allhdu['AIPS FQ'].header
    fqdata = allhdu['AIPS FQ'].data
    channels = (np.multiply.outer(fqdata['CH WIDTH'].reshape((num_spw)), (np.arange(num_channels)-ref_ch)) \
            + fqdata['IF FREQ'].reshape((num_spw,1)) + ref_freq).flatten()
    num_channels = num_channels*num_spw 
    assert channels.size == num_channels

    # read in data
    # TODO: original data is only 32-bit...
    uvcoords = np.array([data['UU'],data['VV'],data['WW']]).T.astype(np.float64, order='C')
    rawvisdata = data['DATA'][:,0,0,:,:,:,0] + 1j*data['DATA'][:,0,0,:,:,:,1]
    rawweights = data['DATA'][:,0,0,:,:,:,2]
    rawmask = (rawweights > 0.0) 
    rawsigma = np.zeros_like(rawweights)
    rawsigma[rawmask] = rawweights[rawmask]**-0.5

    # combine stokes parameters if desired
    # TODO: add a check of the stokes parameter names
    if combine_stokes:
        rawmask = np.all(rawmask,axis=-1)
        # TODO: check this math 
        rr = rawvisdata[:,:,:,0] 
        ll = rawvisdata[:,:,:,1] 
        rawvisdata = 0.5*(rr+ll)
        srr = rawsigma[:,:,:,0] 
        sll = rawsigma[:,:,:,1] 
        rawsigma = np.zeros_like(srr)
        rawsigma[rawmask] = 0.5*np.sqrt(srr[rawmask]**2+sll[rawmask]**2)
        num_stokes = 1
    else:
        rawvisdata = rawvisdata[:,:,:,0] 
        rawsigma = rawsigma[:,:,:,0] 
        rawmask = rawmask[:,:,:,0] 

    # shape and store
    uvspace = VisibilitySpace(channels=channels, uvcoords=uvcoords)
    reordered_data = np.moveaxis(rawvisdata.reshape((uvspace.num_rows, uvspace.num_channels, uvspace.num_stokes)), [0,1,2], [2,0,1]).astype(np.complex128, order='C')
    reordered_sigma = np.moveaxis(rawsigma.reshape((uvspace.num_rows, uvspace.num_channels, uvspace.num_stokes)), [0,1,2], [2,0,1]).astype(np.float64, order='C')
    reordered_mask = np.moveaxis(rawmask.reshape((uvspace.num_rows, uvspace.num_channels, uvspace.num_stokes)), [0,1,2], [2,0,1]).astype(np.bool_, order='C')
    allhdu.close()

    return uvspace, reordered_data, reordered_sigma, reordered_mask

