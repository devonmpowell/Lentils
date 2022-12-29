import sys
sys.path.append('../')
import check_routines as cr
import numpy as np


# lists of files to check against those in 'reference'
data_to_check = ['best_%s.data']
source_to_check = ['best_%s_source.data']
fits_to_check_fft = ['dirty_image_%s.fits', 'dirty_beam_%s.fits', 'best_%s.fits', 'data_%s.fits', 'best_Ls_%s.fits', 'best_residuals_%s.fits', 'best_%s_source.fits']
#fits_to_check_dft = ['dirty_image_%s.fits', 'best_%s.fits', 'data_%s.fits', 'best_Ls_%s.fits', 'best_residuals_%s.fits', 'best_%s_source.fits']
fits_to_check_dft = ['dirty_image_%s.fits', 'best_%s.fits', 'data_%s.fits', 'best_residuals_%s.fits']

num_tot = 0
num_pass = 0

# check the data output (lens parameters and evidence)
for basename in data_to_check:
    testfile = basename % 'test' 
    reffile = 'reference/' + (basename % 'reference_fft') 
    print(' - Checking %s against %s... ' % (testfile, reffile))
    num_tot += 1
    num_pass += cr.check_data_basic(testfile, reffile)
    reffile = 'reference/' + (basename % 'reference_dft') 
    print(' - Checking %s against %s... ' % (testfile, reffile))
    num_tot += 1
    num_pass += cr.check_data_basic(testfile, reffile)


# check the source file (triangulated from text file)
for basename in source_to_check:
    testfile = basename % 'test' 
    reffile = 'reference/' + (basename % 'reference_fft') 
    print(' - Checking %s against %s... ' % (testfile, reffile))
    num_tot += 1
    num_pass += cr.check_source_2d(testfile, reffile)
    reffile = 'reference/' + (basename % 'reference_dft') 
    print(' - Checking %s against %s... ' % (testfile, reffile))
    num_tot += 1
    num_pass += cr.check_source_2d(testfile, reffile)


# check the fits files against reference created with FFT 
for basename in fits_to_check_fft:
    testfile = basename % 'test' 
    reffile = 'reference/' + (basename % 'reference_fft') 
    print(' - Checking %s against %s... ' % (testfile, reffile))
    num_tot += 1
    num_pass += cr.check_fits(testfile, reffile)


# check the fits files against reference created with DFT 
# We need to do this as a special case because of the mask
mask = cr.load_fits('input/mask_1024_zoom.fits')
for basename in fits_to_check_dft:
    testfile = basename % 'test' 
    reffile = 'reference/' + (basename % 'reference_dft') 
    print(' - Checking %s against %s... ' % (testfile, reffile))
    testarr = cr.load_fits(testfile)
    refarr = cr.load_fits(reffile)
    norm = max(np.max(np.abs(testarr)), np.max(np.abs(refarr)))
    testarr /= norm
    refarr /= norm
    if 'source' not in testfile:
        testarr *= mask
        refarr *= mask
    num_tot += 1
    num_pass += cr.assert_equal(testarr/norm, refarr/norm)


# done
print(' - Passed %d of %d checks' % (num_pass, num_tot)) 





