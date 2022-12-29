import sys
sys.path.append('../')
import check_routines as cr


# lists of files to check against those in 'reference'
data_to_check = ['best_%s.data']
source_to_check = ['best_%s_source.data']
fits_to_check = ['best_%s.fits', 'data_%s.fits', 'best_Ls_%s.fits', 'best_residuals_%s.fits', 'best_%s_source.fits']

num_tot = 0
num_pass = 0

# check the data output (lens parameters and evidence)
for basename in data_to_check:
    testfile = basename % 'test' 
    reffile = 'reference/' + (basename % 'reference') 
    print ' - Checking %s against %s... ' % (testfile, reffile)
    num_tot += 1
    num_pass += cr.check_data_basic(testfile, reffile)


# check the source file (triangulated from text file)
for basename in source_to_check:
    testfile = basename % 'test' 
    reffile = 'reference/' + (basename % 'reference') 
    print ' - Checking %s against %s... ' % (testfile, reffile)
    num_tot += 1
    num_pass += cr.check_source_2d(testfile, reffile)


# check the fits files 
for basename in fits_to_check:
    testfile = basename % 'test' 
    reffile = 'reference/' + (basename % 'reference') 
    print ' - Checking %s against %s... ' % (testfile, reffile)
    num_tot += 1
    num_pass += cr.check_fits(testfile, reffile)

# done
print ' - Passed %d of %d checks' % (num_pass, num_tot) 





