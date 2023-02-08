
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, realpath

# path to this test directory
testpath = dirname(realpath(__file__))

# default tolerances for floating-point checks
errtol = 1.0e-6 

# custom definition for relative error between two arrays 
def max_relative_error(a, b, relative_to='max'):
    norm = 1.0
    if relative_to == 'element':
        norm = np.abs(b)
    elif relative_to == 'max':
        norm = np.max(np.abs(b))
    maxerr = np.max(np.abs((a-b)/norm))
    return maxerr

# Arguments for imshow spot-checks
imargs = {'origin': 'lower', 'interpolation': 'nearest', 'cmap': plt.cm.jet}
tripargs = {'shading': 'gouraud', 'cmap': plt.cm.jet}

# callback for CG solver iterations
def cg_callback(x):
    print('.', end='', flush=True)

