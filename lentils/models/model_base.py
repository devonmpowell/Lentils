import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, uniform, loguniform 


_statfuncs = {
    'normal': {'func': norm, 'pars': ['mu', 'sigma'], 'map': lambda mypars: {'loc': mypars['mu'], 'scale': mypars['sigma']}},
    'lognormal': {'func': lognorm, 'pars': ['s'], 'map': lambda mypars: mypars},
    'uniform': {'func': uniform, 'pars': ['a', 'b'], 'map': lambda mypars: {'loc': mypars['a'], 'scale': (mypars['b']-mypars['a'])}},
    'loguniform': {'func': loguniform, 'pars': ['a', 'b'], 'map': lambda mypars: mypars},
}


class Prior:

    def __init__(self, type, **kwargs):
        if type not in _statfuncs:
            raise ValueError(f"{type} is not a valid prior type. Implemented priors are {list(_statfuncs.keys())}.")
        self.type = type
        self.func = _statfuncs[type]['func']
        self.mypars = {}
        for par in _statfuncs[type]['pars']:
            x = float(kwargs[par])
            setattr(self, par, x)
            self.mypars[par] = x
        self.funcpars = _statfuncs[type]['map'](self.mypars)

    def inverse_cdf(self, x):
        return self.func.ppf(x, **self.funcpars)

    def pdf(self, x):
        return self.func.pdf(x, **self.funcpars)

    def sample(self, size=1):
        return self.func.rvs(x, size=size, **self.funcpars)

    def __repr__(self):
        return f'Prior({self.type}, {self.mypars})' 


class Par:
    
    def __init__(self, initial_value=None, free=False, prior=None):

        if initial_value is not None:
            self.value = float(initial_value)
        self.free = bool(free)
        if prior is not None:
            if isinstance(prior, Prior):
                self.prior = prior
            else:
                self.prior = Prior(**prior)
        else:
            self.prior = None 

    def __float__(self):
        return self.value

    def __repr__(self):
        return f"{self.value}, free = {self.free}, prior = {self.prior}"


class Model:
    
    def __init__(self, **pars):



par = Par(0.1, free=True, prior=Prior(type='lognormal', s=0.40))
#par = Par(0.1, free=True, prior=Prior(type='normal', mu=1.0, sigma=0.1))
#par = Par(0.1, free=True, prior=Prior(type='uniform', a=5.0, b=10.0))
#par = Par(0.1, free=True, prior=Prior(type='loguniform', a=1.0, b=10.0))
print(par, float(par))


npts = 1000
x = np.linspace(0.0, 1.0, npts)
invcdf = par.prior.inverse_cdf(x) 
plt.plot(x, invcdf, 'r-', lw=5, alpha=0.6, label='norm pdf')
plt.show()




print(par.value)
par2 = par
print(par2.value)
par2.value = 3.0
print(par2)
print(par)
