import numpy as np

def split_x(opt, x, **kwargs):
    if opt == 'pt':
        return x[2], x[:2]   
    elif opt == 'triple':
        return x[3], x[:3]
    else:
        return kwargs['T'], x
    
def compute_var(self, opt, phases, bounds, **kwargs):
    
    if 'point' in kwargs.keys():
        x = kwargs['point']
    else:
        x = self.compute_mean(opt, phases, bounds, **kwargs) 
    
    t, v = split_x(opt, x, **kwargs)
    
    dF, deltas = self._dF(opt, phases, t, v)
    V = self._cov(opt, phases, t, v, **kwargs)
    
    sigmas = []
    for el in deltas:
        sigmas.append((el@V@el.T)[0, 0]**(1/2.))
    sigma_ar = np.array(sigmas, dtype = object).reshape((len(sigmas), 1))
    
    eq_error = dF@sigma_ar
    if opt == 'tp':
        sigmas.append(np.sum(dF@sigma_ar))
        x = x.tolist()
        x.append(-self.d_func(t, v[0], phase = phases[0], d = "d_0_1")*t)
    return x, sigmas, eq_error
