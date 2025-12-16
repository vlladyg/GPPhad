import numpy as np
from ._X import _constr_X_s0, _constr_X

def _cov_S0(self, opt, phases, t, v):
    X_s0 = _constr_X_s0(opt, phases, t, v)
    m = len(X_s0[phases[0]])
    S_cov = np.zeros((m, m), dtype = object)
    for phase in phases:
        if self.S0[phase]:
            S_cov += self.S0[phase].constr_matrix_melt(X_s0[phase])
            C = self.S0[phase].c_matrix(self.S0[phase].X, X_s0[phase]).T
            S_cov -= C@(self.S0[phase].K_inv)@C.T
    
    return S_cov


def _cov(self, opt, phases, t, v, **kwargs):
    X = _constr_X(self, opt, phases, t, v)
    D = self.constr_matrix_melt(X)
    
    if 'adapt' in kwargs.keys():
        C = np.zeros((len(D), len(self.X)+len(kwargs['adapt']["x_ad"])))
        C[:, :len(self.X)] = self.c_matrix(self.X, X).T
        C[:, len(self.X):] = self.c_matrix(kwargs['adapt']["x_ad"], X).T
        
        res =  D - C@(kwargs['adapt']["K_inv"])@(C.T)
    else:
        C = self.c_matrix(self.X, X).T
        res = D - C@(self.K_inv)@(C.T)
        
    return res + _cov_S0(self, opt, phases, t, v)
    