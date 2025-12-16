import numpy as np
from gmpy2 import mpfr, get_context
get_context().precision=400

from mpinv import fast_mp_matrix_inverse_symm as f_mp_inv
from mpinv import fast_mp_matrix_logdet_symm as f_mp_logdet

#X, Y and err to phases
def _constr_not_melt(self):
    """Asserts err_m, K and K_inv to the GP"""

    self.err_m = {}

    self.K = {}
    self.K_inv = {}

    for phase in self.phases:
        if not isinstance(self.err[phase], type(None)):
            self.err_m[phase] = self.constr_matrix_melt(self.err[phase])
        else:
            self.err[phase] = None
            self.err_m[phase] = turn_to_gp(np.eye(len(self.X[phase]), dtype = object)*1e-8)
        ######################################################################## 
        self.K[phase] = self.constr_matrix_melt(self.X[phase])
        self.K_inv[phase] = f_mp_inv(self.K[phase] + self.err_m[phase])

    pass

def _constr_melt(self):
    """Construct err_m, K, K_inv for united GP"""

    if not isinstance(self.err, type(None)):
        self.err_m = self.constr_matrix_melt(self.err)
    else:
        self.err_m = turn_to_gp(np.eye(len(self.X), dtype = object)*1e-8)


    self.K = self.constr_matrix_melt(self.X)
    self.K_inv = f_mp_inv(self.K + self.err_m)

#Constr final
def _constr(self):
    """Aggregator of constr"""
    self.th = list(map(mpfr, self.th))
    if self.melt:
        self._constr_melt()
    else:
        self._constr_not_melt()