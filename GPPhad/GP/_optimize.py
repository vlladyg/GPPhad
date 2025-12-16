import numpy as np
from scipy.optimize import minimize
import gmpy2 as gp
gp.get_context().precision=400

import mpinv
from mpinv import fast_mp_matrix_inverse_symm as f_mp_inv
from mpinv import fast_mp_matrix_logdet_symm as f_mp_logdet

def marg_like(self, th, phase = None, recomp = False):
    """Marginal likelihood estimation for the given Gaussian process and list of hyperparameters

    INPUT:

    set of hyperparameters (th)

    phase (phase)

    optinal recomputation of inverve covariance matrix of the GP (recomp)

    OUTPUT:

    -mar - marginal likelihood value * -1

    """
    if not self.melt:
        if recomp:
            print(phase, th)
            self.th = th

            # Consruction of new covarience matrix (K) and its inverse (K_inv) for particular phase
            self.K[phase] = self.constr_matrix(self.X[phase])
            self.err_m[phase] = self.constr_matrix(self.err[phase])
            self.K_inv[phase] = f_mp_inv(self.K[phase] + self.err_m[phase])

        # [0, 0] element corresponds to the value
        val_1 = 1./2.*(self.Y[phase].T@self.K_inv[phase]@self.Y[phase])[0, 0]
        val_2 = 1./2.*f_mp_logdet(self.K[phase] + self.err_m[phase])
        val_3 = len(self.Y[phase])/2.*gp.log(2.*gp.const_pi(200))

        mar = -val_1 - val_2 - val_3
        print(-mar)
    else:
        if recomp:
            self.th = th

            # Consruction of new covarience matrix (K) and its inverse (K_inv) for particular phase
            self.K = self.constr_matrix_melt(self.X)
            self.err_m = self.constr_matrix_melt(self.err)
            self.K_inv = f_mp_inv(self.K + self.err_m)

        val_1 = 1./2.*(self.Y.T@self.K_inv@self.Y)[0, 0] # [0, 0] element corresponds to the value
        val_2 = 1./2.*f_mp_logdet(self.K + self.err_m)
        val_3 = len(self.Y)/2.*gp.log(2.*gp.const_pi(200))

        mar = -val_1 - val_2 - val_3
        print(-mar)

    if 0.0 == np.nan_to_num(np.float64(-mar)):
        return 10**10
    else:
        return np.float64(-mar)

#Decorator of optimizer
def optimize(self, th, ind = None, phase = None, optimizer = minimize,
             method = 'L-BFGS-B', grad = False, bounds = None, tol = 1e-12,
             rm = True):
        """

        Optimization of the hyperparameters

        INPUT:

            th - starting point of the algorythm;


            this array represents the set of hyperparametes that is fixed;


            optimizer - used optimized, scipy.minimize by default;

            method - used optimization algorythm, L-BFGS-B  by default;

            bounds - bounds for variables

            grad - option if analytical gradient is known, False by default


        OUTPUT:

            result - value of marginal likelihood (MG)

            th - final set of hyperparametes 

        """
        # Multiphase optimization
        th_0 = []

        if not isinstance(ind, type(None)):    
            for el in ind:
                th_0.append(th[el])
        else:
            th_0 = th

        def mg(x):
            if not isinstance(ind, type(None)):
                th_new = th
                for i, el in enumerate(ind):
                    th_new[el] = gp.mpfr(x[i])
            else:
                th_new = list(map(gp.mpfr, x))
            return self.marg_like(th_new, phase = phase, recomp = True)

        if grad:
            opt_obj = optimizer(mg, jac = self.mg_grad, x0=th_0,
                                method=method, bounds = bounds, tol = tol)#, options = {'maxiter': 200})
        else:
            opt_obj = optimizer(mg, x0=th_0, method=method, 
                                bounds = bounds, tol = tol)#, options = {'maxiter': 200})

        th = self.th
        result = opt_obj.fun

        return result, th
