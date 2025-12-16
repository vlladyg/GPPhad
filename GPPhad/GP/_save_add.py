import dill as pickle
import numpy as np
from scipy.optimize import root_scalar
from mpinv import fast_mp_matrix_inverse_symm as f_mp_inv


def save(self, file, **kwargs):
    with open(file, 'wb') as handle:
        pickle.dump(self.__dict__, handle)

def add_points(self, x, y, err, phase = None):
    """Adds points to the dataset"""
    if not isinstance(phase, type(None)):
        self.K[phase] = self.update(self.K[phase], self.X[phase], x)
        self.Y[phase] = self.Y[phase].tolist()
        self.Y[phase] += y
        self.Y = np.array(Y)

        self.X[phase] += x
        self.err[phase] += err

        self.err_m[phase] = self.constr_matrix_melt(self.err[phase])
        self.K_inv[phase] = f_mp_inv(self.K[phase] + self.err_m[phase])
    else:
        self.K = self.update_matrix(self.K, self.X, x)

        self.Y = self.Y.tolist()
        self.Y += y
        self.Y = np.array(self.Y)

        self.X += x
        self.err += err

        self.err_m = self.constr_matrix_melt(self.err)
        self.K_inv = f_mp_inv(self.K + self.err_m)

def add_H(self, phase, V = None):
    """Adds zero point energy to dataset"""
    X = [[phase, 'd_0_0', 0.0, V, 5, 10**30]]
    res = self.H[phase].predict([['sol', 'd_0_0', 10**30]])
    Y = [[-1/2.0*res[0][0]]]
    err = [['err', hash(tuple(X[0])), 1e-20]]

    self.add_points(X, Y, err)
    
    
def _melt_vol(self, phases, melt_p):
    """Computes volumes of phases at corresponding menting point"""
    P, T = melt_p[0], melt_p[1]
    
    p_1 = lambda v: self.d_func(T, v, phase = phases[0], d = "d_0_1")*T + P
    p_2 = lambda v: self.d_func(T, v, phase = phases[1], d = "d_0_1")*T + P

    v_1 = root_scalar(p_1, bracket=self.bounds[phases[0]]).root
    v_2 = root_scalar(p_2, bracket=self.bounds[phases[1]]).root
    return v_1, v_2

def _melt_point(self, phases, v, melt_p):
    P, T, dT = melt_p[0], melt_p[1], melt_p[2]
    """Returns X, Y, err of the melting point"""
    X_add = [[(T, (phases[0],'d_0_1', T, v[0], self.x_fixed[0], self.x_fixed[1]) )],
        [(T, (phases[1], 'd_0_1', T, v[1], self.x_fixed[0], self.x_fixed[1]))],
        [(T, (phases[0],'d_0_0', T, v[0], self.x_fixed[0], self.x_fixed[1])),
        (-T, (phases[1],'d_0_0', T, v[1], self.x_fixed[0], self.x_fixed[1]))]]
    e_1 = self.d_func(T, v[0], phase = phases[0], d = "d_1_0")
    e_2 = self.d_func(T, v[1], phase = phases[1], d = "d_1_0")
    f_1 = self.d_func(T, v[0], phase = phases[0], d = "d_0_0")
    f_2 = self.d_func(T, v[1], phase = phases[1], d = "d_0_0")

    dP_dT = ((e_1 - e_2)*T - P*(v[0] - v[1]))**2*dT**2
    err_add = [['err', hash(tuple(X_add[0])), dP_dT/(v[0] - v[1])**2],
              ['err', hash(tuple(X_add[1])), dP_dT/(v[0] - v[1])**2],
              ['err', hash(tuple(X_add[2])), dP_dT]]
    
    p_1 = self.d_func(T, v[0], phase = phases[0], d = "d_0_1")
    p_2 = self.d_func(T, v[1], phase = phases[1], d = "d_0_1")
    ref = [p_1*T + self.predict([X_add[0]])[0, 0],
          p_2*T + self.predict([X_add[1]])[0, 0],
          (f_1 - f_2)*T + self.predict([X_add[2]])[0, 0]]
    Y_add = [[P + ref[0]],
            [P + ref[1]],
            [P*(v[0] - v[1]) + ref[2]]]
    return X_add, Y_add, err_add

def add_melt(self, phases, melt_p):
    """Adds melting point to GP"""
    v = self._melt_vol(phases, melt_p)
    X_add, Y_add, err_add = self._melt_point(phases, v, melt_p)
    self.add_points(X_add, Y_add, err_add)
