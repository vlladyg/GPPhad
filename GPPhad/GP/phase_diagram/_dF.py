import numpy as np
from mpinv import fast_mp_matrix_inverse as f_mp_inv_ns

def _pt_dF(self, phases, t, v):
    T = t
    dF = np.zeros((3,3), dtype = object)
    
    dF[0,0] = self.d_func(t, v[0], phase = phases[0], d = "d_0_2")*T
    dF[0,1] = 0.0
    dF[0,2] = self.d_func(t, v[0], phase = phases[0], d = "d_1_1")*T + self.d_func(t, v[0], phase = phases[0], d = "d_0_1")

    dF[1,0] = self.d_func(t, v[0], phase = phases[0], d = "d_0_2")
    dF[1,1] = -self.d_func(t, v[1], phase = phases[1], d = "d_0_2")
    dF[1,2] = self.d_func(t, v[0], phase = phases[0], d = "d_1_1") - self.d_func(t, v[1], phase = phases[1], d = "d_1_1") 

    dF[2,0] = -self.d_func(t, v[0], phase = phases[0], d = "d_0_2")*v[0]
    dF[2,1] = self.d_func(t, v[1], phase = phases[1], d = "d_0_2")*v[1]
    dF[2,2] = self.d_func(t, v[0], phase = phases[0], d = "d_1_0") -self.d_func(t, v[1], phase = phases[1], d = "d_1_0") - self.d_func(t, v[0], phase = phases[0], d = "d_1_1")*v[0] + self.d_func(t, v[1], phase = phases[1], d = "d_1_1")*v[1]
    
    
    return dF


def _tp_dF(self, phases, t, v):
    T = t
    dF = np.zeros((2,2), dtype = object)
    
    dF[0,0] = self.d_func(t, v[0], phase = phases[0], d = "d_0_2")
    dF[0,1] = -self.d_func(t, v[1], phase = phases[1], d = "d_0_2")
    dF[1,0] = -self.d_func(t, v[0], phase = phases[0], d = "d_0_2")*v[0]
    dF[1,1] = self.d_func(t, v[1], phase = phases[1], d = "d_0_2")*v[1]
    
    return dF

def _P_dF(self, phases, t, v):
    T = t
    dF = np.zeros((2,2), dtype = object)
        
    dF[0,0] = self.d_func(t, v[0], phase = phases[0], d = "d_0_2")*T
    dF[0,1] = 0.0

    dF[1,0] = 0.0
    dF[1,1] = self.d_func(t, v[1], phase = phases[1], d = "d_0_2")*T
    
    return dF


def _triple_dF(self, phases, t, v):
    T = t
    dF = np.zeros((4,4), dtype = object)

    dF[0,0] = -self.d_func(t, v[0], phase = phases[0], d = "d_0_2")
    dF[0,1] = self.d_func(t, v[1], phase = phases[1], d = "d_0_2")
    dF[0,2] = 0.0
    dF[0,3] = -self.d_func(t, v[0], phase = phases[0], d = "d_1_1") + self.d_func(t, v[1], phase = phases[1], d = "d_1_1")

    dF[1,0] = self.d_func(t, v[0], phase = phases[0], d = "d_0_2")*v[0]
    dF[1,1] = -self.d_func(t, v[1], phase = phases[1], d = "d_0_2")*v[1]
    dF[1,2] = 0.0
    dF[1,3] = -self.d_func(t, v[0], phase = phases[0], d = "d_1_0") + self.d_func(t, v[1], phase = phases[1], d = "d_1_0") + self.d_func(t, v[0], phase = phases[0], d = "d_1_1")*v[0] -self.d_func(t, v[1], phase = phases[1], d = "d_1_1")*v[1]

    dF[2,0] = 0.0
    dF[2,1] = -self.d_func(t, v[1], phase = phases[1], d = "d_0_2")
    dF[2,2] = self.d_func(t, v[2], phase = phases[2], d = "d_0_2")
    dF[2,3] = -self.d_func(t, v[1], phase = phases[1], d = "d_1_1") + self.d_func(t, v[2], phase = phases[2], d = "d_1_1")

    dF[3,0] = 0.0
    dF[3,1] = self.d_func(t, v[1], phase = phases[1], d = "d_0_2")*v[1]
    dF[3,2] = -self.d_func(t, v[2], phase = phases[2], d = "d_0_2")*v[2]
    dF[3,3] = -self.d_func(t, v[1], phase = phases[1], d = "d_1_0") + self.d_func(t, v[2], phase = phases[2], d = "d_1_0") + self.d_func(t, v[1], phase = phases[1], d = "d_1_1")*v[1] - self.d_func(t, v[2], phase = phases[2], d = "d_1_1")*v[2]

    return dF


#Thermodynamic functions
def _B_dF(self, phases, t, v):
    B = v[1]
    
    dF = np.zeros((2,2), dtype = object)
    
    dF[0,0] = t*self.d_func(t, v[0], phase = phases[0], d = "d_0_2")
    dF[0,1] = 0
    dF[1,0] = v[0]*t*self.d_func(t, v[0], phase = phases[0], d = "d_0_3")
    dF[1,1] = -1
    
    return dF

def _alpha_dF(self, phases, t, v):
    alpha = v[1]
    
    dF = np.zeros((2,2), dtype = object)
    
    dF[0,0] = t*self.d_func(t, v[0], phase = phases[0], d = "d_0_2")
    dF[0,1] = 0
    dF[1,0] = alpha*t*self.d_func(t, v[0], phase = phases[0], d = "d_0_2") + alpha*v[0]*t*self.d_func(t, v[0], phase = phases[0], d = "d_0_3") + self.d_func(t, v[0], phase = phases[0], d = "d_0_2") + t*self.d_func(t, v[0], phase = phases[0], d = "d_1_2")
    dF[1,1] = v[0]*t*self.d_func(t, v[0], phase = phases[0], d = "d_0_2")
    
    return dF

def _gamma_dF(self, phases, t, v):
    gamma = v[1]
    
    dF = np.zeros((2,2), dtype = object)
    
    dF[0,0] = t*self.d_func(t, v[0], phase = phases[0], d = "d_0_2")
    dF[0,1] = 0
    dF[1,0] = gamma*(2*t*self.d_func(t, v[0], phase = phases[0], d = "d_1_1") + t**2*self.d_func(t, v[0], phase = phases[0], d = "d_2_1")) - self.d_func(t, v[0], phase = phases[0], d = "d_0_1") - t*self.d_func(t, v[0], phase = phases[0], d = "d_1_1") - v[0]*self.d_func(t, v[0], phase = phases[0], d = "d_0_2") - v[0]*self.d_func(t, v[0], phase = phases[0], d = "d_1_2")
    dF[1,1] = 2*t*self.d_func(t, v[0], phase = phases[0], d = "d_1_0") + t**2*self.d_func(t, v[0], phase = phases[0], d = "d_2_0")
    
    return dF
def _dF(self, opt, phases, t, v):
    """General engine for defining Jacobian of system 
    of equations and its inverse"""
    if opt == 'pt':
        dF = _pt_dF(self, phases, t, v)
    elif opt == 'tp':
        dF = _tp_dF(self, phases, t, v)
    elif opt == 'triple':
        dF = _triple_dF(self, phases, t, v)
    elif opt == 'B':
        dF = _B_dF(self, phases, t, v)
    elif opt == 'alpha':
        dF = _alpha_dF(self, phases, t, v)
    elif opt == 'gamma':
        dF = _gamma_dF(self, phases, t, v)
        
    dF_inv = f_mp_inv_ns(dF)
    res = []
    for i in range(dF_inv.shape[0]):
        res.append(np.zeros((1, dF_inv.shape[1]), dtype = object))
        res[i][0, :] = dF_inv[i, :]
    
    if opt != 'tp':
        return dF, res
    else:
        return _P_dF(self, phases, t, v), res
