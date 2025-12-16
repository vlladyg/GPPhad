import numpy as np
import gmpy2 as gp
gp.get_context().precision = 400

# reference is calculated via formula
# A + B*E0 | if E0 exists
# A        | otherwise
A = {"liq": {"d_0_0": lambda v: gp.log(v), 
              "d_0_1": lambda v: 1/v,
              "d_0_2": lambda v: -1/v**2,
              "d_0_3": lambda v: 2/v**3,
              "d_0_4": lambda v: -6/v**4},
      "sol": {"d_0_0": lambda t: -1 + 3/2.0*gp.log(2*gp.const_pi(400)) + 3/2.0*gp.log(t), 
               "d_1_0": lambda t: 3/2.0/t,
               "d_2_0": lambda t: -3/2.0/t**2,
               "d_3_0": lambda t: 6/2.0/t**3,
             }
    }

# Valid only for existing E0
B = {"d_0": lambda t: -1/t,
     "d_1": lambda t: 1/t**2,
     "d_2": lambda t: -2/t**3,
     "d_3": lambda t: 6/t**4,
    }

def ref(self, t, v, phase, d):
    """Computes reference for the phase"""
    if phase[:3] == "liq":
        a = A[phase].get(d, lambda v: gp.mpfr(0.0))
        return a(v)
    
    if phase[:3] == "sol":
        if self.S0[phase]:
            a = A["sol"].get(d, lambda t: gp.mpfr(0.0))
            b = B.get(d[:3], lambda t: gp.mpfr(0.0))
            z_E0 = ["sol"] + [f"d_0_{d[4]}"] + [v]
            return a(t) + b(t)*self.S0[phase].predict([z_E0])[0, 0]
            
    return gp.mpfr(0.0) 
    
def d_func(self, t, v, phase = "sol", d = "d_0_0"):
    """Returns differential 'd' of F/T with added reference"""
    z = [phase] + [d] + [t] + [v] + self.x_fixed
    return -np.float64(self.predict([z])[0, 0] + ref(self, t, v, phase, d))
        
#Free energy          
def predict_F(self, t, v, phase = 'sol'):
    """Estimates mean and var of free energy of phase "phase" 
       at given point t, v in eV"""
    F = self.d_func(t, v, phase = phase, d = 'd_0_0')*t
    dF = self.predict([[phase, 'd_0_0', t, v] + self.x_fixed])[0, 1]**(1/2.0)*t
    return np.array([F, dF], dtype = np.float64)
           
def predict_S(self, t, v, phase = 'sol'):
    """Estimates mean and var of entropy energy of phase "phase" 
       at given point t, v in bolzman consts"""
    S = -(t*self.d_func(t, v, phase = phase, d = "d_1_0") + self.d_func(t, v, phase = phase, d = "d_0_0"))
    dS = self.predict([[(t, [phase, 'd_1_0', t, v] + self.x_fixed), (1.0, [phase, 'd_0_0', t, v] + self.x_fixed)]])[0, 1]**(1/2.0)
    return np.array([S, dS], dtype = np.float64)
    
def predict_P(self, t, v, phase = 'sol'):
    """Estimates mean and var of pressure of phase "phase" 
       at given point t, v in eV/A^3"""
    P = -self.d_func(t, v, phase = phase, d = "d_0_1")*t
    dP = self.predict([[phase, 'd_0_1', t, v] + self.x_fixed])[0, 1]**(1/2.0)*t
    return np.array([P, dP], dtype = np.float64)
           
           
def predict_E(self, t, v, phase = 'sol'):
    """Estimates mean and var of potential energy of phase "phase" 
    at given point t, v in eV"""
    E = self.d_func(t, v, phase = phase, d = "d_1_0")*t**2
    dE = self.predict([[phase, 'd_1_0', t, v] + self.x_fixed])[0, 1]**(1/2.0)*t**2
    return np.array([E, dE], dtype = np.float64)

# Functions that estimate only mean (no var)
def predict_B(self, t, v, phase = 'sol'):
    """Estimates mean of bulk modulus of phase "phase" 
    at given point t, v in eV/A^3"""
    return t*v*self.d_func(t, v, phase = phase, d = "d_0_2")

def predict_alpha(self, t, v, phase = 'sol'):
    """Estimates mean of volumetric thermal expansion of phase "phase" 
    at given point t, v in bolzman consts"""
    num = self.d_func(t, v, phase = phase, d = "d_0_1") + self.d_func(t, v, phase = phase, d = "d_1_1")*t
    denom = v*self.d_func(t, v, phase = phase, d = "d_0_2")*t
    return -num/denom

def predict_gamma(self, t, v, phase = 'sol'):
    """Estimates mean of volumetric thermal expansion of phase "phase" 
    at given point t, v in bolzman consts"""
    num = v*(self.d_func(t, v, phase = phase, d = "d_0_1") + self.d_func(t, v, phase = phase, d = "d_1_1")*t)
    denom = 2*t*self.d_func(t, v, phase = phase, d = "d_1_0") + t**2*self.d_func(t, v, phase = phase, d = "d_2_0")
    return num/denom
