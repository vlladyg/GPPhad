import numpy as np
from scipy.optimize import fsolve

def inf_to_fin(t, bound):
    return bound[1]*(1 + t**2)/(bound[1]/bound[0] + t**2)

def fin_to_inf(t, bound):
    return np.sqrt((bound[1] - t*bound[1]/bound[0])/(t - bound[1]))

def pt_func(self, t, x,phases, bounds):

    v1 = inf_to_fin(t[0], bounds[0]); v2 = inf_to_fin(t[1], bounds[1])
    t0 = inf_to_fin(t[2], bounds[2]); T = t0

    res_11 = self.d_func(t0, v1, phase = phases[0], d = "d_0_1")
    res_12 = self.d_func(t0, v2, phase = phases[1], d = "d_0_1")

    res_21 = self.d_func(t0, v1, phase = phases[0], d = "d_0_0")
    res_22 = self.d_func(t0, v2, phase = phases[1], d = "d_0_0")
    res = np.array([res_11*T + x,res_12*T+x, res_21 - res_22 + x/T*(v1 - v2)],
                   dtype = np.float64)
    print(res)
    return res

def tp_func(self, t, x,phases, bounds):

    v1 = inf_to_fin(t[0], bounds[0]); v2 = inf_to_fin(t[1], bounds[1])

    T = x
    res_11 = self.d_func(T, v1, phase = phases[0], d = "d_0_1")
    res_12 = self.d_func(T, v2, phase = phases[1], d = "d_0_1")

    res_21 = self.d_func(T, v1, phase = phases[0], d = "d_0_0")
    res_22 = self.d_func(T, v2, phase = phases[1], d = "d_0_0")
    
    res = np.array([res_11 - res_12, res_21 - res_22 - res_11*(v1 - v2)],
                   dtype = np.float64)
    print(res)
    return res

def triple_func(self, t, phases, bounds):

    v1 = inf_to_fin(t[0], bounds[0]); v2 = inf_to_fin(t[1], bounds[1])
    v3 = inf_to_fin(t[2], bounds[2]); t0 = inf_to_fin(t[3], bounds[3])

    T = t0

    res_11 = self.d_func(t0, v1, phase = phases[0], d = "d_0_1")
    res_12 = self.d_func(t0, v2, phase = phases[1], d = "d_0_1")
    res_13 = self.d_func(t0, v3, phase = phases[2], d = "d_0_1")

    res_21 = self.d_func(t0, v1, phase = phases[0], d = "d_0_0")
    res_22 = self.d_func(t0, v2, phase = phases[1], d = "d_0_0")
    res_23 = self.d_func(t0, v3, phase = phases[2], d = "d_0_0")

    res = np.array([res_11 - res_12, res_21 - res_22 - res_11*v1 + res_12*v2,
                    res_12 - res_13, res_22 - res_23 - res_12*v2 + res_13*v3],
                   dtype = np.float64)
    print(res)
    return res

def v_func(self, t, x, phases, bounds):

    v1 = inf_to_fin(t[0], bounds[0])

    T = x[0]
    P = x[1]
    res_1 = self.d_func(T, v1, phase = phases[0], d = "d_0_1")

    res = np.array([res_1 + P/T],
                   dtype = np.float64)
    print(res)
    return res

def compute_param(self, opt, t, v, phase):
    assert(opt in ["B", "alpha", "gamma"])
    if opt == "B":
        return self.predict_B(t, v, phase)
    if opt == "alpha":
        return self.predict_alpha(t, v, phase)
    if opt == "gamma":
        return self.predict_gamma(t, v, phase)
    
def compute_mean(self, opt, phases, bounds, **kwargs):
    assert(opt in ['pt', 'tp', 'triple', "v", "B", "alpha", "gamma"])
    if opt == 'pt':
        func_opt = lambda t: pt_func(self, t, kwargs['P'], phases, bounds)
    elif opt == 'tp':
        func_opt = lambda t: tp_func(self, t, kwargs['T'], phases, bounds)
    elif opt == 'triple':
        func_opt = lambda t: triple_func(self, t, phases, bounds)
    else:
        func_opt = lambda t: v_func(self, t, [kwargs['T'], kwargs['P']], phases, bounds)

    v_0 = []
    for bound in bounds:
        v_0.append(fin_to_inf(np.mean(bound), bound))
    v_0 = np.array(v_0)

    sol = fsolve(func_opt, v_0)
    
    if opt in ['pt', 'tp', 'triple', "v"]:
        y = np.zeros(len(bounds), dtype = object)
        for i, bound in enumerate(bounds):
            y[i] = inf_to_fin(sol[i], bound)
    else:
        y = np.zeros(2, dtype = object)
        y[0] = inf_to_fin(sol[0], bounds[0])
        y[1] = compute_param(self, opt, kwargs['T'], y[0], phases[0])
    return y
