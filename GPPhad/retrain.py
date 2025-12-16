from .utils import consts
from .GP import GP_full

import numpy as np

def devide(self):
    """Devides X, Y, err of the GP into parts"""
    bounds = self.bounds; H = self.H; S0 = self.S0; x_fixed = self.x_fixed
    f_dict = self.f_dict; th = self.th; phases = self.phases
    
    X = {}; Y = {}; err = {}
    for phase in phases:
        X[phase] = np.array(self.X, dtype = object)[np.array(self.X, dtype = object)[:, 0] == phase].tolist()
        Y[phase] = self.Y[np.array(self.X, dtype = object)[:, 0] == phase].tolist()
        err[phase] = np.array(self.err, dtype = object)[np.array(self.X, dtype = object)[:, 0] == phase].tolist() 
    return X, Y, err, f_dict, th, phases, S0, H, bounds, x_fixed 


def retrain(self, melt_points, ind_bounds):
    X, Y, err, f_dict, th_temp, phases, S0, H, bounds, x_fixed = devide(self)
    
    for phase in phases:
        GP_ = GP_full(f_dict, th_temp, X[phase], Y[phase], err[phase], S0 = S0,
                      H = H, bounds = bounds, phases = phases, x_fixed = x_fixed)
        res, th_temp = GP_.optimize(th_temp, ind = ind_bounds[phase])
    
    melt_inds = None
    for point in melt_points:
            
        X_cur = X[point[0][0]] + X[point[0][1]]
        Y_cur = Y[point[0][0]] + Y[point[0][1]]
        err_cur = err[point[0][0]] + err[point[0][1]]
        GP_ = GP_full(f_dict, th_temp, X_cur, Y_cur, err_cur, S0 = S0, H = H, 
                      bounds = bounds, phases = phases, x_fixed = x_fixed)
        GP_.add_melt(point[0], point[1])
        
        inds = [ind_bounds[point[0][0]][0], ind_bounds[point[0][1]][0]]
        if isinstance(melt_inds, type(None)):
            melt_inds = inds
        else:
            melt_inds += inds
        res, th_temp = GP_.optimize(th_temp, ind = inds)
    
    melt_inds = list(set(melt_inds))
    X_cur = X[phases[0]]
    Y_cur = Y[phases[0]]
    err_cur = err[phases[0]]
    for i in range(1, len(phases)):
        X_cur += X[phases[i]]
        Y_cur += Y[phases[i]]
        err_cur += err[phases[i]]
    
    GP_ = GP_full(f_dict, th_temp, X_cur, Y_cur, err_cur, S0 = S0, H = H,
                  bounds = bounds, phases = phases, x_fixed = x_fixed)
    for point in melt_points:
        GP_.add_melt(point[0], point[1])
    
    res, th_temp = GP_.optimize(th_temp, ind = melt_inds)
    return GP_, th_temp