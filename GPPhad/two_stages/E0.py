import pandas as pd
import numpy as np
import os
from gmpy2 import mpfr, get_context
get_context().precision=400

from ..cov_dicts import cov_E0_1, cov_E0_2 
from ..GP import GP_full, load

def first_stage(data, ind = 2):
    X = np.zeros((len(data), 3), dtype = object)
    Y = np.zeros((len(data), 1), dtype = object)
    err = np.zeros((len(data), 2), dtype = object)
    
    X[:, 0] = 'sol'
    X[:, 1] = 'd_0_0'
    X[:, 2] = data[:, 1]
    
    Y[:, 0] = data[:, ind]
    err[:, 0] = 'err'
    for i in range(len(X)):
        err[i, 1] = hash(tuple(X[i]))
        err[i, 2] = mpfr(0)
        
    th = [3., 1, 5.]
    
    GP_1 = GP_full(cov_E0_1, th, X, Y, err)
    GP_1.optimize([1.,],[1., 5], method="L-BFGS-B", tol = 1e-4)
    return GP_1

def second_stage(data):
    X = np.zeros((len(data)*2, 3), dtype = object)
    Y = np.zeros((len(data)*2, 1), dtype = object)
    err = np.zeros((len(data)*2, 4), dtype = object)
    
    X[:, 0] = 'sol'
    X[::2, 1] = 'd_0_0'
    X[1::2, 1] = 'd_0_1'
    X[::2, 2] = data[:, 0]
    X[1::2, 2] = data[:, 0]
    
    Y[::2, 0] = data[:, 2]
    Y[1::2, 0] = -data[:, 3]
    
    err[:, 0] = 'err'
    if len(data[0]) == 6:
        err[::2, 2] = data[:, 5]**2
        err[1::2, 2] = data[:, 6]**2
    else:
        err[:, 2] = mpfr(1e-10)
    for i in range(len(X)):
        err[i, 1] = hash(tuple(X[i]))
        
    th = [-4.000000054159204, 1.001932834519879, -0.0038950240557817923, -4]
    
    GP_2 = GP_full(cov_E0_2, th, X, Y, err)
    GP_2.optimize(GP_2.th, method="L-BFGS-B", tol = 1e-12)
    return GP_2
    
    
def two_stages(data, V_list):
    data = []
    for V in V_list:
        data[data[:, 0] == V][::2]
        GP_1 = first_stage(data[data[:, 0] == V])
        line_1 = GP_1.predict([[10**10]])[0].tolist()
        GP_1 = first_stage(data[data[1:, 0] == V], ind = 3)
        line_2 = GP_1.predict([[10**10]])[0].tolist()
        
        data.append([V, 10**10] + [line_1[0]] + [line_2[0]] + [line_1[1]**(1/2.)] + [line_2[1]**(1/2.)])
    
    data_2 = np.array(data_2, dtype = object)
    return second_stage(data_2)

def constr_E0(folder):
    if os.path.exists(f"{folder}/GP_E0.pickle"):
        return load(f"{folder}/GP_E0.pickle")

    data = pd.read_csv(f"{folder}/E0_ref.dat", header = None, dtype = object)
    data = np.array(data, dtype = object)
    
    data = data.tolist()
    for i in range(len(data)):
        data[i] = list(map(mpfr, data[i]))
    
    data = np.array(data, dtype = object)
    c_list = pd.unique(data[:, 1])
    V_list = pd.unique(data[:, 0])
    if len(c_list) == len(V_list) and len(V_list) == 1:
        return data[0][2]
    elif len(c_list) == 1 and len(V_list) != 1:
        GP_E0 = second_stage(data)
    else:
        GP_E0 = two_stages(data, V_list)

    GP_E0.save(f"{folder}/GP_E0.pickle")
    return GP_E0