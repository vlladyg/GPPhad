import os
import numpy as np
import pandas as pd
from gmpy2 import mpfr, get_context
get_context().precision=400

from ..cov_dicts import cov_H_1
from ..GP import GP_full, load

def first_stage_N(data):
    X = np.zeros((len(data), 3), dtype = object)
    Y = np.zeros((len(data), 1), dtype = object)
    err = np.zeros((len(data), 3), dtype = object)
    
    X[:, 0] = 'sol'
    X[:, 1] = 'd_0_0'
    X[:, 2] = data[:, 1]
    
    Y[:, 0] = data[:, 2]
    err[:, 0] = 'err'
    for i in range(len(X)):
        err[i, 1] = hash(tuple(X[i]))
    if data.shape[1] == 4:
        err[:, 2] = data[:, 3]**2
    else:
        err[:, 2] = mpfr(0)
    
    th = [1, 100]
    
    GP_1 = GP_full(cov_H_1, th, X, Y, err)
    GP_1.optimize(GP_1.th, method="L-BFGS-B", tol = 1e-4)
    return GP_1



def constr_H(folder):
    if os.path.exists(f"{folder}/GP_H.pickle"):
        return load(f"{folder}/GP_H.pickle")
    
    data = pd.read_csv(f"{folder}/H.dat", header = None, dtype = object)
    data = np.array(data, dtype = object)
    
    data = data.tolist()
    for i in range(len(data)):
        data[i] = list(map(mpfr, data[i]))
    
    data = np.array(data, dtype = object)
    N_list = pd.unique(data[:, 1])
    V_list = pd.unique(data[:, 0])
    assert(len(V_list) == 1)
    GP_H = first_stage_N(data)
        
    GP_H.save(f"{folder}/GP_H.pickle")
    
    return GP_H