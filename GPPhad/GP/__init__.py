#External modules
import numpy as np
import gmpy2 as gp
import dill as pickle
from copy import deepcopy as dc
import os
gp.get_context().precision = 400

#Local modules
from ._kernel import kernel

class GP_full(kernel):
    """Multiphase Gaussian process main class"""
    from ._constr import _constr_melt, _constr_not_melt, _constr
    
    # Initialization
    def __init__(self, cov_dict, th = None, X = None, Y = None, err = None, phases = None, **kwargs):
        """INPUT: 
        
            train data (X), 
        
            train labels (Y), 
        
            vector of hyperparameters th,
            
            train error (err); by default set to 0 matrix,
        
            covariance function (cf),
            
            zero temperature entropy S0,
            
            bounds where phase is stable (in volume units)
        
        FORMATS:
        
            X - list with size (N, m)}
        
            Y - list with size (N, 1)
        
            err - list with size (N, 1)
            
            th - dict {phase: th[phase]}
            
            S0 - single phase Gaussian process
            
            bounds - dict {phase : bounds[phase]} 
        
        """
        
        ########################################
        super().__init__(cov_dict, th)
        
        # Initialization of main arrays
        self.X, self.Y, self.err = dc(X), dc(np.array(Y, dtype = object)), dc(err)
        # Checking if phases datasets are independent 
        
        if isinstance(self.X, dict):
            self.phases = self.X.keys()
            self.melt = False
        else:
            self.melt = True
            self.phases = phases
        # assigning each phase covariance matrix
        self._constr()
        ########################################
        if 'x_fixed' in kwargs.keys():
            self.x_fixed = kwargs['x_fixed']
        if 'S0' in kwargs.keys():
            self.S0 = kwargs['S0']
        if 'H' in kwargs.keys():
            self.H = kwargs['H']
        if 'bounds' in kwargs.keys():
            self.bounds = kwargs['bounds']
        if 'cluster' in kwargs.keys():
            self.cluster = kwargs['cluster']
        else:
            self.cluster = {'wd': os.getcwd()}
        
        # engine for holding previous calculations
        self.prev = {}
        self.crit_t = 0.0
        self.crit_v = 0.0
        
    # Prediction on the test set
    def predict(self, X_test):
        """
        Function for Gaussian process regression
    
        INPUT:
    
            X_test - list of test points with size (L, m);
    
            L - number of points for prediction
    
        OUTPUT:
    
            predict - mp matrix with size (L, 2),
    
            where 
    
            predict[i, 0] - the MEAN of Gaussian process for the given point
   
            predict[i, 1] - the VARIANCE of Gaussian process for the given point
            
        """
        L = len(X_test)
        predict = np.zeros((L, 2), dtype = object)
        
        for i in range(L):
            k = self.c_col(self.X, X_test[i])        

            #Prediction and Covariation
            val = (k.T@self.K_inv@self.Y)
            predict[i, 0] = val[0, 0]
            predict[i, 1] = self.c_elem(X_test[i], X_test[i]) - (k.T@self.K_inv@k)[0, 0]
            
        return predict
    
    from ._optimize import marg_like, optimize
    from ._func import d_func
    from ._func import predict_F, predict_S, predict_P, predict_E
    from ._func import predict_B, predict_alpha, predict_gamma
    from ._save_add import save, add_points, add_H
    from ._save_add import _melt_point, _melt_vol, add_melt
    from .phase_diagram import compute_mean, _dF, _cov, compute_var, ad_step
    
    
#load GP from file
def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    
    add_opt = {}
    if 'bounds' in data.keys():
        add_opt['bounds'] = data['bounds']
    if 'cluster' in data.keys():
        add_opt['cluster'] = data['cluster']
    if 'S0' in data.keys():
        add_opt['S0'] = data['S0']
    if 'H' in data.keys():
        add_opt['H'] = data['H']
    if 'x_fixed' in data.keys():
        add_opt['x_fixed'] = data['x_fixed']
        
    return GP_full(cov_dict = data['f_dict'], th = data['th'], X = data['X'], 
                  Y = data['Y'], err = data['err'], phases = data['phases'], **add_opt)
