import numpy as np
import copy
import dill as pickle
import gmpy2 as gp
from gmpy2 import mpfr, get_context
get_context().precision=400

from .var import compute_var
from ...utils import consts, print_point, square_root

def one_rank_apd(self, x, K_inv, X):
    N = len(K_inv)    

    k = np.zeros((N, 1), dtype = object)
 
    K_inv_new = np.zeros((N+1, N+1), dtype = object)       

    k = self.c_col(X, x)

    d = self.c_elem(x, x)
    g = (d - k.T@K_inv@k)[0 , 0]**(-1)

    K_inv_new[N, N] = g
    K_inv_new[:N, :N] = K_inv@(np.eye(N, dtype = object) + k@k.T@K_inv.T*g)

    K_inv_new[N, :N] = -(K_inv@k*g).T
    K_inv_new[:N, N] = (-K_inv@k*g)[:, 0]   

    X_new = copy.deepcopy(X)
    X_new.append(x)

    return K_inv_new, X_new


def ad_step(self, opt, net, phases, **kwargs):
        """Function for adaptive search of the calculation points
        for best decreasing of the triple point variation
        """
        # Previous value of goal function
        point_old, var_old, eq_err = self.compute_var(opt, phases, 
                                                      bounds = kwargs['bounds'])
        
        VAR = square_root(var_old)
        point_, point_var_ = print_point(opt, phases, point_old, var_old)

        best_score = -mpfr('inf')
        best_ind = 0
        score_net = []
        
        for i in range(len(net)//2):
            # Calculation of the new inverse function
            x = net[i*2:i*2+2]
                
            K_inv_new, X_new = one_rank_apd(self, x[0], self.K_inv, self.X)
            K_inv_new, X_new = one_rank_apd(self, x[1], K_inv_new, X_new)

            point_old, var_new, eq_err = self.compute_var(opt, phases, point = point_old, adapt = {"K_inv": K_inv_new, "x_ad": x},**kwargs)
            VAR_new = square_root(var_new)
            
            # Calcution of the information function value for the given point
            if "w" not in kwargs:    
                score_temp = -gp.log(VAR_new/VAR)
            else:
                score_temp = -kwargs['w'](x_[3], x_[4])*gp.log(Var_new/Var_old)
            score_net.append(score_temp)        
    
            if score_temp > best_score:
                    best_score = score_temp
                    best_ind = i
                    
        # Track of the procesed points
        if 'it' in kwargs.keys():
            out = [(point_, 'point'), (point_var_, 'point_var'),
                   ([net[best_ind*2], net[best_ind*2+1]], 'x_new'),
                   (best_score, 'best score'), (score_net, 'score net')]
            with open(f'iter_{kwargs["it"]}.pickle', 'wb') as f:
                pickle.dump(out, f)
        
        return best_ind, best_score, VAR