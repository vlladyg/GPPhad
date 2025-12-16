import numpy as np
from ._cf import cf 
class kernel:   
    """ 
    The universal shell for the different covariance function
    
    Takes covariance function (cf) for initialization
    
    """
    
    def __init__(self, f_dict, th):
        self.f_dict = f_dict
        self.cf = lambda x1, x2, th: cf(x1, x2, th, f_dict)
        self.th = th
        
    def c_elem(self, el_1, el_2):
        if isinstance(el_1[0], tuple):
            if isinstance(el_2[0], tuple):
                result = 0.0
                for sam_1 in el_1:
                    for sam_2 in el_2:
                        result += sam_1[0]*sam_2[0]*self.cf(sam_1[1], sam_2[1], self.th)
                return result
            else:
                result = 0.0
                for sam_1 in el_1:
                    result += sam_1[0]*self.cf(sam_1[1], el_2, self.th)
                return result
        else:
            if isinstance(el_2[0], tuple):
                result = 0.0
                for sam_2 in el_2:
                    result += sam_2[0]*self.cf(el_1, sam_2[1], self.th)
                return result
            else:
                return self.cf(el_1, el_2, self.th)
    
    def c_col(self, X, el):
        col = np.zeros((len(X), 1), dtype = object)
        
        for i in range(len(X)):
            col[i, 0] = self.c_elem(X[i], el)
        
        return col
    
    def c_matrix(self, X_1, X_2):
        matrix = np.zeros((len(X_1), len(X_2)), dtype = object)
        
        for i in range(len(X_2)):
            matrix[:, i] = self.c_col(X_1, X_2[i])[:, 0]
        
        return matrix
    
    def constr_matrix(self, X):
        """
        Covariance matrix constuctor for given covariance function (cf)
        
        and hyperparameters vector (th)
        
        """
        N = len(X)
        K = np.zeros((N, N), dtype = object)
            
        for i in range(N):
            for j in range(i, N):
                K[i, j] = self.cf(X[i], X[j], self.th)
        
        for i in range(1, N):
            for j in range(i):
                K[i, j] = K[j, i]
                
        return K 
    
    def constr_matrix_melt(self, X):
        """
        Covariance matrix constuctor for given covariance function (cf)
        
        and hyperparameters vector (th)
        
        """
        N = len(X)
        K = np.zeros((N, N), dtype = object)
            
        for i in range(N):
            for j in range(i, N):
                K[i, j] = self.c_elem(X[i], X[j])
        
        for i in range(1, N):
            for j in range(i):
                K[i, j] = K[j, i]        
            
        return K 

    def update_matrix(self, K, X, x_):
        """
        Updates covariance matrix K after additiong of x_
        to the training set
        """
        N = len(K)
        K_new = np.zeros((N + len(x_),N + len(x_)), dtype = object) 
        
        K_new[:N, :N] = K[:, :]
        K_new[N:,N:] = self.constr_matrix_melt(x_)[:,:]
        K_new[N:,:N] = self.c_matrix(X, x_).T[:,:]
        K_new[:N,N:] = self.c_matrix(X, x_)[:,:]
        
        return K_new 