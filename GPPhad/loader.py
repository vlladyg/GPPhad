import subprocess as sp
import pandas as pd
import numpy as np
import os
from gmpy2 import mpfr, get_context
get_context().precision=400

from .utils import *
from .two_stages import constr_E0, constr_H

class GP_loader():
    
    def __init__(self, phases, melt = True):
        self.phases = phases
        self.melt = melt
        
        self.S0 = {}
        self.H = {}
        for phase in phases:
            self.S0[phase] = None
            self.H[phase] = None
    #Read data from the file
    def _init_data(self, folder = "./data", cut = 0):
        """Initialisation of the data"""
        self.data = {}
        self.data_E0 = {}
        self.seeds = {}
        for phase in self.phases:
            sp.run(f"sed '/^#/d' -i {folder}/{phase}/{phase}.dat",shell = True)
            self.data[phase] = pd.read_csv(f"{folder}/{phase}/{phase}.dat", sep = '\s+', 
                                           header = None, skip_blank_lines=True)
            
            self.seeds[phase] = pd.unique(self.data[phase].iloc[:, -1])
            np.random.seed(1)
            np.random.shuffle(self.seeds[phase])
            self.seeds[phase] = self.seeds[phase][cut:]
            self.data[phase] = np.array(self.data[phase], dtype = object)

            if os.path.exists(f"{folder}/{phase}/E0.dat"):
                sp.run(f"sed '/^#/d' -i {folder}/{phase}/E0.dat",shell = True)
                self.data_E0[phase] = pd.read_csv(f"{folder}/{phase}/E0.dat",
                                                  sep =',', header = None, skip_blank_lines=True).values
                self.data_E0[phase] = np.array(self.data_E0[phase], dtype = object)
                
                self.S0[phase] = constr_E0(f"data/{phase}/")
                self.H[phase] = constr_H(f"data/{phase}/")
            else:
                self.data_E0[phase] = None

    #Initalization of X, Y, err
    def _init_dataset(self):
        """Init X, Y, err from data"""
        if not self.melt:
            self.X = {}
            self.Y = {}
            self.err = {}
            for phase in self.phases:
                self.X[phase] = []
                self.Y[phase] = []
                self.err[phase] = []
                for seed in self.seeds[phase]:
                    X_GP, Y_GP, err_GP = self.lmp_to_GP(phase, seed)
                    self.X[phase] += X_GP
                    self.Y[phase] += Y_GP
                    self.err[phase] += err_GP
                self.Y[phase] = np.array(self.Y[phase], dtype = object)
        else:
            self.X = []
            self.Y = []
            self.err = []
            for phase in self.phases:
                for seed in self.seeds[phase]:
                    X_GP, Y_GP, err_GP = self.lmp_to_GP(phase, seed)
                    self.X += X_GP
                    self.Y += Y_GP
                    self.err += err_GP
            self.Y = np.array(self.Y, dtype = object)
    
    # Reads data from the file
    def read_pt(self, phase, seed, treshold = 10000):
        """Reads one point"""

        data = self.data[phase][self.data[phase][:, -1] == seed]

        data = data[data[:, 0] > (treshold + 0.2*np.max(data[:, 0]))]
        
        try:
            assert data.tolist() != []
        except AssertionError:
            return [], [], []

        x = list(map(mpfr, data[0, :].tolist()))
        X = [x[1], x[2], x[3], x[4]]
        Y = [mean(data[:, -4]), mean(data[:, -3])]
        err = [std(data[:,-4]), std(data[:, -3])]

        return X, Y, err
    
    def read_E0(self, phase, seed):
        """Reads one point of E0"""
        data = self.data_E0[phase][self.data_E0[phase][:, -1] == seed][0]
        assert data.tolist() != []
        return [data[-3], data[-2]]
    
    def post_pt(self, X, Y, err, E0, phase):
        """Postprocess one point to fit the format of current GP"""
        T = X[0]
        teta = X[0]*consts['k']

        X_new = [[phase, 'd_1_0', teta, X[1], X[2], X[3]], 
                [phase, 'd_0_1', teta, X[1], X[2], X[3]]]

        if E0 != None:
            Y_new = [[(Y[0] - E0[0])/teta**2 - 3.0/2./teta], [(Y[1] - E0[1])/teta + 1/X[1]]]
        elif phase == "liq":
            Y_new = [[Y[0]/teta**2], [Y[1]/teta]]
        else:
            Y_new = [[Y[0]/teta**2], [Y[1]/teta + 1/X[1]]]

        err_new = [['err', hash(tuple(X_new[0])), err[0]**2/teta**4], ['err', hash(tuple(X_new[1])), err[1]**2/teta**2]]
        return X_new, Y_new, err_new
    
    # Full tranfer from lmp data to GP
    def lmp_to_GP(self, phase, seed):
        """Collects one e_p point"""

        X, Y, err = self.read_pt(phase, seed, treshold = 20000)

        if X != []:
            if os.path.exists(f"./data/{phase}/E0.dat"):
                E0 = self.read_E0(phase, seed)
            else:
                E0 = None
        else:
            return [], [], []

        X_GP, Y_GP, err_GP = self.post_pt(X, Y, err, E0, phase)

        return X_GP, Y_GP, err_GP
    