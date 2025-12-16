import numpy as np

from .GP import GP_full
from .GP import load

from .loader import GP_loader
from .retrain import retrain
from .utils import consts, print_point 
from .cov_dicts import cov_real, cov_E0_1, cov_E0_2, cov_H_1

#Construct from *.dat files
def create_from_scratch(cov_dict, th, phases, melt = True,cut = 0, **kwargs):
    """Reads *.dat files in the current folder
        and return X, Y, err suitable for GP"""
    loader = GP_loader(phases, melt)
    loader._init_data(cut = cut)
    loader._init_dataset()
    return GP_full(cov_dict, th, X = loader.X, Y = loader.Y, err = loader.err, phases = phases, S0 = loader.S0, H = loader.H, **kwargs)
