
def constr_X_pt(self, phases, t, v):
    return [[(t, [phases[0], "d_0_1", t, v[0]] + self.x_fixed)],
         [(1, [phases[0], "d_0_1", t, v[0]] + self.x_fixed), 
          (-1, [phases[1], "d_0_1", t, v[1]] + self.x_fixed)],
        [(1, [phases[0], "d_0_0", t, v[0]] + self.x_fixed),
         (-1, [phases[1], "d_0_0", t, v[1]] + self.x_fixed),
        (-v[0], [phases[0], "d_0_1", t, v[0]] + self.x_fixed),
         (v[1], [phases[1], "d_0_1", t, v[1]] + self.x_fixed)]]


def constr_X_tp(self, phases, t, v):
    return [[(-1, [phases[0], "d_0_1", t, v[0]] + self.x_fixed), 
          (1, [phases[1], "d_0_1", t, v[1]] + self.x_fixed)],
        [(v[0], [phases[0], "d_0_1", t, v[0]] + self.x_fixed), 
          (-v[1], [phases[1], "d_0_1", t, v[1]] + self.x_fixed)]]

def constr_X_triple(self, phases, t, v):
    return [[(1, [phases[0], "d_0_1", t, v[0]] + self.x_fixed),
             (-1, [phases[1], "d_0_1", t, v[1]] + self.x_fixed)],
         [(1, [phases[0], "d_0_0", t, v[0]] + self.x_fixed),
          (-1, [phases[1], "d_0_0", t, v[1]] + self.x_fixed),
        (-v[0], [phases[0], "d_0_1", t, v[0]] + self.x_fixed), 
          (v[1], [phases[1], "d_0_1", t, v[1]] + self.x_fixed)],
         [(1, [phases[1], "d_0_1", t, v[1]] + self.x_fixed), 
          (-1, [phases[2], "d_0_1", t, v[2]] + self.x_fixed)],
        [(1, [phases[1], "d_0_0", t, v[1]] + self.x_fixed),
         (-1, [phases[2], "d_0_0", t, v[2]] + self.x_fixed),
        (-v[1], [phases[1], "d_0_1", t, v[1]] + self.x_fixed),
         (v[2], [phases[2], "d_0_1", t, v[2]] + self.x_fixed)]]

def constr_X_B(self, phases, t, v):
    B = v[1]
    return [[(-t, [phases[0], "d_0_1", t, v[0]] + self.x_fixed)],
        [(-v[0]*t, [phases[0], "d_0_2", t, v[0]] + self.x_fixed)]]

def constr_X_alpha(self, phases, t, v):
    alpha = v[1]
    return [[(-t, [phases[0], "d_0_1", t, v[0]] + self.x_fixed)],
        [(-alpha*v[0]*t, [phases[0], "d_0_2", t, v[0]] + self.x_fixed), 
          (-1, [phases[0], "d_0_1", t, v[0]] + self.x_fixed), 
         (-t, [phases[0], "d_1_1", t, v[0]] + self.x_fixed)]]

def constr_X_gamma(self, phases, t, v):
    gamma = v[1]
    return [[(-t, [phases[0], "d_0_1", t, v[0]] + self.x_fixed)],
        [(-gamma*2*t, [phases[0], "d_1_0", t, v[0]] + self.x_fixed), 
          (-gamma*t**2, [phases[0], "d_2_0", t, v[0]] + self.x_fixed), 
         (v[0], [phases[0], "d_0_1", t, v[0]] + self.x_fixed),
        (v[0]*t, [phases[0], "d_1_1", t, v[0]] + self.x_fixed)]]

def _constr_X(self, opt, phases, t, v):
    
    if opt == 'pt':
        return constr_X_pt(self, phases, t, v)
    elif opt == 'tp':
        return constr_X_tp(self, phases, t, v)
    elif opt == 'triple':
        return constr_X_triple(self, phases, t, v)
    elif opt == 'B':
        return constr_X_B(self, phases, t, v)
    elif opt == 'alpha':
        return constr_X_alpha(self, phases, t, v)
    elif opt == 'gamma':
        return constr_X_gamma(self, phases, t, v)
    
def constr_X_s0_pt(phases, t, v):
    return { phases[0]: [[(-1, ['sol', "d_0_1", v[0]])], [(-1/t, ['sol', "d_0_1", v[0]])],
                       [(-1/t, ['sol', "d_0_0", v[0]]), (v[0]/t, ['sol', "d_0_1", v[0]])]],
             phases[1]: [[(0, ['sol', "d_0_1", v[1]])], [(1/t, ['sol', "d_0_1", v[1]])],
                         [(1/t, ['sol', "d_0_0", v[1]]), (-v[1]/t, ['sol', "d_0_1", v[1]])]]
           }


def constr_X_s0_tp(phases, t, v):
    return { phases[0]: [[(1, ['sol', "d_0_1", v[0]])], [(-v[0], ['sol', "d_0_1", v[0]])]],
             phases[1]: [[(-1, ['sol', "d_0_1", v[1]])], [(v[1], ['sol', "d_0_1", v[1]])]]
           }


def constr_X_s0_triple(phases, t, v):
    return { phases[0]: [[(-1/t, ['sol', "d_0_1", v[0]])],
                         [(-1/t, ['sol', "d_0_0", v[0]]), (v[0]/t, ['sol', "d_0_1", v[0]])],
                         [(0.0, ['sol', "d_0_1", v[0]])], [(0.0, ['sol', "d_0_1", v[0]])]],
             phases[1]: [[(1/t, ['sol', "d_0_1", v[1]])], 
                         [(1/t, ['sol', "d_0_0", v[1]]), (-v[1]/t, ['sol', "d_0_1", v[1]])], 
                    [(-1/t, ['sol', "d_0_1", v[1]])], 
                         [(-1/t, ['sol', "d_0_0", v[1]]), (v[1]/t, ['sol', "d_0_1", v[1]])]],
             phases[2]: [[(0.0, ['sol', "d_0_1", v[2]])], [(0.0, ['sol', "d_0_1", v[2]])],
                        [(1/t, ['sol', "d_0_1", v[2]])], [(1/t, ['sol', "d_0_0", v[2]]),
                                                      (-v[2]/t, ['sol', "d_0_1", v[2]])]]
           }

def constr_X_s0_B(phases, t, v):
    return { phases[0]: [[(1, ['sol', "d_0_1", v[0]])], 
                         [(v[0], ['sol', "d_0_2", v[0]])]]}

def constr_X_s0_alpha(phases, t, v):
    alpha = v[1]
    return { phases[0]: [[(1, ['sol', "d_0_1", v[0]])], 
                         [(v[0]*alpha, ['sol', "d_0_2", v[0]]), (1/t, ['sol', "d_0_1", v[0]]), (-1/t, ['sol', "d_0_1", v[0]])]]}


def constr_X_s0_gamma(phases, t, v):
    gamma = v[1]
    return { phases[0]: [[(1, ['sol', "d_0_1", v[0]])], 
                         [(-2*gamma/t, ['sol', "d_0_0", v[0]]), 
                          (gamma*2/t, ['sol', "d_0_0", v[0]]),
                          (v[0], ['sol', "d_0_1", v[0]]),
                         (-v[0]/t, ['sol', "d_0_1", v[0]])]]}

def _constr_X_s0(opt, phases, t, v):
    
    if opt == 'pt':
        return constr_X_s0_pt(phases, t, v)
    elif opt == 'tp':
        return constr_X_s0_tp(phases, t, v)
    elif opt == 'triple':
        return constr_X_s0_triple(phases, t, v)
    elif opt == 'B':
        return constr_X_s0_B(phases, t, v)
    elif opt == 'alpha':
        return constr_X_s0_alpha(phases, t, v)
    elif opt == 'gamma':
        return constr_X_s0_gamma(phases, t, v)
