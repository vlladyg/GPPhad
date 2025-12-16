from gmpy2 import mpfr, get_context
get_context().precision=400

consts = {'Pk': 160.2176621, 'k': 8.617333262*10**(-5)}

# Mean calc
def mean(slice_):
    """Mean value of the slice_"""
    s = list(map(mpfr, slice_))
    return sum(s)/mpfr(len(slice_))

# Std calc
def std(slice_):
    """Computes std from the mean value of the slice_"""
    N = mpfr(len(slice_))
    mean_ = mean(slice_)
    sq = list(map(lambda x: (mpfr(x) - mean_)**2, slice_))
    
    return (sum(sq))**(1/2.)/(N - 1)**(1/2.)/N**(1/2.)


def square_root(result):
    res = 0.0
    for el in result:
        res += el**2

    return res**(1/2.0)

def tolist(X, Y, err):
    return X.tolist(), Y.tolist(), err.tolist()

def print_point(opt, phases, point, point_var):
    if opt == 'pt':
        point_  = [point[0], point[1], point[2]/consts['k']]
        point_var_  = [point_var[0], point_var[1], point_var[2]/consts['k']]
        print("%s volume: %.5f ± %.5f"%(phases[0], point_[0],point_var_[0]))
        print("%s volume: %.5f ± %.5f"%(phases[1], point_[1],point_var_[1]))
        print("pt point temp: %.5f ± %.5f"%(point_[2],point_var_[2]))
    elif opt == 'tp':
        point_  = [point[0], point[1], point[2]*consts['Pk']]
        point_var_  = [point_var[0], point_var[1], point_var[2]*consts['Pk']]
        print("%s volume: %.5f ± %.5f"%(phases[0], point_[0],point_var_[0]))
        print("%s volume: %.5f ± %.5f"%(phases[1], point_[1],point_var_[1]))
        print("tp point pressure: %.5f ± %.5f"%(point_[2],point_var_[2]))
    elif opt == 'triple':
        point_  = [point[0], point[1], point[2], point[3]/consts['k']]
        point_var_  = [point_var[0], point_var[1],
                       point_var[2], point_var[3]/consts['k']]
        print("%s volume: %.5f ± %.5f"%(phases[0], point_[0],point_var_[0]))
        print("%s volume: %.5f ± %.5f"%(phases[1], point_[1],point_var_[1]))
        print("%s volume: %.5f ± %.5f"%(phases[2], point_[2],point_var_[2]))
        print("triple point temp: %.5f ± %.5f"%(point_[3],point_var_[3]))
    elif opt == 'B':
        point_  = [point[0], point[1]*consts['Pk']]
        point_var_  = [point_var[0], point_var[1]*consts['Pk']]
        print("%s volume: %.5f ± %.5f"%(phases[0], point_[0],point_var_[0]))
        print("%s bulk modulus in GPa: %.5f ± %.5f"%(phases[0], point_[1],point_var_[1]))
    elif opt == 'alpha':
        point_  = [point[0], point[1]*consts['k']]
        point_var_  = [point_var[0], point_var[1]*consts['k']]
        print("%s volume: %.5f ± %.5f"%(phases[0], point_[0],point_var_[0]))
        print("%s volumetric thermal expansion in 1/K: %.5f ± %.5f"%(phases[0], point_[1],point_var_[1]))       
    elif opt == 'gamma':
        point_  = [point[0], point[1]]
        point_var_  = [point_var[0], point_var[1]]
        print("%s volume: %.5f ± %.5f"%(phases[0], point_[0],point_var_[0]))
        print("%s grunaisen parameter in arbitary units: %.5f ± %.5f"%(phases[0], point_[1],point_var_[1]))       
    return point_, point_var_