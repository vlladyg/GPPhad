import gmpy2 as gp
gp.get_context().precision=400

def cf(x1_, x2_, th_, f_dict):
    if x1_[0] != "err" and x2_[0] != "err":
        # If not error
        if x1_[0] == x2_[0]:
            func = f_dict[x1_[0]].get(x1_[1] + '_' + x2_[1], None)
            
            if func:
                return func(x1_[2:], x2_[2:], th_)
            else:
                return f_dict[x1_[0]][x2_[1] + '_' + x1_[1]](x2_[2:], x1_[2:], th_)
        else:
            return gp.mpfr(0.0)
        
    # If error
    elif x1_[0] == "err" and x2_[0] == "err":
            if x1_[1] == x2_[1]:
                return x1_[2]
            else:
                return gp.mpfr(0.0)