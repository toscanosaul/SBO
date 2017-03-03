import numpy as np

def finite_differences(dh, x, f, dim):
    f_ = f(x)
    x2 = x[]
    f2 = f(x+dh)
    return (f2-f)/dh