#!/usr/bin/env python

import numpy as np

###Objective function
def g(x,w1,w2):
    val=(w2)/(w1)
    return -(val)*(x**2)-w1

####Function F
def Fvalues(x,w1):
    return -x**2-w1

kernel=GPy.kern.RBF(input_dim=n1+n2, variance=sigma0**2, lengthscale=np.concatenate(((0.5/np.array(alpha1))**(.5),(0.5/np.array(alpha2))**(.5))),ARD=True)+GPy.kern.White(n1+n2,0.1)