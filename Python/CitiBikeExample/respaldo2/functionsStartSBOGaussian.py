#!/usr/bin/env python

import numpy as np
from sklearn.gaussian_process import GaussianProcess

###Objective function
def g(x,w1,w2):
    val=(w2)/(w1)
    return -(val)*(x**2)-w1

####Function F
def Fvalues(x,w1):
    return -x**2-w1

######Kernel
n1=1
n2=1
alpha1=[.5]
alpha2=[.5]
sigma0=1
muStart=0

def kernelRBF():
    
kernel=GPy.kern.RBF(input_dim=n1+n2, variance=sigma0**2, lengthscale=np.concatenate(((0.5/np.array(alpha1))**(.5),(0.5/np.array(alpha2))**(.5))),ARD=True)


######Train kernel
n1=5
#####n: number of points to estimate hyperparameters
def train(X,Y,n,noise=):
    



####B(x,X,W) is a function that computes B(x,i). X=X[i,:],W[i,:]. x is a matrix where each row is a point to be evaluated
####Gaussian Kernel, and w1 follows a normal distribution
def B(x,X,W):
    tmp=-(((mu/variance)+2.0*(alpha2)*np.array(W))**2)/(4.0*(-alpha2-(1/(2.0*variance))))
    tmp2=-alpha2*(np.array(W)**2)
    tmp3=-(mu**2)/(2.0*variance)
    tmp=np.exp(tmp+tmp2+tmp3)
    tmp=tmp*(1/(sqrt(2.0*variance)))*(1/(sqrt((1/(2.0*variance))+alpha2)))
    x=np.array(x).reshape((x.size,n1))
    tmp1=variance0*np.exp(np.sum(-alpha1*((np.array(x)-np.array(X))**2),axis=1))
    return np.prod(tmp)*tmp1



