#!/usr/bin/env python

from SBO import *


if __name__ == '__main__':
    sigma=[1]
    mu=[0]
    n1=1
    n2=1
    l=50
    a=[-3]
    b=[3]
    w=np.array([1])
    muW2=np.array([0])
    sigmaW2=np.array([1])
    M=1000000
    X=np.loadtxt("oldX.txt",ndmin=2)
    W=np.loadtxt("oldW.txt",ndmin=2)
    Y=np.loadtxt("oldY.txt",ndmin=2)
    T=np.loadtxt("hyperparameters.txt",ndmin=2)
    alpha1=T[0,:]
    alpha2=T[1,:]
    sigma0=T[2,:][0]
    muStart=T[3,:][0]
    alg=SBO(f,alpha1,alpha2,sigma0,sigma,mu,muStart,n1,n2,l,a,b,w,muW2,sigmaW2,M,old=True,Xold=X,Wold=W,yold=Y)