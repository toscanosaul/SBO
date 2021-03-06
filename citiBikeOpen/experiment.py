#!/usr/bin/env python
import sys
sys.path.append("..")
import numpy as np
from math import *
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm,poisson
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson
import json
#from BGO.Source import *
import time
from AffineBreakPoints import *

nTemp=int(sys.argv[1])
randomSeed=nTemp
np.random.seed(randomSeed)

###regression

beta=np.array([0.1,1.0])
sigma=3.0
K=2 #number of covariates
a=[[1,2],[3,4]] #values of covariates
nRoutes=2
def f(i):
    mu=np.dot(a[i],beta)
    sigma=3.0
    res=np.random.normal(mu, sigma, 1)
    return res

def newY(x):
    N=len(x)
    res=0
    count=0
    for i in range(N):
        if x[i]==1:
            count+=1
            res+=f(i)
            
    return res,count*(sigma**2)


###prior
alpha0=[0.15,1.5]
beta0=np.zeros((nRoutes,nRoutes))

beta0[0,0]=1.0
beta0[1,1]=1.0

#####


def a_0(x):
    n=len(x)
    z=0
    for i in range(n):
        if x[i]==1:
            z+=np.dot(alpha0,a[i])
            
    return z



def tau0(e,f):
    z1=a[e]
    z2=a[f]
    res=np.dot(z1,beta0)
    res=np.dot(res,z2)
    return res

def sigma_0(x,z):
    res=0
    count=0
    for i in range(nRoutes):
        for j in range(nRoutes):
            if x[i]==1 and z[j]==1:
    
                res+=tau0(i,j)
                
    return res

import itertools
totalOpenings=2
possiblePoints2 = list(itertools.product([0, 1], repeat=2))
possiblePoints=[i for i in possiblePoints2 if np.sum(i)<=totalOpenings]

aVec=np.zeros(len(possiblePoints))

i=0
for x in possiblePoints:
    aVec[i]=a_0(x)
    i+=1

def bfunc(x):
    z=np.zeros(len(possiblePoints))
    ind=0
    for j in possiblePoints:
        z[ind]=sigma_0(x,j)
        ind+=1

    div=np.sqrt(sigma**2+(sigma_0(x,x)))
    z=z/div
    return z

def hvoi (b,c,keep):
    M=len(keep)
    if M>1:
        c=c[keep+1]
        c2=-np.abs(c[0:M-1])
        tmp=norm.pdf(c2)+c2*norm.cdf(c2) 
        return np.sum(np.diff(b[keep])*tmp)
    else:
        return 0
    


def VOI(x,a=aVec,grad=False):
    bVec=bfunc(x)
    a,b,keep=AffineBreakPointsPrep(a,bVec)
    keep1,c=AffineBreakPoints(a,b)
    keep1=keep1.astype(np.int64)
    M=len(keep1)
    keep2=keep[keep1]
    
    return hvoi(b,c,keep1)+a_0(x)

x=np.array([1,0])

VOIval=np.zeros(len(possiblePoints))

Npoint=len(possiblePoints)
for i in range(Npoint):
    VOIval[i]=VOI(possiblePoints[i])

print "first"
print possiblePoints[np.argmax(VOIval)]
oldPoint=possiblePoints[np.argmax(VOIval)]

newEval,varP=newY(oldPoint)
oldA0=a_0(oldPoint)
oldA=sigma_0(oldPoint,oldPoint)
print newEval
###optimize a_1

def mu1(z,old=oldPoint,newEval=newEval,var=varP,a02=oldA0,A=oldA):
    res=a_0(z)
    res2=sigma_0(old,z)
    res3=res2*(1.0/A)*(newEval-a02)
    return res+res3
    

def a_1(x):
    n=len(x)
    z=0
    for i in range(n):
        if x[i]==1:
            z+=np.dot(alpha0,a[i])
            
    return z
possiblePoints=[i for i in possiblePoints2 if np.sum(i)==totalOpenings]
Aval=np.zeros(len(possiblePoints))



Npoint=len(possiblePoints)
A0val=np.zeros(len(possiblePoints))
for i in range(Npoint):
    Aval[i]=a_1(possiblePoints[i])
    A0val[i]=a_0(possiblePoints[i])

newSol=possiblePoints[np.argmax(Aval)]

print "final"
print newSol
print np.max(Aval)+oldA0





###one step
print "finalOneStep"
classSol=possiblePoints[np.argmax(A0val)]
print 2.0*np.max(A0val)
