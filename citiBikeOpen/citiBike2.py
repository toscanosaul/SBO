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
import random
from sklearn import datasets, linear_model
import itertools

nTemp=int(sys.argv[1])
randomSeed=nTemp
np.random.seed(randomSeed)

###regression

beta=np.loadtxt("coefficients.txt")
sigma= 53.7
K=58 #number of covariates

z=np.loadtxt("mymatrix.txt")
r=np.ones((z.shape[0],1))

nRoutes=10





nTraining=10

t2= np.random.choice(z.shape[0], size=nTraining, replace=False)

possibleIndexes=[i for i in range(z.shape[0]) if i not in t2]
t1= np.random.choice(z.shape[0]-nTraining, size=nRoutes, replace=False)


t1=[possibleIndexes[i] for i in t1]


a=z[t1,1:] #values of covariates
a=np.append(r[0:nRoutes,0:1],a,axis=1)

nRoutes=a.shape[0]

def f(i):
    
    mu=np.dot(a[i,:],beta)
    res=np.random.normal(mu, sigma, 1)
    return res




###prior
alpha0=beta+0.001
beta0=np.identity(len(beta))




Xdata=z[t2,1:]
yData=z[t2,0]

regr = linear_model.LinearRegression()

regr.fit(Xdata, yData)


sigmaEst=np.sqrt(np.mean((regr.predict(Xdata) - yData) ** 2))

alpha0=np.array(regr.coef_)
alpha0=np.append(np.array([regr.intercept_]),regr.coef_)

beta0=np.identity(len(beta))*(sigmaEst**2)

def newY(x):
    N=len(x)
    res=0
    count=0
    for i in range(N):
        if x[i]==1:
            count+=1
            res+=f(i)
    
    return res,count*(sigmaEst**2)

###weekly period-a unit observations, iid, throw weeks with big holyday.
###you have several observations  (ignore geostatics), fit variance with MLE
###non-informtive prior on this common variance
##nyc: may-august. 4 iid on the course of month

###choose 10 bike routes. fit the prior.
###new transportation that and we're only using data from citibike
###who is my tarjet market from point to point transportation? towards ride bikes

###1 

###2 

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


totalOpenings=5

possiblePoints2 = list(itertools.product([0, 1], repeat=nRoutes))
indexesPoints=range(len(possiblePoints2))

possiblePoints3=[(i,possiblePoints2[i]) for i in indexesPoints if np.sum(possiblePoints2[i])<=totalOpenings]
possiblePoints=[(i,j) for (i,j) in possiblePoints3 if 0<np.sum(j)<totalOpenings]
indexesPoints=range(len(possiblePoints))
possiblePoints=[(i,possiblePoints[i][1]) for i in indexesPoints  ]
aVec=np.zeros(len(possiblePoints))

i=0

for i in range(len(possiblePoints)):
    aVec[i]=a_0(possiblePoints[i][1])
       # i+=1

    

def previous():

    possiblePoints2 = list(itertools.product([0, 1], repeat=nRoutes))
    
    possiblePoints=[i for i in possiblePoints2 if np.sum(i)<=totalOpenings]
    
    aVec=np.zeros(len(possiblePoints))
    
    i=0
    for x in possiblePoints:
        aVec[i]=a_0(x)
        i+=1
        
    possiblePoints=[i for i in possiblePoints if np.sum(i)==totalOpenings]
    Aval=np.zeros(len(possiblePoints))
    
    
    
    Npoint=len(possiblePoints)
    A0val=np.zeros(len(possiblePoints))
    for i in range(Npoint):
        A0val[i]=a_0(possiblePoints[i])


def bfunc(x,possiblePoints3):
    z=np.zeros(len(possiblePoints3))
    ind=0
    for j in possiblePoints3:
        z[ind]=sigma_0(x,j)
        ind+=1
    
    div=np.sqrt((sigma_0(x,x)))
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
    


def VOI(x,a2=aVec,grad=False):
    lsum=np.sum(x)
    

    newIndexes=[(i,j) for (i,j) in possiblePoints if np.sum(j)==totalOpenings-lsum]
    bVec=bfunc(x,[j for (i,j) in newIndexes])

    bVec=bVec+np.sqrt(sigma_0(x,x))
  
    a2=np.array([a2[i] for (i,j) in newIndexes])
    a2=a2+a_0(x)

    a,b,keep=AffineBreakPointsPrep(a2,bVec)
    
    keep1,c=AffineBreakPoints(a,b)
    keep1=keep1.astype(np.int64)
    M=len(keep1)
    keep2=keep[keep1]
    
    return hvoi(b,c,keep1)+a_0(x)

x=np.array([1,0])

VOIval=np.zeros(len(possiblePoints))

Npoint=len(possiblePoints)
for i in range(Npoint):
    VOIval[i]=VOI(possiblePoints[i][1])

print "first"
#print possiblePoints[np.argmax(VOIval)
oldPoint=possiblePoints[np.argmax(VOIval)][1]


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

possiblePointsAn=[(i,j) for (i,j) in possiblePoints3 if np.sum(j)==totalOpenings-np.sum(oldPoint)]
Aval=np.zeros(len(possiblePointsAn))



Npoint=len(possiblePointsAn)
A0val=np.zeros(len(possiblePointsAn))
for i in range(Npoint):
    Aval[i]=a_1(possiblePointsAn[i][1])
    A0val[i]=a_0(possiblePointsAn[i][1])

newSol=possiblePointsAn[np.argmax(Aval)][1]


print "final"
print newSol
print np.max(Aval)+oldA0
print "real value"

val=0
for i in range(len(newSol)):
    if newSol[i]==1:
        val+=np.dot(a[i,:],beta)
    if oldPoint[i]==1:
        val+=2.0*np.dot(a[i,:],beta)
print val
valSol=val




###one step
print "finalOneStep"
classSol=possiblePoints[np.argmax(A0val)][1]
print 2.0*np.max(A0val)
print "real Value"

val=0
for i in range(len(newSol)):

    if classSol[i]==1:
        val+=np.dot(a[i,:],beta)


path="Results"

if not os.path.exists(path):
    os.makedirs(path)

f=open(os.path.join(path,'%d'%randomSeed+"results.txt"),'w')
f.close()

with open(os.path.join(path,'%d'%randomSeed+"results.txt"), "a") as f:
   # var=np.array(var).reshape(1)
    aux=np.array(2.0*val).reshape(1)
    np.savetxt(f,aux)
    aux=np.array(valSol).reshape(1)
    np.savetxt(f,aux)
    
    quot=(valSol-2.0*val)/(2.0*np.abs(val))

    aux=np.array(quot).reshape(1)
    np.savetxt(f,aux)
