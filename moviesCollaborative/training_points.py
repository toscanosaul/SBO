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
from BGO.Source import *
import time
from pmf import cross_validation,PMF
import multiprocessing as mp


n1=4
n2=1

###rate leraning, regularizing parameter, rank, epoch
lowerX=[0.01,0.1,1,1]
upperX=[1.01,2.1,21,201]



nGrid=[6,6,11,6]

domainX=[]
for i in range(n1):
    domainX.append(np.linspace(lowerX[i],upperX[i],nGrid[i]))
    
domain=[[a,b,c,d] for a in domainX[0] for b in domainX[1] for c in domainX[2] for d in domainX[3]]

"""
We define the objective object.
"""
num_user=943
num_item=1682

train=[]
validate=[]

for i in range(1,6):
    data=np.loadtxt("ml-100k/u%d.base"%i)
    test=np.loadtxt("ml-100k/u%d.test"%i)
    train.append(data)
    validate.append(test)


def g(x,w1):
    indexes = np.arange(5)
    indexes = np.delete(indexes,w1)
    train_data = data_all[indexes[0]]
    indexes = np.delete(indexes,indexes[0])
    for i in indexes:
        train_data = np.concatenate((train_data,data_all[i]),axis=0)
    
        
    val=PMF(num_user,num_item,train_data,data_all[w1],x[0],x[1],int(x[3]),int(x[2]))
    return -val*100.0
    

def noisyF(XW,n):
    """Estimate F(x,w)=E(f(x,w,z)|w)
      
       Args:
          XW: Vector (x,w)
          n: Number of samples to estimate F
    """
    
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]

    w=int(w)
    result = np.zeros(n)
    for j in range(n):
        np.random.seed(j)
        result[j] = g(x,w)
        
    return np.mean(result), np.std(result)

def sampleFromXAn(n):
    """Chooses n points in the domain of x at random
      
       Args:
          n: Number of points chosen
    """
    s1=np.random.uniform(lowerX[0:2],upperX[0:2],(n,2))
    a=np.random.randint(lowerX[2],upperX[2],n).reshape((n,1))
    b=np.random.randint(lowerX[3],upperX[3],n).reshape((n,1))
    
    
    return np.concatenate((s1,a,b),1)

sampleFromXVn=sampleFromXAn

n_folds=5

trainingPoints=60

Xtrain=sampleFromXVn(trainingPoints).reshape((trainingPoints,n1))

dt=int(ceil(trainingPoints/n_folds))
Wtrain=[]
for i in range(n_folds):
    Wtrain+=[i]*dt
Wtrain=np.array(Wtrain).reshape((trainingPoints,1))

XWtrain=np.concatenate((Xtrain,Wtrain),1)

np.savetxt("XWtrain_cluster.txt",XWtrain)

nCores = mp.cpu_count()

yTrain_2 = Parallel(n_jobs=nCores)(
    delayed(noisyF)(
        XW=XWtrain[i:i+1,:],
        n=10,
    )for i in range(trainingPoints))

yTrain_2_2 = np.zeros((trainingPoints,2))
for i in range(trainingPoints):
    yTrain_2_2[i,0] = yTrain_2[i][0]
    yTrain_2_2[i,1] = yTrain_2[i][1]
    

np.savetxt("yTrain_cluster.txt",yTrain_2_2)





