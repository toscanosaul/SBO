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
from BGOis.Source import *
import time
from pmf import cross_validation,PMF

nTemp=int(sys.argv[1])  #random seed 
nTemp2=int(sys.argv[2]) #number of training points

n1=4
n2=1

###rate leraning, regularizing parameter, rank, epoch
lowerX=[0.01,0.1,1,1]
upperX=[1.01,2.1,21,201]



nGrid=[6,6,11,6]

np.random.seed(nTemp)
trainingPoints=nTemp2

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
    val=PMF(num_user,num_item,train[w1],validate[w1],x[0],x[1],int(x[3]),int(x[2]))
    return -val*100
    

def noisyF(XW,n):
    """Estimate F(x,w)=E(f(x,w,z)|w)
      
       Args:
          XW: Vector (x,w)
          n: Number of samples to estimate F
    """
    
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]

    w=int(w)
 #   return g(x,w),0.0
    return np.sum(x)+w,0.0



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



def simulatorW(n):
    """Simulate n vectors w
      
       Args:
          n: Number of vectors simulated
    """
    return np.random.randint(0,numberIS,n).reshape((n,n2))


numberIS=5

Xtrain=sampleFromXVn(trainingPoints*numberIS).reshape((trainingPoints*numberIS,n1))


Wtrain=[]
for i in range(numberIS):
    Wtrain+=[i]*trainingPoints
Wtrain=np.array(Wtrain).reshape((trainingPoints*numberIS,1))

XWtrain=np.concatenate((Xtrain,Wtrain),1)

print XWtrain
dataObj=inter.data(XWtrain,yHist=None,varHist=None)

numberSamplesForF=1

dataObj.getTrainingDataSBO(trainingPoints*numberIS,noisyF,numberSamplesForF,True)
trainingPoints*=numberIS

path=os.path.join(trainingPoints,"%dnumberOfTP%d"%(nTemp,trainingPoints))

fl.createNewFilesFunc(path,nTemp)

rs=nTemp
tempDir=os.path.join(path,'%d'%rs+"XHist.txt")
with open(tempDir, "a") as f:
    np.savetxt(f,dataObj.Xhist)
tempDir=os.path.join(path,'%d'%rs+"yhist.txt")
with open(tempDir, "a") as f:
    np.savetxt(f,dataObj.yHist)
tempDir=os.path.join(path,'%d'%rs+"varHist.txt")
with open(tempDir, "a") as f:
    np.savetxt(f,dataObj.varHist)
