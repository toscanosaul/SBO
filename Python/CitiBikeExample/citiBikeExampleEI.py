#!/usr/bin/env python


import sys
sys.path.append("..")
#sys.path.insert(0, '/Users/saultoscano/Documents/research/optimal_globalization/repositoryOnlyForCluster/cluster/SBONew')

import numpy as np
import SquaredExponentialKernel as SK
from grid import *
import EI
from simulationPoissonProcess import *
from math import *
#import pylab
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson

nTemp=int(sys.argv[1])
nTemp2=int(sys.argv[2])
nTemp3=int(sys.argv[3])
#except:
#    nTemp=1
#    nTemp2=10
#    nTemp3=10

randomSeed=nTemp
np.random.seed(randomSeed)

g=unhappyPeople

n1=4
n2=4
numberSamplesForG=nTemp3
fil="2014-05PoissonParameters.txt"
nSets=4
A,lamb=generateSets(nSets,fil)
#####
parameterSetsPoisson=np.zeros(n2)
for j in xrange(n2):
    parameterSetsPoisson[j]=np.sum(lamb[j])
####

TimeHours=4.0
trainingPoints=nTemp2
numberBikes=6000
lowerX=100*np.ones(4)
UpperX=numberBikes*np.ones(4)
dimensionKernel=n1
nGrid=50
####to generate points run poissonGeneratePoints.py
#pointsVOI=np.loadtxt("pointsPoisson.txt")

def simulatorW(n):
    wPrior=np.zeros((n,n2))
    for i in range(n2):
        wPrior[:,i]=np.random.poisson(parameterSetsPoisson[i],n)
    return wPrior

def sampleFromX(n):
    aux1=(numberBikes/float(n1))*np.ones((1,n1-1))
    if n>1:
        temp=np.random.dirichlet(np.ones(n1),n-1)
        temp=(numberBikes-500.0*n1)*temp+500.0
        temp=temp[:,0:n1-1]
        temp=np.floor(temp)
        aux1=np.concatenate((aux1,temp),0)
    return aux1

def noisyG(X,n):
    if len(X.shape)==2:
       X=X[0,:]
    estimator=n
    W=simulatorW(estimator)
    result=np.zeros(estimator)
    for i in range(estimator):
        result[i]=g(TimeHours,W[i,:],X,nSets,lamb,A,"2014-05")

    return np.mean(result),float(np.var(result))/estimator


tempX=sampleFromX(trainingPoints)
tempFour=numberBikes-np.sum(tempX,1)
tempFour=tempFour.reshape((trainingPoints,1))
Xtrain=np.concatenate((tempX,tempFour),1)
yTrain=np.zeros([0,1])
NoiseTrain=np.zeros(0)

jobs = []
pool = mp.Pool()
for i in xrange(trainingPoints):
    job = pool.apply_async(noisyG,(Xtrain[i,:],numberSamplesForG))
    jobs.append(job)

pool.close()  # signal that no more data coming in
pool.join()  # wait for all the tasks to complete
for j in range(trainingPoints):
    temp=jobs[j].get()
    yTrain=np.vstack([yTrain,temp[0]])
    NoiseTrain=np.append(NoiseTrain,temp[1])

#########

scaleAlpha=1000.0
kernel=SK.SEK(n1,X=Xtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)

#########

logFactorial=[np.sum([log(i) for i in range(1,j+1)]) for j in range(1,501)]
logFactorial.insert(1,0)
logFactorial=np.array(logFactorial)

#Computes log*sum(exp(x)) for a vector x, but in numerically careful way
def logSumExp(x):
    xmax=np.max(np.abs(x))
    y=xmax+np.log(np.sum(np.exp(x-xmax)))
    return y


def gradXKernel(x,n,objVOI):
    kern=objVOI._k
    alpha=0.5*((kern.alpha)**2)/scaleAlpha**2
    tempN=n+trainingPoints
    X=objVOI._Xhist[0:tempN,:]
    gradX=np.zeros((tempN,n1))
    for j in xrange(n1):
        for i in xrange(tempN):
            gradX[i,j]=kern.K(x,X[i,:].reshape((1,n1)))*(-2.0*alpha[j]*(x[0,j]-X[i,j]))
    return gradX


def projectGradientDescent(x,direction,xo):
    minx=np.min(x)
    alph=[]
    if (minx < 0):
        ind=np.where(direction<0)[0]
        quotient=xo[ind].astype(float)/direction[ind]
        alp=-1.0*np.max(quotient)
        alph.append(alp)
    if (np.sum(x[0:n1])>numberBikes):
        if (np.sum(direction[0:n1])>0):
            alph2=(float(numberBikes)-np.sum(xo[0:n1]))/(np.sum(direction[0:n1]).astype(float))
            alph.append(alph2)
    if (len(alph)==0):
        return x
    return xo+direction*min(alph)

##EI object
def functionGradientAscentVn(x,grad,EI,i):
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    tempX=x[0:1,0:n1-1]
    x2=np.concatenate((tempX,x4),1)
    temp=EI._VOI.VOIfunc(i,x2,grad=grad)
    if grad==True:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
	grad2=np.dot(temp[1],L)
        return temp[0],grad2
    else:
        return temp
    

def functionGradientAscentMuN(x,grad,SBO,i):
    x4=np.array(numberBikes-np.sum(x)).reshape((1,1))
    x=np.concatenate((x,x4),1)
    temp=SBO._VOI._GP.muN(x,i,grad)
    if grad==False:
        return temp
    else:
        t=np.diag(np.ones(n1-1))
	s=-1.0*np.ones((1,n1-1))
	L=np.concatenate((t,s))
        grad2=np.dot(temp[1],L)
        return temp[0],grad2

dimXsteepest=n1-1

##transform the result steepest ascent is getting (x1,x2,x3) to  the right domain of x (x1,x2,x3,x4)
def transformationDomainX(x):
    x4=np.array(numberBikes-np.sum(np.floor(x))).reshape((1,1))
    x=np.concatenate((np.floor(x),x4),1)
    return x

###returns the value and the variance
def estimationObjective(x):
    estimator=1000
    W=simulatorW(estimator)
    result=np.zeros(estimator)
    for i in range(estimator):
        result[i]=g(TimeHours,W[i,:],x,nSets,lamb,A,"2014-05")

    return np.mean(result),float(np.var(result))/estimator

nameDirectory="Results"+'%d'%numberSamplesForG+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"

l={}
l['folderContainerResults']=os.path.join(nameDirectory,"EI")
l['estimationObjective']=estimationObjective
l['transformationDomainX']=transformationDomainX
l['dimXsteepest']=dimXsteepest
l['functionGradientAscentVn']=functionGradientAscentVn
l['functionGradientAscentMuN']=functionGradientAscentMuN
l['sampleFromX']=sampleFromX
l['projectGradient']=projectGradientDescent
l['fobj']=g
l['dimensionKernel']=dimensionKernel
l['numberTrainingData']=trainingPoints
l['numberEstimateG']=numberSamplesForG
l['constraintA']=lowerX
l['constraintB']=UpperX
l['Xhist']=Xtrain
l['yHist']=yTrain
l['varHist']=NoiseTrain
l['kernel']=kernel
l['randomSeed']=randomSeed
#l['pointsVOI']=pointsVOI
l['gradXKern']=gradXKernel
l['numberParallel']=3
l['noisyG']=noisyG
l['scaledAlpha']=scaleAlpha
l['xtol']=1.0
def conditionOpt(x):
    return np.max((np.floor(np.abs(x))))
l['functionConditionOpt']=conditionOpt
eiObj=EI.EI(**l)

eiObj.EIAlg(10,nRepeat=3,Train=True)

