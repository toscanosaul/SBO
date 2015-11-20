#!/usr/bin/env python

"""
We consider a queuing simulation based on New York City's Bike system,
in which system users may remove an available bike from a station at one
location within the city, and ride it to a station with an available dock
in some other location within the city. The optimization problem that we
consider is the allocation of a constrained number of bikes (6000) to available
docks within the city at the start of rush hour, so as to minimize, in simulation,
the expected number of potential trips in which the rider could not find an
available bike at their preferred origination station, or could not find an
available dock at their preferred destination station. We call such trips
"negatively affected trips".

To use the KG algorithm, we need to create 6 objets:

Objobj: Objective object (See InterfaceSBO).
miscObj: Miscellaneous object (See InterfaceSBO).
VOIobj: Value of Information function object (See VOIGeneral).
optObj: Opt object (See InterfaceSBO).
statObj: Statistical object (See statGeneral).
dataObj: Data object (See InterfaceSBO).

"""

import sys
sys.path.append("..")
import numpy as np
from simulationPoissonProcess import *
from math import *
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson
from BGO.Source import *

nTemp=int(sys.argv[1])
nTemp2=int(sys.argv[2])
nTemp3=int(sys.argv[3])
nTemp4=int(sys.argv[4]) #number of iterations
nTemp5=sys.argv[5] #True if code is run in parallel; False otherwise.

if nTemp5=='F':
    nTemp5=False
    nTemp6=1
elif nTemp5=='T':
    nTemp6=int(sys.argv[6]) #number of restarts for the optimization method
    nTemp5=True

randomSeed=nTemp
np.random.seed(randomSeed)

######

n1=4
n2=4
numberSamplesForG=nTemp3

######

"""
We define the variables needed for the queuing simulation. 
"""

g=unhappyPeople


fil="2014-05PoissonParameters.txt"
nSets=4
A,lamb=generateSets(nSets,fil)

parameterSetsPoisson=np.zeros(n2)
for j in xrange(n2):
    parameterSetsPoisson[j]=np.sum(lamb[j])
    
exponentialTimes=np.loadtxt("2014-05"+"ExponentialTimes.txt")
with open ('json.json') as data_file:
    data=json.load(data_file)

f = open(str(4)+"-cluster.txt", 'r')
cluster=eval(f.read())
f.close()

bikeData=np.loadtxt("bikesStationsOrdinalIDnumberDocks.txt",skiprows=1)

TimeHours=4.0
numberBikes=6000

"""
We define the objective object.
"""

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
        result[i]=g(TimeHours,W[i,:],X,nSets,lamb,A,"2014-05",
                    exponentialTimes,data,cluster,bikeData)
    return np.mean(result),float(np.var(result))/estimator

def estimationObjective(x,N=100):
    estimator=N
    W=simulatorW(estimator)
    result=np.zeros(estimator)
    for i in range(estimator):
        result[i]=g(TimeHours,W[i,:],x,nSets,lamb,A,"2014-05",
                    exponentialTimes,data,cluster,bikeData)
    return np.mean(result),float(np.var(result))/estimator

Objective=inter.objective(g,n1,noisyG,numberSamplesForG,sampleFromX,
                          simulatorW,estimationObjective)

"""
We define the miscellaneous object.
"""
parallel=nTemp5

trainingPoints=nTemp2

#nameDirectory="Results"+'%d'%numberSamplesForG+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"
#folder=os.path.join(nameDirectory,"KG")

misc=inter.Miscellaneous(randomSeed,parallel,nF=numberSamplesForG,tP=trainingPoints,ALG="KG")

"""
We define the data object.
"""

"""
Generate the training data
"""

tempX=sampleFromX(trainingPoints)
tempFour=numberBikes-np.sum(tempX,1)
tempFour=tempFour.reshape((trainingPoints,1))
Xtrain=np.concatenate((tempX,tempFour),1)

dataObj=inter.data(Xtrain,yHist=None,varHist=None)
dataObj.getTrainingDataKG(trainingPoints,noisyG,numberSamplesForG,parallel)

"""
We define the statistical object.
"""

dimensionKernel=n1

scaleAlpha=1000.0

stat=stat.KG(dimKernel=dimensionKernel,numberTraining=trainingPoints,
                scaledAlpha=scaleAlpha, dimPoints=n1,trainingData=dataObj)

"""
We define the VOI object.
"""

pointsVOI=np.loadtxt("pointsPoisson.txt")

voiObj=VOI.KG(numberTraining=trainingPoints,
           pointsApproximation=pointsVOI,dimX=n1)

"""
We define the Opt object.
"""

dimXsteepest=n1-1 #Dimension of x when the VOI and a_{n} are optimized.

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
def functionGradientAscentVn(x,grad,VOI,i,L,data,kern,temp1,temp2,a,onlyGrad):
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    tempX=x[0:1,0:n1-1]
    x2=np.concatenate((tempX,x4),1)
    temp=VOI.VOIfunc(i,x2,L,data,kern,temp1,temp2,grad,a,onlyGrad)
    if onlyGrad:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        grad2=np.dot(temp,L)
        return grad2
        
    
    if grad==True:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        grad2=np.dot(temp[1],L)
        return temp[0],grad2
    else:
        return temp
    
def functionGradientAscentMuN(x,grad,data,stat,i,L,temp1,onlyGrad):
    x4=np.array(numberBikes-np.sum(x)).reshape((1,1))
    x=np.concatenate((x,x4),1)
    temp=stat.muN(x,i,data,L,temp1,grad,onlyGrad)
    if onlyGrad:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        grad2=np.dot(temp,L)
        return grad2
    if grad:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        grad2=np.dot(temp[1],L)
        return temp[0],grad2
    else:
        return temp

dimXsteepest=n1-1

def transformationDomainX(x):
    x4=np.array(numberBikes-np.sum(np.floor(x))).reshape((1,1))
    x=np.concatenate((np.floor(x),x4),1)
    return x


def conditionOpt(x):
    return np.max((np.floor(np.abs(x))))
###returns the value and the variance

opt=inter.opt(nTemp6,dimXsteepest,transformationDomainX,None,projectGradientDescent,functionGradientAscentVn,
              functionGradientAscentMuN,conditionOpt,1.0)


#nameDirectory="Results"+'%d'%numberSamplesForG+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"

l={}
l['VOIobj']=voiObj
l['Objobj']=Objective
l['miscObj']=misc
l['optObj']=opt
l['statObj']=stat
l['dataObj']=dataObj

kgObj=KG.KG(**l)

kgObj.KGAlg(nTemp4,nRepeat=10,Train=True)

