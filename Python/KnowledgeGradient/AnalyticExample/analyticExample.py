#!/usr/bin/env python

import sys
sys.path.append("..")
sys.path.append("../..")
import numpy as np
import SquaredExponentialKernel as SK
from grid import *
import KG
import sys
from math import *
import os

randomSeed=int(sys.argv[1])
np.random.seed(randomSeed)



###Objective function
def g(x,w1,w2):
    val=(w2)/(w1)
    return -(val)*(x**2)-w1

n1=1
n2=1
varianceW2givenW1=1.0
dimensionKernel=1
trainingPoints=int(sys.argv[2])
numberSamplesForG=int(sys.argv[3])
lowerX=[-3.0]
UpperX=[3.0]
nGrid=50
pointsVOI=grid(lowerX,UpperX,nGrid)


###parameters of w1
varianceW1=np.array(1.0).reshape(1)
muW1=np.array(0.0).reshape(1)



def simulatorW(n):
    return np.random.normal(0,1,n).reshape((n,n2))

def noisyG(X,n):
    W=simulatorW(n)
    Z=np.random.normal(W,1)
    samples=g(X,W,Z)
    return np.mean(samples),float(np.var(samples))/n

Xtrain=np.random.uniform(lowerX,UpperX,trainingPoints).reshape((trainingPoints,n1)) ##to estimate the kernel
#Wtrain=simulatorW(trainingPoints)
#XWtrain=np.concatenate((Xtrain,Wtrain),1)
yTrain=np.zeros([0,1])
NoiseTrain=np.zeros(0)

for i in xrange(trainingPoints):
    temp=noisyG(Xtrain[i,:],numberSamplesForG)
    yTrain=np.vstack([yTrain,temp[0]])
    NoiseTrain=np.append(NoiseTrain,temp[1])


kernel=SK.SEK(n1,X=Xtrain,y=yTrain[:,0],noise=NoiseTrain) ###check que kernel cambie sus param para funcion B

#gamma=np.transpose(self._k.A(new,past,noise=self._noiseHist))


def gradXKernel(x,n,objVOI):
    kern=objVOI._k
    alpha=0.5*((kern.alpha)**2)
    tempN=n+trainingPoints
    X=objVOI._Xhist[0:tempN,:]
    gradX=np.zeros((tempN,n1))
    for j in xrange(n1):
        for i in xrange(tempN):
            gradX[i,j]=kern.K(x,X[i,:].reshape((1,n1)))*(2.0*alpha[j]*(x[0,j]-X[i,j]))
    return gradX

def gradXKernel2(x,i,keep,j,objVOI):
    kern=objVOI._k
    alpha=0.5*((kern.alpha)**2)
    return kern.K(x,pointsVOI[keep[i]:keep[i]+1,:])*(2.0*alpha[j]*(x[0,j]-pointsVOI[keep[i],j]))

def projectGradientDescent(x):
    c=lowerX
    d=UpperX
    
    if (any(x<c)):
        temp1=np.array(X[0,0:n1]).reshape(n1)
        index2=np.where(temp1<c)
        x[index2[0]]=c[index2[0]]
   
    if (any(x>d)):
        index2=np.where(x>d)
        x[index2[0]]=d[index2[0]]
        
    return x

def sampleFromX(n):
    return np.random.uniform(lowerX,UpperX,(n,n1))

##EI object
def functionGradientAscentVn(x,grad,KG,i):
    temp=KG._VOI.VOIfunc(i,x,grad=grad)
    if grad==True:
        return temp[0],temp[1]
    else:
        return temp
    

def functionGradientAscentMuN(x,grad,KG,i):
    temp=KG._VOI._GP.muN(x,i,grad)
    if grad==False:
        return temp
    else:
        return temp[0],temp[1]

dimXsteepest=n1

def transformationDomainX(x):
    return x


###returns the value and the variance
def estimationObjective(x):
    return -x**2,0

nameDirectory="Results"+'%d'%numberSamplesForG+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"

l={}
l['folderContainerResults']=os.path.join(nameDirectory,"KG")
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
l['pointsVOI']=pointsVOI

l['gradXKern']=gradXKernel
l['gradXKern2']=gradXKernel2
l['numberParallel']=1
l['noisyG']=noisyG

kgObj=KG.KG(**l)

kgObj.KGAlg(20,nRepeat=1,Train=True)

