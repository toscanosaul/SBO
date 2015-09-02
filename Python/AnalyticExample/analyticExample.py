#!/usr/bin/env python

import sys
sys.path.append("..")

import numpy as np
import SquaredExponentialKernel as SK
from grid import *
import SBOGeneral2 as SB
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
dimensionKernel=2
trainingPoints=int(sys.argv[2])
numberSamplesForF=int(sys.argv[3])
lowerX=[-3.0]
UpperX=[3.0]
nGrid=50
pointsVOI=grid(lowerX,UpperX,nGrid)


###parameters of w1
varianceW1=np.array(1.0).reshape(1)
muW1=np.array(0.0).reshape(1)


def noisyF(XW,n):
    X=XW[0,0:n1]
    W=XW[0,n1:n1+n2]
    t=np.array(np.random.normal(W,varianceW2givenW1,n))
    t=g(X,W,t)
    return np.mean(t),float(np.var(t))/n

def simulatorW(n):
    return np.random.normal(0,1,n).reshape((n,n2))

Xtrain=np.random.uniform(lowerX,UpperX,trainingPoints).reshape((trainingPoints,n1)) ##to estimate the kernel
Wtrain=simulatorW(trainingPoints)
XWtrain=np.concatenate((Xtrain,Wtrain),1)
yTrain=np.zeros([0,1])
NoiseTrain=np.zeros(0)

for i in xrange(trainingPoints):
    temp=noisyF(XWtrain[i,:].reshape((1,n1+n2)),numberSamplesForF)
    yTrain=np.vstack([yTrain,temp[0]])
    NoiseTrain=np.append(NoiseTrain,temp[1])


kernel=SK.SEK(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain) ###check que kernel cambie sus param para funcion B

def B(x,XW,n1,n2):
    X=XW[0:n1]
    W=XW[n1:n1+n2]
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)
    alpha1=0.5*((kernel.alpha[0:n1])**2)
    variance0=kernel.variance
    tmp=-(((muW1/varianceW1)+2.0*(alpha2)*np.array(W))**2)/(4.0*(-alpha2-(1/(2.0*varianceW1))))
    tmp2=-alpha2*(np.array(W)**2)
    tmp3=-(muW1**2)/(2.0*varianceW1)
    tmp=np.exp(tmp+tmp2+tmp3)
    tmp=tmp*(1/(sqrt(2.0*varianceW1)))*(1/(sqrt((1/(2.0*varianceW1))+alpha2)))
    x=np.array(x).reshape((x.size,n1))
    tmp1=variance0*np.exp(np.sum(-alpha1*((np.array(x)-np.array(X))**2),axis=1))
    return np.prod(tmp)*tmp1

#gamma=np.transpose(self._k.A(new,past,noise=self._noiseHist))

def gradXWSigmaOfunc(n,new,objVOI,Xtrain2,Wtrain2):
    gradXSigma0=np.zeros([n+trainingPoints+1,n1])
    kern=objVOI._k
    tempN=n+trainingPoints
    past=objVOI._PointsHist[0:tempN,:]
    gamma=np.transpose(kern.A(new,past))
    alpha1=0.5*((kern.alpha[0:n1])**2)
    Xtrain=past[:,0:n1]
    gradWSigma0=np.zeros([n+trainingPoints+1,n2])
    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)
    xNew=new[0,0:n1]
    wNew=new[0,n1:n1+n2]
    for i in xrange(n+trainingPoints):
        gradXSigma0[i,:]=-2.0*gamma[i]*alpha1*(xNew-Xtrain2[i,:])
        gradWSigma0[i,:]=-2.0*gamma[i]*alpha2*(wNew-Wtrain2[i,:])
    return gradXSigma0,gradWSigma0
    

####these gradients are evaluated in all set of the points of the discretization (in the approximation)
def gradXB(new,objVOI,BN,keep):
    points=objVOI._points
    kern=objVOI._k
    alpha1=0.5*((kern.alpha[0:n1])**2)
    xNew=new[0,0:n1].reshape((1,n1))
    gradXBarray=np.zeros([len(keep),n1])
    M=len(keep)
    for i in xrange(n1):
        for j in xrange(M):
            gradXBarray[j,i]=-2.0*alpha1[i]*BN[keep[j],0]*(xNew[0,i]-points[keep[j],i])
    return gradXBarray

def gradWB(new,objVOI,BN,keep):
    points=objVOI._points
    kern=objVOI._k
    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)
    wNew=new[0,n1:n1+n2].reshape((1,n2))
    gradWBarray=np.zeros([len(keep),n2])
    M=len(keep)
    for i in xrange(n2):
        for j in xrange(M):
            gradWBarray[j,i]=BN[keep[j],0]*(alpha2[i]*(muW1[i]-wNew[0,i]))/((varianceW1[i]*alpha2[i]+.5))
    return gradWBarray

def gradXBforAn(x,n,B,objGP,X):
    gradXB=np.zeros((n1,n+trainingPoints))
    kern=objGP._k
    alpha1=0.5*((kern.alpha[0:n1])**2)
    for i in xrange(n+trainingPoints):
        gradXB[:,i]=B[i]*(-2.0*alpha1*(x-X[i,:]))
    return gradXB

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

##SBO object
def functionGradientAscentVn(x,grad,SBO,i):
    temp=SBO._VOI.VOIfunc(i,x,grad=grad)
    if grad==True:
        return temp[0],temp[1]
    else:
        return temp
    
##SBO object
def functionGradientAscentVn(x,grad,SBO,i):
    temp=SBO._VOI.VOIfunc(i,x,grad=grad)
    if grad==True:
        return temp[0],temp[1]
    else:
        return temp

def functionGradientAscentAn(x,grad,SBO,i,L):
    temp=SBO._VOI._GP.aN_grad(x,L,i,grad)
    if grad==False:
        return temp
    else:
        return temp[0],temp[1]

dimXsteepest=n1

def transformationDomainX(x):
    return x

def transformationDomainW(w):
    return w

###returns the value and the variance
def estimationObjective(x):
    return -x**2,0

nameDirectory="Results"+'%d'%numberSamplesForF+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"

l={}
l['folderContainerResults']=os.path.join(nameDirectory,"SBO")
l['estimationObjective']=estimationObjective
l['transformationDomainW']=transformationDomainW
l['transformationDomainX']=transformationDomainX
l['dimXsteepest']=dimXsteepest
l['functionGradientAscentVn']=functionGradientAscentVn
l['functionGradientAscentAn']=functionGradientAscentAn
l['sampleFromX']=sampleFromX
l['projectGradient']=projectGradientDescent
l['fobj']=g
l['dimensionKernel']=dimensionKernel
l['noisyF']=noisyF
l['dimSeparation']=n1
l['numberTrainingData']=trainingPoints
l['numberEstimateF']=numberSamplesForF
l['constraintA']=lowerX
l['constraintB']=UpperX
l['simulatorW']=simulatorW
l['XWhist']=XWtrain
l['yHist']=yTrain
l['varHist']=NoiseTrain
l['kernel']=kernel
l['B']=B
l['gradXWSigmaOfunc']=gradXWSigmaOfunc
l['gradXBfunc']=gradXB
l['gradWBfunc']=gradWB
l['randomSeed']=randomSeed
l['pointsVOI']=pointsVOI
l['gradXBforAn']=gradXBforAn
l['numberParallel']=10

sboObj=SB.SBO(**l)

sboObj.SBOAlg(20,nRepeat=10,Train=True)

