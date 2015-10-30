#!/usr/bin/env python


import sys
sys.path.append("..")
#sys.path.insert(0, '/Users/saultoscano/Documents/research/optimal_globalization/repositoryOnlyForCluster/cluster/SBONew')

import numpy as np
import SquaredExponentialKernel as SK
from grid import *
import SBOGeneral2 as SB
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
import optimization as op


directory=[]

directory.append(os.path.join("..","CitiBikeExample","Results15AveragingSamples5TrainingPoints","SBO","1300run"))

nTemp=1300
nTemp2=15
nTemp3=15
#except:
#    nTemp=1
#    nTemp2=10
#    nTemp3=10

randomSeed=nTemp
np.random.seed(randomSeed)

g=unhappyPeople

n1=4
n2=4
numberSamplesForF=nTemp3
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
dimensionKernel=n1+n2
nGrid=50
####to generate points run poissonGeneratePoints.py
pointsVOI=np.loadtxt("pointsPoisson.txt")

####CAMBIAR NOISYf NO DEBE DEPENDERR DE RANDOM SEED!!!
def noisyF(XW,n):
    simulations=np.zeros(n)
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]
    for i in xrange(n):
        simulations[i]=g(TimeHours,w,x,nSets,lamb,A,"2014-05")
    return np.mean(simulations),float(np.var(simulations))/n




#####work this
def simulatorW(n):
    wPrior=np.zeros((n,n2))
    for i in range(n2):
        wPrior[:,i]=np.random.poisson(parameterSetsPoisson[i],n)
    return wPrior




###Used to select a starting point for gradient ascent
def sampleFromX(n):
    aux1=(numberBikes/float(n1))*np.ones((1,n1-1))
    if n>1:
        temp=np.random.dirichlet(np.ones(n1),n-1)
	temp=(numberBikes-500.0*n1)*temp+500.0
    	temp=temp[:,0:n1-1]
    	temp=np.floor(temp)
	aux1=np.concatenate((aux1,temp),0)
    return aux1



###################################################
parallel=False
yTrain=np.zeros([0,1])
NoiseTrain=np.zeros(0)
ind=1300
def readTrainingData(n,directory):
    XWtrain=np.loadtxt(os.path.join(directory,"%d"%ind+"XWHist.txt"))[0:n,:]
    yTrain=np.loadtxt(os.path.join(directory,"%d"%ind+"yhist.txt"))[0:n]
    yTrain=yTrain.reshape((n,1))
    NoiseTrain=np.loadtxt(os.path.join(directory,"%d"%ind+"varHist.txt"))[0:n]
    return XWtrain,yTrain,NoiseTrain

XWtrain,yTrain,NoiseTrain=readTrainingData(trainingPoints,directory[0])



#########

scaleAlpha=1000.0
kernel=SK.SEK(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)

def readKernelParam(file1,dimKernel):
    f=open(file1, 'r')
    v=f.read()
    val=v.split(':')
    temp=val[1].split(",")
    alpha=np.zeros(dimKernel)
    alpha[0]=float(temp[0].split("[")[1])
    for i in range(dimKernel-2):
        alpha[i+1]=float(temp[i+1])
    alpha[dimKernel-1]=float(temp[dimKernel-1].split("]")[0])
    variance=float(val[2].split(",")[0])
    mu=float(val[3].split("(")[1].split(')')[0])
    return alpha,variance,mu

alpha,variance,mu=readKernelParam(os.path.join(directory[0],"1300hyperparameters.txt"),n1+n2)

kernel.variance=variance
kernel.mu=mu
kernel.alpha=np.sqrt(2.0*(scaleAlpha*alpha))

#########

logFactorial=[np.sum([log(i) for i in range(1,j+1)]) for j in range(1,501)]
logFactorial.insert(1,0)
logFactorial=np.array(logFactorial)

#Computes log*sum(exp(x)) for a vector x, but in numerically careful way
def logSumExp(x):
    xmax=np.max(np.abs(x))
    y=xmax+np.log(np.sum(np.exp(x-xmax)))
    return y


def B(x,XW,n1,n2,logproductExpectations=None):
    x=np.array(x).reshape((x.shape[0],n1))
    results=np.zeros(x.shape[0])
    parameterLamb=parameterSetsPoisson
    X=XW[0:n1]
    W=XW[n1:n1+n2]
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha**2
    alpha1=0.5*((kernel.alpha[0:n1])**2)/scaleAlpha**2
    variance0=kernel.variance
    
    if logproductExpectations is None:
        logproductExpectations=0.0
        for j in xrange(n2):
            G=poisson(parameterLamb[j])
            temp=G.dist.expect(lambda z: np.exp(-alpha2[j]*((z-W[j])**2)),G.args)
            logproductExpectations+=np.log(temp)
            
    for i in xrange(x.shape[0]):
        results[i]=logproductExpectations+np.log(variance0)-np.sum(alpha1*((x[i,:]-X)**2))
    return np.exp(results)



####this function is the same for any squared exponential kernel
def gradXWSigmaOfunc(n,new,objVOI,Xtrain2,Wtrain2):
    gradXSigma0=np.zeros([n+trainingPoints+1,n1])
    kern=objVOI._k
    tempN=n+trainingPoints
    past=objVOI._PointsHist[0:tempN,:]
    gamma=np.transpose(kern.A(new,past))

    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
    Xtrain=past[:,0:n1]
    gradWSigma0=np.zeros([n+trainingPoints+1,n2])
    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)/scaleAlpha**2
    xNew=new[0,0:n1]
    wNew=new[0,n1:n1+n2]
    for i in xrange(n+trainingPoints):
        gradXSigma0[i,:]=-2.0*gamma[i]*alpha1*(xNew-Xtrain2[i,:])
        gradWSigma0[i,:]=-2.0*gamma[i]*alpha2*(wNew-Wtrain2[i,:])
    return gradXSigma0,gradWSigma0



    

####these gradients are evaluated in all set of the points of the discretization (in the approximation)
###this function is the same for any squared exponential kernel
def gradXB(new,objVOI,BN,keep):
    points=objVOI._points
    kern=objVOI._k
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
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
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)/scaleAlpha**2
    variance0=kern.variance
    wNew=new[0,n1:n1+n2].reshape((1,n2))
    gradWBarray=np.zeros([len(keep),n2])
    M=len(keep)
    parameterLamb=parameterSetsPoisson
   # quantil=int(poisson.ppf(.99999999,max(parameterLamb)))
   # expec=np.array([i for i in xrange(quantil)])
   # logproductExpectations=0.0
  #  a=range(n2)
    X=new[0,0:n1]
    W=new[0,n1:n1+n2]
   
    for i in xrange(n2):
        logproductExpectations=0.0
        a=range(n2)
        del a[i]
        for r in a:
            G=poisson(parameterLamb[r])
            temp=G.dist.expect(lambda z: np.exp(-alpha2[r]*((z-W[r])**2)),G.args)
            logproductExpectations+=np.log(temp)
        G=poisson(parameterLamb[i])
        temp=G.dist.expect(lambda z: -2.0*alpha2[i]*(-z+W[i])*np.exp(-alpha2[i]*((z-W[i])**2)),G.args)
        productExpectations=np.exp(logproductExpectations)*temp
        for j in xrange(M):
            gradWBarray[j,i]=np.log(variance0)-np.sum(alpha1*((points[keep[j],:]-X)**2))
            gradWBarray[j,i]=np.exp(gradWBarray[j,i])*productExpectations
    return gradWBarray

###the same for any squared exponential kernel
def gradXBforAn(x,n,B,objGP,X):
    gradXB=np.zeros((n1,n+trainingPoints))
    kern=objGP._k
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
    for i in xrange(n+trainingPoints):
        gradXB[:,i]=B[i]*(-2.0*alpha1*(x-X[i,:]))
    return gradXB

####don't need it: we can use the chain rule
####reduce dimensions of the problem
def gradXBforAn2(x,n,B,objGP,X):
    #x=x.
    x4=numberBikes-np.sum(x)
    x2=np.concatenate((x,[[numberBikes-np.sum(x)]]),1)
    gradXB=np.zeros((n1-1,n+trainingPoints))
    kern=objGP._k
    alpha1=0.5*((kern.alpha[0:n1])**2)
    for i in xrange(n+trainingPoints):
        gradXB[:,i]=B[i]*(-2.0*alpha1[0:n1-1]*(x-X[i,0:n1-1])+2.0*alpha1[n1]*(x4-X[i,n1-1]))
    return gradXB








###at eatch steept of the gradient ascent method, it will project the point using this function
##direction is the gradient; xo is the starting point before moving
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

####we eliminate one variable to optimize the function
def functionGradientAscentVn(x,grad,SBO,i,L,onlyGradient=False):
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    tempX=x[0:1,0:n1-1]
    x2=np.concatenate((tempX,x4),1)
    tempW=x[0:1,n1-1:n1-1+n2]
    xFinal=np.concatenate((x2,tempW),1)
    temp=SBO._VOI.VOIfunc(i,xFinal,L=L,grad=grad,onlyGradient=onlyGradient)
    

    if onlyGradient:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        subMatrix=np.zeros((n2,n1-1))
        L=np.concatenate((L,subMatrix))
        subMatrix=np.zeros((n1,n2))
        temDiag=np.identity(n2)
        sub=np.concatenate((subMatrix,temDiag))
        L=np.concatenate((L,sub),1)
        grad2=np.dot(temp,L)
        return grad2
        

    if grad==True:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L=np.concatenate((t,s))
        subMatrix=np.zeros((n2,n1-1))
        L=np.concatenate((L,subMatrix))
        subMatrix=np.zeros((n1,n2))
        temDiag=np.identity(n2)
        sub=np.concatenate((subMatrix,temDiag))
        L=np.concatenate((L,sub),1)
        grad2=np.dot(temp[1],L)
        return temp[0],grad2
    else:
        return temp

####the function that steepest ascent will optimize
def functionGradientAscentAn(x,grad,SBO,i,L,onlyGradient=False,logproductExpectations=None):
    x4=np.array(numberBikes-np.sum(x)).reshape((1,1))
    x=np.concatenate((x,x4),1)
    if onlyGradient:
        temp=SBO._VOI._GP.aN_grad(x,L,i,grad,onlyGradient,logproductExpectations)
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L2=np.concatenate((t,s))
        grad2=np.dot(temp,L2)
        return grad2

    temp=SBO._VOI._GP.aN_grad(x,L,i,gradient=grad,logproductExpectations=logproductExpectations)
    if grad==False:
        return temp
    else:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L2=np.concatenate((t,s))
        grad2=np.dot(temp[1],L2)
        return temp[0],grad2

dimXsteepest=n1-1

##transform the result steepest ascent is getting (x1,x2,x3) to  the right domain of x (x1,x2,x3,x4)
def transformationDomainX(x):
    x4=np.array(numberBikes-np.sum(np.floor(x))).reshape((1,1))
    x=np.concatenate((np.floor(x),x4),1)
    return x

####In this case, we want to transform the result steepest ascent is getting to  the right domain of W
def transformationDomainW(w):
    return np.round(w)

def estimationObjective(x):
    estimator=1000
    W=simulatorW(estimator)
    result=np.zeros(estimator)
    for i in range(estimator):
        result[i]=g(TimeHours,W[i,:],x,nSets,lamb,A,"2014-05")
    
    return np.mean(result),float(np.var(result))/estimator

##W is a matrix

def computeLogProductExpectationsForAn(W,N):
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha**2
    logproductExpectations=np.zeros(N)
    parameterLamb=parameterSetsPoisson
    for i in xrange(N):
        logproductExpectations[i]=0.0
        for j in xrange(n2):
            G=poisson(parameterLamb[j])
            temp=G.dist.expect(lambda z: np.exp(-alpha2[j]*((z-W[i,j])**2)),G.args)
            logproductExpectations[i]+=np.log(temp)
    return logproductExpectations





l={}
l['computeLogProductExpectationsForAn']=computeLogProductExpectationsForAn
l['parallel']=parallel
l['folderContainerResults']=os.path.join(directory[0],"SBO")
l['estimationObjective']=estimationObjective
l['transformationDomainW']=transformationDomainW
l['transformationDomainX']=transformationDomainX
l['dimXsteepest']=n1-1
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
l['numberParallel']=1
l['scaledAlpha']=scaleAlpha
l['xtol']=1.0

def conditionOpt(x):
    return np.max((np.floor(np.abs(x))))
l['functionConditionOpt']=conditionOpt
print 'ok'
sboObj=SB.SBO(**l)



def optimizeVOI(sboObj,start, i):
    opt=op.OptSteepestDescent(n1=sboObj.dimXsteepest,projectGradient=sboObj.projectGradient,stopFunction=sboObj.functionConditionOpt,xStart=start,xtol=sboObj.xtol)
    opt.constraintA=sboObj._constraintA
    opt.constraintB=sboObj._constraintB
    tempN=sboObj.numberTraining+i
    A=sboObj._VOI._k.A(sboObj._VOI._Xhist[0:tempN,:],noise=sboObj._VOI._noiseHist[0:tempN])
    L=np.linalg.cholesky(A)
  #  self.functionGradientAscentAn
    def g(x,grad,onlyGradient=False):
        return sboObj.functionGradientAscentVn(x,grad,sboObj,i,L,onlyGradient=onlyGradient)

        #temp=self._VOI.VOIfunc(i,x,grad=grad)
        #if grad==True:
        #    return temp[0],temp[1]
        #else:
        #    return temp
    opt.run(f=g)
    sboObj.optRuns.append(opt)
    xTrans=sboObj.transformationDomainX(opt.xOpt[0:1,0:sboObj.dimXsteepest])
    sboObj.optPointsArray.append(xTrans)
    
def optVOInoParal(sboObj,i):
    n1=sboObj._n1
    n2=sboObj._dimW
    Xst=sboObj.sampleFromX(1)
    wSt=sboObj._simulatorW(1)
    x1=Xst[0:0+1,:]
    w1=wSt[0:0+1,:]
    st=np.concatenate((x1,w1),1)
    args2={}
    args2['start']=st
    args2['i']=i
       # misc.VOIOptWrapper(self,**args2)
    sboObj.optRuns.append(optimizeVOI(sboObj,**args2))
    j=0
    temp=sboObj.optRuns[j].xOpt
    gradOpt=sboObj.optRuns[j].gradOpt
    numberIterations=sboObj.optRuns[j].nIterations
    gradOpt=np.sqrt(np.sum(gradOpt**2))
    gradOpt=np.array([gradOpt,numberIterations])
    xTrans=sboObj.transformationDomainX(sboObj.optRuns[j].xOpt[0:1,0:sboObj.dimXsteepest])
    wTrans=sboObj.transformationDomainW(sboObj.optRuns[j].xOpt[0:1,sboObj.dimXsteepest:sboObj.dimXsteepest+sboObj._dimW])
    ###falta transformar W
    temp=np.concatenate((xTrans,wTrans),1)
    print temp






optVOInoParal(sboObj,0)





