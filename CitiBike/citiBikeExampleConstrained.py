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

To use the SBO algorithm, we need to create 6 objets:

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
import json
from BGO.Source import *

##################

nTemp=int(sys.argv[1])  #random seed 
nTemp2=int(sys.argv[2]) #number of training points
nTemp3=int(sys.argv[3]) #number of samples to estimate F
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

##############

n1=4
n2=4
numberSamplesForF=nTemp3


"""
We define the variables needed for the queuing simulation. 
"""

g=unhappyPeople  #Simulator

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

def noisyF(XW,n):
    """Estimate F(x,w)=E(f(x,w,z)|w)
      
       Args:
          XW: Vector (x,w)
          n: Number of samples to estimate F
    """
    simulations=np.zeros(n)
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]
    for i in xrange(n):
        simulations[i]=g(TimeHours,w,x,nSets,lamb,A,"2014-05",exponentialTimes,
                         data,cluster,bikeData)
    return np.mean(simulations),float(np.var(simulations))/n

def sampleFromX(n):
    """Chooses n points in the domain of x at random
      
       Args:
          n: Number of points chosen
    """
 #   aux1=(numberBikes/float(n1))*np.ones((1,n1))
    if n>=1:
        temp=np.random.dirichlet(np.ones(n1),n)
	temp=(numberBikes-500.0*n1)*temp+500.0
        temp[:,0:n1-1]=np.floor(temp[:,0:n1-1])
    	temp[:,n1-1]=numberBikes-np.sum(temp[:,0:n1-1],1)
    #	temp=np.floor(temp)
     #   temp=np.concatenate((temp,numberBikes-np.sum(temp)))
	#aux1=np.concatenate((aux1,temp),0)
    return temp

def simulatorW(n):
    """Simulate n vectors w
      
       Args:
          n: Number of vectors simulated
    """
    wPrior=np.zeros((n,n2))
    for i in range(n2):
        wPrior[:,i]=np.random.poisson(parameterSetsPoisson[i],n)
    return wPrior

def estimationObjective(x,N=100):
    """Estimate g(x)=E(f(x,w,z))
      
       Args:
          x
          N: number of samples used to estimate g(x)
    """
    estimator=N
    W=simulatorW(estimator)
    result=np.zeros(estimator)
    for i in range(estimator):
        result[i]=g(TimeHours,W[i,:],x,nSets,lamb,A,"2014-05",exponentialTimes,
                         data,cluster,bikeData)
    
    return np.mean(result),float(np.var(result))/estimator

Objective=inter.objective(g,n1,noisyF,numberSamplesForF,sampleFromX,
                          simulatorW,estimationObjective)

"""
We define the miscellaneous object.
"""
parallel=nTemp5

trainingPoints=nTemp2


#nameDirectory="Results"+'%d'%numberSamplesForF+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"
#folder=os.path.join(nameDirectory,"SBO")

misc=inter.Miscellaneous(randomSeed,parallel,nF=numberSamplesForF,tP=trainingPoints)

"""
We define the data object.
"""

"""
Generate the training data
"""

tempX=sampleFromX(trainingPoints)
#tempFour=numberBikes-np.sum(tempX,1)
#tempFour=tempFour.reshape((trainingPoints,1))
#Xtrain=np.concatenate((tempX,tempFour),1)
Wtrain=simulatorW(trainingPoints)
XWtrain=np.concatenate((tempX,Wtrain),1)

dataObj=inter.data(XWtrain,yHist=None,varHist=None)
dataObj.getTrainingDataSBO(trainingPoints,noisyF,numberSamplesForF,parallel)

"""
We define the statistical object.
"""

dimensionKernel=n1+n2
scaleAlpha=1000.0
#kernel=SK.SEK(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)

def B(x,XW,n1,n2,kernel,logproductExpectations=None):
    """Computes B(x)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
      
       Args:
          x: Vector of points where B is evaluated
          XW: Point (x,w)
          n1: Dimension of x
          n2: Dimension of w
          kernel
          logproductExpectations: Vector with the logarithm
                                  of the product of the
                                  expectations of
                                  np.exp(-alpha2[j]*((z-W[i,j])**2))
                                  where W[i,:] is a point in the history.
          
    """
    x=np.array(x).reshape((x.shape[0],n1))
    results=np.zeros(x.shape[0])
    parameterLamb=parameterSetsPoisson
    X=XW[0:n1]
    inda=n1+n2
    W=XW[n1:inda]
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

def computeLogProductExpectationsForAn(W,N,kernel):
    """Computes the logarithm of the product of the
       expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
        where W[i,:] is a point in the history.
      
       Args:
          W: Matrix where each row is a past random vector used W[i,:]
          N: Number of observations
          kernel: kernel
    """
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

stat=stat.SBOGP(B=B,dimNoiseW=n2,dimPoints=n1,trainingData=dataObj,
                dimKernel=n1+n2, numberTraining=trainingPoints,
                computeLogProductExpectationsForAn=
                computeLogProductExpectationsForAn,scaledAlpha=scaleAlpha)


"""
We define the VOI object.
"""

pointsVOI=np.loadtxt("pointsPoisson.txt") #Discretization of the domain of X

def gradWB(new,kern,BN,keep,points):
    """Computes the vector of gradients with respect to w_{n+1} of
	B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
	where x_{p} is a point in the discretization of the domain of x.
        
       Args:
          new: Point (x_{n+1},w_{n+1})
          kern: Kernel
          keep: Indexes of the points keeped of the discretization of the domain of x,
                after using AffineBreakPoints
          BN: Vector B(x_{p},n+1), where x_{p} is a point in the discretization of
              the domain of x.
          points: Discretization of the domain of x
    """
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)/scaleAlpha**2
    variance0=kern.variance
    wNew=new[0,n1:n1+n2].reshape((1,n2))
    gradWBarray=np.zeros([len(keep),n2])
    M=len(keep)
    parameterLamb=parameterSetsPoisson
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

VOIobj=VOI.VOISBO(dimX=n1, pointsApproximation=pointsVOI,
                  gradWBfunc=gradWB,dimW=n2,
                  numberTraining=trainingPoints)


"""
We define the Opt object.
"""
def functionGradientAscentVn(x,i,VOI,L,temp2,a,kern,XW,scratch,Bfunc,onlyGradient=False):
    temp=VOI.VOIfunc(i,x,L=L,temp2=temp2,a=a,grad=onlyGradient,scratch=scratch,
                     onlyGradient=onlyGradient,kern=kern,XW=XW,B=Bfunc)
    return temp
    

def functionGradientAscentAn(x,L,i,dataObj,onlyGrad,logproductExpectations=None):
    temp=stat.aN_grad(x,L,i,dataObj,gradient=onlyGrad,onlyGradient=onlyGrad,
                      logproductExpectations=logproductExpectations)
    return temp

def const(x):
    return np.sum(x[0:n1])-numberBikes

def jac(x):
    return np.array([1,1,1,1,0,0,0,0])

cons=({'type':'eq',
#      'fun':lambda x:np.sum(x[0:n1])-numberBikes,
        'fun': const,
       'jac': jac})


def transformationDomainX(x):
    """ Transforms the point x given by the steepest ascent method to
        the right domain of x.
        
       Args:
          x: Point to be transformed.
    """
    x4=np.array(numberBikes-np.sum(np.floor(x[0:1,0:n1-1]))).reshape((1,1))
    x=np.concatenate((np.floor(x[0:1,0:n1-1]),x4),1)
    return x

def transformationDomainW(w):
    """ Transforms the point w given by the steepest ascent method to
        the right domain of w.
        
       Args:
          w: Point to be transformed.
    """
    return np.round(w)

opt=inter.opt(nTemp6,n1,transformationDomainX,transformationDomainW,None,functionGradientAscentVn,
              functionGradientAscentAn,None,None,cons)


"""
We define the SBO object.
"""
l={}
l['VOIobj']=VOIobj
l['Objobj']=Objective
l['miscObj']=misc
l['optObj']=opt
l['statObj']=stat
l['dataObj']=dataObj


sboObj=SBO.SBO(**l)


"""
We run the SBO algorithm.
"""

sboObj.SBOAlg(nTemp4,nRepeat=10,Train=True)


