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

##################

nTemp=int(sys.argv[1])  #random seed 
nTemp2=int(sys.argv[2]) #number of training points
nTemp3=int(sys.argv[3]) #number of samples to estimate F
nTemp4=int(sys.argv[4]) #number of iterations
nTemp5=sys.argv[5] #True if code is run in parallel; False otherwise.
nTemp6=int(sys.argv[6]) #number of restarts for the optimization method

if nTemp5=='F':
    nTemp5=False
    nTemp6=1
elif nTemp5=='T':
    nTemp5=True
    
betah=float(sys.argv[7])
Aparam=float(sys.argv[8])

alphah=Aparam/float(nTemp3)
alphad=(1.0/float(nTemp3))-alphah



print "random seed is "
print nTemp

randomSeed=nTemp
np.random.seed(randomSeed)

##############

n1=1
n2=1
numberSamplesForF=nTemp3


"""
We define the objective object.
"""

def sigmah(xw,xw2):
    return alphah*np.exp(-betah*np.sqrt(np.sum((xw-xw2)**2)))
    
#xw is matrix where each row is a point where h is evaluated
def covarianceh(xw):
    cov=np.zeros((xw.shape[0],xw.shape[0]))
    for i in xrange(xw.shape[0]):
        for j in xrange(i,xw.shape[0]):
            cov[i,j]=sigmah(xw[i,:],xw[j,:])
            cov[j,i]=cov[i,j]
    return cov

#xw is matrix where each row is a point where h is evaluated
def h2(xw,L=None):
    if L is None:
        cov=covarianceh(xw)
        L=np.linalg.cholesky(cov)

    
    Z=np.random.normal(0,1,xw.shape[0])

    return np.dot(L,Z)

######cambiar ngrid

######Define the function
ngrid=50
domainX=np.linspace(0,1,ngrid)
domainW=np.linspace(0,1,ngrid)


defineFunction=sys.argv[9]

if defineFunction=='F':
    defineFunction=False
else:
    defineFunction=True

if defineFunction:
    z=np.zeros((ngrid*ngrid,2))
    
    for i in range(ngrid):
        for j in range(ngrid):
            z[i*ngrid+j,0]=domainX[i]
            z[i*ngrid+j,1]=domainW[j]
    
    output=h2(z)
    
    f=open(os.path.join("functions","function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%nTemp3+".txt",'w'))
    np.savetxt(f,output)
    f.close()
    
    valuesOutput=np.zeros(ngrid)
    
    for j in xrange(ngrid):
        results=np.zeros(ngrid)
        for i in xrange(ngrid):
            results[i]=output[j*ngrid+i]
        valuesOutput[j]=np.mean(results)
    
    f=open(os.path.join("functions","valuesof"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%nTemp3+".txt",'w'))
    np.savetxt(f,valuesOutput)
    f.close()

    
 #   f=open("noise"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+".txt",'w')
 #   noisy=np.random.normal(0,np.sqrt(alphad),(ngrid*ngrid,ngrid))
  #  np.savetxt(f,noisy)
  #  f.close()
else:
    output=np.loadtxt(os.path.join("functions","function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%nTemp3+".txt"))
  #  noisy=np.loadtxt("noise"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+".txt")


###########

def getindex(x):
    dx=1.0/(ngrid-1)
    i=round(x/dx)
    return i

#k is already index
def evalf(x,w):
    i=getindex(x)
    j=getindex(w)
    h1=output[i*ngrid+j]
    #h1+noisy[i*ngrid+j,k]
    return h1+np.random.normal(0,np.sqrt(alphad),1)

def evalf2(x,j,k):
    i=getindex(x)
    h1=output[i*ngrid+j]
    return h1+noisy[i*ngrid*ngrid+j*ngrid+k]
    

def simulateZ(n):
    l=np.random.randint(0,ngrid,n)
    return l
    

def noisyF(XW,n):
    """Estimate F(x,w)=E(f(x,w,z)|w)
      
       Args:
          XW: Vector (x,w)
          n: Number of samples to estimate F
    """
    
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]
    #z=simulateZ(n)
    res=np.zeros(n)
    for i in xrange(n):
        res[i]=evalf(x,w)
        ##alphad/n
    return np.mean(res),alphad/n

lowerX=0
upperX=1

def sampleFromXAn(n):
    """Chooses n points in the domain of x at random
      
       Args:
          n: Number of points chosen
    """
    return np.random.uniform(lowerX,upperX,(n,1))

sampleFromXVn=sampleFromXAn

lowerW=0
upperW=1

def simulatorW(n):
    """Simulate n vectors w
      
       Args:
          n: Number of vectors simulated
    """
    temp=np.random.randint(0,ngrid,n)
    w=np.zeros((n,1))
    for i in range(n):
        w[i,0]=domainW[temp[i]]
    return w


    



def estimationObjective(x,N=1000):
    """Estimate g(x)=E(f(x,w,z))
      
       Args:
          x
          N: number of samples used to estimate g(x)
    """
    w=simulatorW(N)
 #   z=simulateZ(N)
    j=getindex(x)
    results=np.zeros(ngrid)

    for i in xrange(ngrid):
        results[i]=output[j*ngrid+i]


    return np.mean(results),0


#checar todo, pero parace que hasta aqui casi ya
Objective=inter.objective(None,n1,noisyF,numberSamplesForF,sampleFromXVn,
                          simulatorW,estimationObjective,sampleFromXAn)


"""
We define the miscellaneous object.
"""
parallel=nTemp5

trainingPoints=nTemp2


#nameDirectory="Results"+'%d'%numberSamplesForF+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"
#folder=os.path.join(nameDirectory,"SBO")

misc=inter.Miscellaneous(randomSeed,parallel,nF=numberSamplesForF,tP=trainingPoints,
                         prefix="function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%nTemp3)

"""
We define the data object.
"""

"""
Generate the training data
"""

Xtrain=sampleFromXVn(trainingPoints).reshape((trainingPoints,1))
Wtrain=simulatorW(trainingPoints).reshape((trainingPoints,1))
XWtrain=np.concatenate((Xtrain,Wtrain),1)

dataObj=inter.data(XWtrain,yHist=None,varHist=None)

dataObj.getTrainingDataSBO(trainingPoints,noisyF,numberSamplesForF,parallel)
#dataObj.getTrainingDataSBO(trainingPoints,noisyF,numberSamplesForF,True)


"""
We define the statistical object.
"""

dimensionKernel=n1+n2
scaleAlpha=0.5
#kernel=SK.SEK(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)



def expectation(z,alpha):
    w=domainW
    aux=np.exp(-alpha*((z-w)**2))
    return (1.0/ngrid)*np.sum(aux)

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
    #parameterLamb=parameterSetsPoisson
    X=XW[0:n1]
    inda=n1+n2
    W=XW[n1:inda]
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha**2
    alpha1=0.5*((kernel.alpha[0:n1])**2)/scaleAlpha**2
    variance0=kernel.variance
    if logproductExpectations is None:
        logproductExpectations=0.0
        for j in xrange(n2):
	    temp=expectation(W[j],alpha2[j])
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
  #  parameterLamb=parameterSetsPoisson
    for i in xrange(N):
        logproductExpectations[i]=0.0
        for j in xrange(n2):
	    temp=expectation(W[i,j],alpha2[j])
            logproductExpectations[i]+=np.log(temp)
    return logproductExpectations

stat=stat.SBOGP(B=B,dimNoiseW=n2,dimPoints=n1,trainingData=dataObj,
                dimKernel=n1+n2, numberTraining=trainingPoints,
                computeLogProductExpectationsForAn=
                computeLogProductExpectationsForAn,scaledAlpha=scaleAlpha)


"""
We define the VOI object.
"""

pointsVOI=domainX.reshape((ngrid,1)) #Discretization of the domain of X


def expectation2(z,alpha):
    w=domainW
    aux=-2.0*alpha*(z-w)*np.exp(-alpha*((z-w)**2))
    return np.sum(aux)*(1.0/ngrid)

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
   # parameterLamb=parameterSetsPoisson
    X=new[0,0:n1]
    W=new[0,n1:n1+n2]
   
    for i in xrange(n2):
        logproductExpectations=0.0
        a=range(n2)
        del a[i]
        for r in a:
	    temp=expectation(W[r],alpha2[r])
            logproductExpectations+=np.log(temp)
	temp=expectation2(W[i],alpha2[i])
    #    G=poisson(parameterLamb[i])
    #    temp=G.dist.expect(lambda z: -2.0*alpha2[i]*(-z+W[i])*np.exp(-alpha2[i]*((z-W[i])**2)),G.args)
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

dimXsteepestAn=n1 #Dimension of x when the VOI and a_{n} are optimized.


def functionGradientAscentVn(x,VOI,i,L,temp2,a,kern,XW,scratch,Bfunc,onlyGradient=False,grad=None):
    """ Evaluates the VOI and it can compute its derivative. It evaluates the VOI,
        when grad and onlyGradient are False; it evaluates the VOI and computes its
        derivative when grad is True and onlyGradient is False, and computes only its
        gradient when gradient and onlyGradient are both True.
    
        Args:
            x: VOI is evaluated at (x,numberBikes-sum(x)).Note that we reduce the dimension
               of the space of x.
            grad: True if we want to compute the gradient; False otherwise.
            i: Iteration of the SBO algorithm.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
            Bfunc: Computes B(x,XW)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
            temp2: temp2=inv(L)*B.T, where B is a matrix such that B(i,j) is
                   \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
                   where points x_{p} is a point of the discretization of
                   the space of x; and (x_{j},w_{j}) is a past observation.
            a: Vector of the means of the GP on g(x)=E(f(x,w,z)). The means are evaluated on the
               discretization of the space of x.
            VOI: VOI object
            kern: kernel
            XW: Past observations
            scratch: matrix where scratch[i,:] is the solution of the linear system
                     Ly=B[j,:].transpose() (See above for the definition of B and L)
            onlyGradient: True if we only want to compute the gradient; False otherwise.
    """
    grad=onlyGradient
    x=np.array(x).reshape([1,n1+n2])

    tempX=x[0:1,0:n1]

    tempW=x[0:1,n1:n1+n2]
    xFinal=np.concatenate((tempX,tempW),1)
    temp=VOI.VOIfunc(i,xFinal,L=L,temp2=temp2,a=a,grad=grad,scratch=scratch,onlyGradient=onlyGradient,
                          kern=kern,XW=XW,B=Bfunc)

    

    if onlyGradient:

        return temp
        

    if grad==True:

        return temp[0],temp[1]
    else:

        return temp
    

def functionGradientAscentAn(x,grad,stat,i,L,dataObj,onlyGradient=False,logproductExpectations=None):
    """ Evaluates a_{i} and its derivative, which is the expectation of the GP on g(x).
        It evaluates a_{i}, when grad and onlyGradient are False; it evaluates the a_{i}
        and computes its derivative when grad is True and onlyGradient is False, and
        computes only its gradient when gradient and onlyGradient are both True.
    
        Args:
            x: a_{i} is evaluated at (x,numberBikes-sum(x)).Note that we reduce the dimension
               of the space of x.
            grad: True if we want to compute the gradient; False otherwise.
            i: Iteration of the SBO algorithm.
            L: Cholesky decomposition of the matrix A, where A is the covariance
               matrix of the past obsevations (x,w).
            dataObj: Data object.
            stat: Statistical object.
            onlyGradient: True if we only want to compute the gradient; False otherwise.
            logproductExpectations: Vector with the logarithm of the product of the
                                    expectations of np.exp(-alpha2[j]*((z-W[i,j])**2))
                                    where W[i,:] is a point in the history.
    """
   
    x=np.array(x).reshape([1,n1])

  #  x4=np.array(numberBikes-np.sum(x)).reshape((1,1))
   # x=np.concatenate((x,x4),1)
   
    if onlyGradient:
        temp=stat.aN_grad(x,L,i,dataObj,grad,onlyGradient,logproductExpectations)

        return temp

    temp=stat.aN_grad(x,L,i,dataObj,gradient=grad,logproductExpectations=logproductExpectations)
    if grad==False:
        return temp
    else:
        return temp[0],temp[1]

lower=0


def const2(x):
    return x[0]-lower

def jac2(x):
    return np.array([1,0])

def const3(x):
    return x[1]-lower

def jac3(x):
    return np.array([0,1])

upper=1

def const6(x):
    return upper-x[0]

def jac6(x):
    return np.array([-1,0])

def const7(x):
    return upper-x[1]

def jac7(x):
    return np.array([0,-1])


cons=({'type':'ineq',
        'fun': const2,
       'jac': jac2},
    {'type':'ineq',
        'fun': const3,
       'jac': jac3},
        {'type':'ineq',
        'fun': const6,
       'jac': jac6},
        {'type':'ineq',
        'fun': const7,
       'jac': jac7})


def transformationDomainXAn(x):
    """ Transforms the point x given by the steepest ascent method to
        the right domain of x.
        
       Args:
          x: Point to be transformed.
    """
    return x

transformationDomainXVn=transformationDomainXAn

def transformationDomainW(w):
    """ Transforms the point w given by the steepest ascent method to
        the right domain of w.
        
       Args:
          w: Point to be transformed.
    """
    return w




def const2A(x):
    return x[0]-lower

def jac2A(x):
    return np.array([1])


def const6A(x):
    return upper-x[0]

def jac6A(x):
    return np.array([-1])


consA=({'type':'ineq',
        'fun': const2A,
       'jac': jac2A},
        {'type':'ineq',
        'fun': const6A,
       'jac': jac6A})

opt=inter.opt(nTemp6,n1,n1,transformationDomainXVn,transformationDomainXAn,
              transformationDomainW,None,functionGradientAscentVn,
              functionGradientAscentAn,None,1.0,cons,consA,"SLSQP","SLSQP")


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

