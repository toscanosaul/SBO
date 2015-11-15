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
import SquaredExponentialKernel as SK
from grid import *
import SBOGeneral2 as SB
import VOIGeneral as VOI
import statGeneral as stat
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
import InterfaceSBO as inter

##################

nTemp=int(sys.argv[1])  #random seed 
nTemp2=int(sys.argv[2]) #number of training points
nTemp3=int(sys.argv[3]) #number of samples to estimate F

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
    aux1=(numberBikes/float(n1))*np.ones((1,n1-1))
    if n>1:
        temp=np.random.dirichlet(np.ones(n1),n-1)
	temp=(numberBikes-500.0*n1)*temp+500.0
    	temp=temp[:,0:n1-1]
    	temp=np.floor(temp)
	aux1=np.concatenate((aux1,temp),0)
    return aux1

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
parallel=False

trainingPoints=nTemp2

nameDirectory="ResultsTest"+'%d'%numberSamplesForF+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"
folder=os.path.join(nameDirectory,"SBO")

misc=inter.Miscellaneous(randomSeed,parallel,folder,True)

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
Wtrain=simulatorW(trainingPoints)
XWtrain=np.concatenate((Xtrain,Wtrain),1)

yTrain=np.zeros([0,1])
NoiseTrain=np.zeros(0)

if parallel:
    jobs = []
    pool = mp.Pool()
    for i in xrange(trainingPoints):
        job = pool.apply_async(noisyF,(XWtrain[i,:].reshape((1,n1+n2)),numberSamplesForF))
        jobs.append(job)
    pool.close()  
    pool.join()  
    for j in range(trainingPoints):
        temp=jobs[j].get()
        yTrain=np.vstack([yTrain,temp[0]])
        NoiseTrain=np.append(NoiseTrain,temp[1])
else:
    for i in xrange(trainingPoints):
        temp=noisyF(XWtrain[i,:].reshape((1,n1+n2)),numberSamplesForF)
        yTrain=np.vstack([yTrain,temp[0]])
        NoiseTrain=np.append(NoiseTrain,temp[1])

dataObj=inter.data(XWtrain,yTrain,NoiseTrain)

"""
We define the statistical object.
"""

dimensionKernel=n1+n2
scaleAlpha=1000.0
kernel=SK.SEK(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)

def B(x,XW,n1,n2,logproductExpectations=None):
    """Computes B(x)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
      
       Args:
          x: Vector of points where B is evaluated
          XW: Point (x,w)
          n1: Dimension of x
          n2: Dimension of w
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

###The function is the same for any squared exponential kernel
def gradXBforAn(x,n,B,kern,X):
    """Computes the gradient of B(x,i) for i in {1,...,n+nTraining}
       where nTraining is the number of training points
      
       Args:
          x: Argument of B
          n: Current iteration of the algorithm
          B: Vector {B(x,i)} for i in {1,...,n}
          kern: kernel
          X: Past observations X[i,:] for i in {1,..,n+nTraining}
    """
    gradXB=np.zeros((n1,n+trainingPoints))
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
    for i in xrange(n+trainingPoints):
        gradXB[:,i]=B[i]*(-2.0*alpha1*(x-X[i,:]))
    return gradXB

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

stat=stat.SBOGP(kernel=kernel,B=B,dimNoiseW=n2,dimPoints=n1,
                dimKernel=n1+n2, numberTraining=trainingPoints,
                gradXBforAn=gradXBforAn, computeLogProductExpectationsForAn=
                computeLogProductExpectationsForAn,scaledAlpha=scaleAlpha)


"""
We define the VOI object.
"""

pointsVOI=np.loadtxt("pointsPoisson.txt") #Discretization of the domain of X

####The function is the same for any squared exponential kernel
def gradXWSigmaOfunc(n,new,kern,Xtrain2,Wtrain2):
    """Computes the vector of the gradients of Sigma_{0}(new,XW[i,:]) for
        all the past observations XW[i,]. Sigma_{0} is the covariance of
        the GP on F.
        
       Args:
          n: Number of iteration
          new: Point where Sigma_{0} is evaluated
          kern: Kernel
          Xtrain2: Past observations of X
          Wtrain2: Past observations of W
          N: Number of observations
    """
    gradXSigma0=np.zeros([n+trainingPoints+1,n1])
    tempN=n+trainingPoints
    past=np.concatenate((Xtrain2,Wtrain2),1)
    gamma=np.transpose(kern.A(new,past))
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
    gradWSigma0=np.zeros([n+trainingPoints+1,n2])
    alpha2=0.5*((kern.alpha[n1:n1+n2])**2)/scaleAlpha**2
    xNew=new[0,0:n1]
    wNew=new[0,n1:n1+n2]
    for i in xrange(n+trainingPoints):
        gradXSigma0[i,:]=-2.0*gamma[i]*alpha1*(xNew-Xtrain2[i,:])
        gradWSigma0[i,:]=-2.0*gamma[i]*alpha2*(wNew-Wtrain2[i,:])
    return gradXSigma0,gradWSigma0


###The function is the same for any squared exponential kernel
def gradXB(new,kern,BN,keep,points):
    """Computes the vector of gradients with respect to x_{n+1} of
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
    xNew=new[0,0:n1].reshape((1,n1))
    gradXBarray=np.zeros([len(keep),n1])
    M=len(keep)
    for i in xrange(n1):
        for j in xrange(M):
            gradXBarray[j,i]=-2.0*alpha1[i]*BN[keep[j],0]*(xNew[0,i]-points[keep[j],i])
    return gradXBarray

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

VOIobj=VOI.VOISBO(gradXWSigmaOfunc=gradXWSigmaOfunc,dimX=n1,
                  pointsApproximation=pointsVOI,gradXBfunc=gradXB,
                  gradWBfunc=gradWB,dimW=n2,numberTraining=trainingPoints)


"""
We define the Opt object.
"""

dimXsteepest=n1-1 #Dimension of x when the VOI and a_{n} are optimized.

def projectGradientDescent(x,direction,xo):
    """ Project a point x to its domain (which is the simplex)
        at each step of the gradient ascent method if needed.
        
       Args:
          x: Point that is projected
          direction: Gradient of the function at xo
          xo: Starting point at the iteration of the gradient ascent method
    """
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

def functionGradientAscentVn(x,grad,VOI,i,L,temp2,a,kern,XW,scratch,Bfunc,onlyGradient=False):
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
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    tempX=x[0:1,0:n1-1]
    x2=np.concatenate((tempX,x4),1)
    tempW=x[0:1,n1-1:n1-1+n2]
    xFinal=np.concatenate((x2,tempW),1)
    temp=VOI.VOIfunc(i,xFinal,L=L,temp2=temp2,a=a,grad=grad,scratch=scratch,onlyGradient=onlyGradient,
                          kern=kern,XW=XW,B=Bfunc)

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
    x4=np.array(numberBikes-np.sum(x)).reshape((1,1))
    x=np.concatenate((x,x4),1)
    if onlyGradient:
        temp=stat.aN_grad(x,L,i,dataObj,grad,onlyGradient,logproductExpectations)
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L2=np.concatenate((t,s))
        grad2=np.dot(temp,L2)
        return grad2

    temp=stat.aN_grad(x,L,i,dataObj,gradient=grad,logproductExpectations=logproductExpectations)
    if grad==False:
        return temp
    else:
        t=np.diag(np.ones(n1-1))
        s=-1.0*np.ones((1,n1-1))
        L2=np.concatenate((t,s))
        grad2=np.dot(temp[1],L2)
        return temp[0],grad2

def transformationDomainX(x):
    """ Transforms the point x given by the steepest ascent method to
        the right domain of x.
        
       Args:
          x: Point to be transformed.
    """
    x4=np.array(numberBikes-np.sum(np.floor(x))).reshape((1,1))
    x=np.concatenate((np.floor(x),x4),1)
    return x

def transformationDomainW(w):
    """ Transforms the point w given by the steepest ascent method to
        the right domain of w.
        
       Args:
          w: Point to be transformed.
    """
    return np.round(w)

def conditionOpt(x):
    """ Gives the stopping rule for the steepest ascent method, e.g.
        the function could be the Euclidean norm. 
        
       Args:
          x: Point where the condition is evaluated.
    """
    return np.max((np.floor(np.abs(x))))

opt=inter.opt(3,dimXsteepest,transformationDomainX,transformationDomainW,projectGradientDescent,functionGradientAscentVn,
              functionGradientAscentAn,conditionOpt,1.0)


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


print 'ok'
sboObj=SB.SBO(**l)
print 'ok2'

"""
We run the SBO algorithm.
"""

sboObj.SBOAlg(2,nRepeat=10,Train=True)


