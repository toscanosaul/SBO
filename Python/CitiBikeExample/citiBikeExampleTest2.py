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

try:
    nTemp=int(sys.argv[1])
    nTemp2=int(sys.argv[2])
    nTemp3=int(sys.argv[3])
except:
    nTemp=1
    nTemp2=10
    nTemp3=10

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
####borrar
def noisyF2(XW,n,seed):
    np.random.seed(seed)
    return np.random.uniform(0,1)
######borrar



#####work this
def simulatorW(n):
    wPrior=np.zeros((n,n2))
    for i in range(n2):
        wPrior[:,i]=np.random.poisson(parameterSetsPoisson[i],n)
    return wPrior


def euclidean_proj_simplex(v, s=numberBikes):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def euclidean_proj_l1ball(v, s=numberBikes):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w

###Used to select a starting point for gradient ascent
def sampleFromX(n):
    temp=np.random.dirichlet(np.ones(n1),n)
    temp=temp[:,0:n1-1]
    temp=numberBikes*temp
    temp=np.floor(temp)
    return temp


####Prior Data
#randomIndexes=np.random.random_integers(0,pointsVOI.shape[0]-1,trainingPoints)
tempX=sampleFromX(trainingPoints)
tempFour=numberBikes-np.sum(tempX,1)
tempFour=tempFour.reshape((trainingPoints,1))
Xtrain=np.concatenate((tempX,tempFour),1)
Wtrain=simulatorW(trainingPoints)
XWtrain=np.concatenate((Xtrain,Wtrain),1)



###################################################

yTrain=np.zeros([0,1])
NoiseTrain=np.zeros(0)


#jobs = []
#pool = mp.Pool()
#for i in xrange(trainingPoints):
#    job = pool.apply_async(noisyF,(XWtrain[i,:].reshape((1,n1+n2)),numberSamplesForF))
#    jobs.append(job)

#pool.close()  # signal that no more data coming in
#pool.join()  # wait for all the tasks to complete
#for j in range(trainingPoints):
#    temp=jobs[j].get()
#    yTrain=np.vstack([yTrain,temp[0]])
#    NoiseTrain=np.append(NoiseTrain,temp[1])
directory=[]

directory.append(os.path.join("..","CitiBikeExample","Results100AveragingSamples20TrainingPoints","SBO","1run"))
ind=1
def readTrainingData(n,directory):
    XWtrain=np.loadtxt(os.path.join(directory,"%d"%ind+"XWHist.txt"))[0:n,:]
    yTrain=np.loadtxt(os.path.join(directory,"%d"%ind+"yhist.txt"))[0:n]
    yTrain=yTrain.reshape((n,1))
    NoiseTrain=np.loadtxt(os.path.join(directory,"%d"%ind+"varHist.txt"))[0:n]
    return XWtrain,yTrain,NoiseTrain

XWtrain,yTrain,NoiseTrain=readTrainingData(20,directory[0])

#########

scaleAlpha=1000.0
kernel=SK.SEK(n1+n2,X=XWtrain,y=yTrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)

#########

logFactorial=[np.sum([log(i) for i in range(1,j+1)]) for j in range(1,501)]
logFactorial.insert(1,0)
logFactorial=np.array(logFactorial)

#Computes log*sum(exp(x)) for a vector x, but in numerically careful way
def logSumExp(x):
    xmax=np.max(np.abs(x))
    y=xmax+np.log(np.sum(np.exp(x-xmax)))
    return y


def B(x,XW,n1,n2):
    x=np.array(x).reshape((x.shape[0],n1))
    results=np.zeros(x.shape[0])
    parameterLamb=parameterSetsPoisson
    X=XW[0:n1]
    W=XW[n1:n1+n2]
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha**2
    alpha1=0.5*((kernel.alpha[0:n1])**2)/scaleAlpha**2
    variance0=kernel.variance
    
    logproductExpectations=0.0
    print "veamos B"
    print "x is"
    print x
    print "parameters"
    print parameterLamb
    print "X"
    print X
    print "W"
    print W
    print "alphas,variance"
    print alpha1,alpha2,variance0

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
def projectGradientDescent(x):
   # z=x[0:n1-1]
   # z=np.abs(z)
   # y=euclidean_proj_simplex(z)
   # y=np.floor(y)
    x=np.abs(x)   
    return euclidean_proj_l1ball(x)

####we eliminate one variable to optimize the function
def functionGradientAscentVn(x,grad,SBO,i):
    x4=np.array(numberBikes-np.sum(x[0,0:n1-1])).reshape((1,1))
    tempX=x[0:1,0:n1-1]
    x2=np.concatenate((tempX,x4),1)
    tempW=x[0:1,n1-1:n1-1+n2]
    xFinal=np.concatenate((x2,tempW),1)
    temp=SBO._VOI.VOIfunc(i,xFinal,grad=grad)
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
def functionGradientAscentAn(x,grad,SBO,i,L):
    x4=np.array(numberBikes-np.sum(x)).reshape((1,1))
    x=np.concatenate((x,x4),1)
    temp=SBO._VOI._GP.aN_grad(x,L,i,grad)
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
    x4=np.array(numberBikes-np.sum(x)).reshape((1,1))
    x=np.concatenate((x,x4),1)
    return np.floor(x)

####In this case, we want to transform the result steepest ascent is getting to  the right domain of W
def transformationDomainW(w):
    return np.round(w)

def estimationObjective(x):
    estimator=100
    W=simulatorW(estimator)
    result=np.zeros(estimator)
    for i in range(estimator):
        result[i]=g(TimeHours,W[i,:],x,nSets,lamb,A,"2014-05")
    
    return np.mean(result),float(np.var(result))/estimator





nameDirectory="Results"+'%d'%numberSamplesForF+"AveragingSamples"+'%d'%trainingPoints+"TrainingPoints"
l={}
l['folderContainerResults']=os.path.join(nameDirectory,"SBO")
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
l['scaledAlpha']=100.0

print 'ok'
sboObj=SB.SBO(**l)
print "TRAININNNNNGGGGG"
sboObj.trainModel(numStarts=10)
print "alpha,variance,mu"
print kernel.alpha
print kernel.variance
print kernel.mu
print 'ok2'
tempN=20
#print "points"
#print sboObj._XWhist
#print sboObj._varianceObservations
#print sboObj._yHist
#print "points end"
A=sboObj._k.A(sboObj._XWhist[0:tempN,:],noise=sboObj._varianceObservations[0:tempN])
L=np.linalg.cholesky(A)
tempX=np.array([[ 905.60177805 , 131.2408212, 1053.83098478]])
#print functionGradientAscentAn(tempX,True,sboObj,0,L)
#print sboObj.optimizeAn(tempX,0)
#sboObj.SBOAlg(1,nRepeat=1,Train=True)
tempFour=numberBikes-np.sum(tempX,1)
tempFour=tempFour.reshape((1,1))
Xtrain=np.concatenate((tempX,tempFour),1)
print "B is"
print B(Xtrain,sboObj._VOI._GP._Xhist[0,:],n1,n2)

