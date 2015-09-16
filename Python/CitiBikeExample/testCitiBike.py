import numpy as np
import sys
import os
sys.path.append("..")

import numpy as np
import SquaredExponentialKernel as SK
from grid import *
import SBOGeneral2 as SB
from simulationPoissonProcess import *
from math import *
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson
import misc




directory=[]

directory.append(os.path.join("..","CitiBikeExample","Results100AveragingSamples20TrainingPoints","SBO","1run"))

#directory.append(os.path.join("..","CitiBikeExample","Results100AveragingSamples50TrainingPoints","SBO","23run"))

#directory.append(os.path.join("..","CitiBikeExample","Results100AveragingSamples50TrainingPoints","SBO","84run"))



randomSeed=1
np.random.seed(randomSeed)
numberSamplesForF=100
numberPoints=20 #this is the iteration n
trainingPoints=20
g=unhappyPeople

n1=4
n2=4
fil="2014-05PoissonParameters.txt"
nSets=4
A,lamb=generateSets(nSets,fil)
#####
parameterSetsPoisson=np.zeros(n2)
for j in xrange(n2):
    parameterSetsPoisson[j]=np.sum(lamb[j])
####

TimeHours=4.0
trainingPoints=20
numberBikes=6000
lowerX=100*np.ones(4)
UpperX=numberBikes*np.ones(4)
dimensionKernel=n1+n2
nGrid=50
####to generate points run poissonGeneratePoints.py
pointsVOI=np.loadtxt("pointsPoisson.txt")

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

ind=1
def readTrainingData(n,directory):
    XWtrain=np.loadtxt(os.path.join(directory,"%d"%ind+"XWHist.txt"))[0:n,:]
    yTrain=np.loadtxt(os.path.join(directory,"%d"%ind+"yhist.txt"))[0:n]
    yTrain=yTrain.reshape((n,1))
    NoiseTrain=np.loadtxt(os.path.join(directory,"%d"%ind+"varHist.txt"))[0:n]
    return XWtrain,yTrain,NoiseTrain

XWtrain,yTrain,NoiseTrain=readTrainingData(20,directory[0])

def kernelFunc(Xtrain,ytrain,noiseTrain,scaleAlpha):
    return SK.SEK(n1+n2,X=Xtrain,y=ytrain[:,0],noise=NoiseTrain,scaleAlpha=scaleAlpha)


kernel=kernelFunc(XWtrain,yTrain,NoiseTrain,1000.0)
logFactorial=[np.sum([log(i) for i in range(1,j+1)]) for j in range(1,501)]
logFactorial.insert(1,0)
logFactorial=np.array(logFactorial)

#Computes log*sum(exp(x)) for a vector x, but in numerically careful way
def logSumExp(x):
    xmax=np.max(np.abs(x))
    y=xmax+np.log(np.sum(np.exp(x-xmax)))
    return y


def B(x,XW,n1,n2,scaleAlpha):
    x=np.array(x).reshape((x.shape[0],n1))
    results=np.zeros(x.shape[0])
    parameterLamb=parameterSetsPoisson
    X=XW[0:n1]
    W=XW[n1:n1+n2]
    alpha2=0.5*((kernel.alpha[n1:n1+n2])**2)/scaleAlpha**2
    alpha1=0.5*((kernel.alpha[0:n1])**2)/scaleAlpha**2
    variance0=kernel.variance
    
    logproductExpectations=0.0
    for j in xrange(n2):
        G=poisson(parameterLamb[j])
        temp=G.dist.expect(lambda z: np.exp(-alpha2[j]*((z-W[j])**2)),G.args)
        logproductExpectations+=np.log(temp)
    for i in xrange(x.shape[0]):
        results[i]=logproductExpectations+np.log(variance0)-np.sum(alpha1*((x[i,:]-X)**2))
    return np.exp(results)



####this function is the same for any squared exponential kernel
def gradXWSigmaOfunc(n,new,objVOI,Xtrain2,Wtrain2,scaleAlpha):
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
def gradXB(new,objVOI,BN,keep,scaleAlpha):
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



def gradWB(new,objVOI,BN,keep,scaleAlpha):
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
def gradXBforAn(x,n,B,objGP,X,scaleAlpha):
    gradXB=np.zeros((n1,n+trainingPoints))
    kern=objGP._k
    alpha1=0.5*((kern.alpha[0:n1])**2)/scaleAlpha**2
    for i in xrange(n+trainingPoints):
        gradXB[:,i]=B[i]*(-2.0*alpha1*(x-X[i,:]))
    return gradXB


###at eatch steept of the gradient ascent method, it will project the point using this function
def projectGradientDescent(x):
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
        print "Derivativitve of SBO"
        print temp[1]
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


def noisyF(XW,n):
    simulations=np.zeros(n)
    x=XW[0,0:n1]
    w=XW[0,n1:n1+n2]
    for i in xrange(n):
        simulations[i]=g(TimeHours,w,x,nSets,lamb,A,"2014-05")
    return np.mean(simulations),float(np.var(simulations))/n

def readKernelParam(file1,dimKernel):
    f=open(file1, 'r')
    v=f.read()
    val=v.split(':')
    temp=val[1].split(",")
    alpha=bike.np.zeros(dimKernel)
    alpha[0]=float(temp[0].split("[")[1])
    for i in range(dimKernel-2):
        alpha[i+1]=float(temp[i+1])
    alpha[dimKernel-1]=float(temp[dimKernel-1].split("]")[0])
    variance=float(val[2].split(",")[0])
    mu=float(val[3].split("(")[1].split(')')[0])
    return alpha,variance,mu

def startSBOobjects(directory,ind,kern,scaleAlpha):
    nameDirectory="TestResults"+"%d"%ind
    l={}
    l['folderContainerResults']=os.path.join(nameDirectory,"SBO")
    l['numberTrainingData']=numberPoints
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
    l['numberEstimateF']=numberSamplesForF
    l['constraintA']=lowerX
    l['constraintB']=UpperX
    l['simulatorW']=simulatorW
    l['XWhist']=np.loadtxt(os.path.join(directory,"%d"%ind+"XWHist.txt"))[0:numberPoints,:]
    l['yHist']=np.loadtxt(os.path.join(directory,"%d"%ind+"yhist.txt"))[0:numberPoints]
    l['yHist']=l['yHist'].reshape((numberPoints,1))
    l['varHist']=np.loadtxt(os.path.join(directory,"%d"%ind+"varHist.txt"))[0:numberPoints]
    l['kernel']=kern
    def B2(x,XW,n1,n2):
        return B(x,XW,n1,n2,scaleAlpha)
    l['B']=B2
    def gradXWSigma0func2(n,new,objVOI,Xtrain2,Wtrain2):
        return gradXWSigmaOfunc(n,new,objVOI,Xtrain2,Wtrain2,scaleAlpha=scaleAlpha)
    l['gradXWSigmaOfunc']=gradXWSigma0func2
    def gradXBf2(new,objVoi,BN,keep):
        return gradXB(new,objVoi,BN,keep,scaleAlpha)
    l['gradXBfunc']=gradXBf2
    def gradWBf2(new,objVoi,BN,keep):
        return gradWB(new,objVoi,BN,keep,scaleAlpha)
    l['gradWBfunc']=gradWBf2
    l['randomSeed']=ind
    l['pointsVOI']=pointsVOI
    def gradXBAn2(x,n,B,objGP,X):
        return gradXBforAn(x,n,B,objGP,X,scaleAlpha)
    l['gradXBforAn']=gradXBAn2
    l['numberParallel']=10
    return SB.SBO(**l)    

def testKernel(scaleAlpha=1000.0,n=0,numberPoints=20,ind=-1):
    kern=kernelFunc(XWtrain,yTrain,NoiseTrain,scaleAlpha)
    bike1=startSBOobjects(directory[0],1,kern,scaleAlpha)
    bike1.trainModel(numStarts=1)
   # print kern.alpha
    tempX=sampleFromX(1)
    tempFour=numberBikes-np.sum(tempX,1)
    tempFour=tempFour.reshape((1,1))
    Xtrain=np.concatenate((tempX,tempFour),1)
    Wtrain=simulatorW(1)
    new=np.concatenate((Xtrain,Wtrain),1)
    if ind!=-1:
        return 0
    print "pointsNew"
    print new
    ######An################
    args2={}
    args2['start']=tempX
    args2['i']=0
    print "Anopt"
    misc.AnOptWrapper(bike1,**args2) 
    print "endAnOpti"
    ########################
    c1,c2=functionGradientAscentVn(np.concatenate((tempX,Wtrain),1),True,bike1,0)
    a1,b1=gradXWSigmaOfunc(n,new,bike1._VOI,XWtrain[0:numberPoints,0:n1],XWtrain[0:numberPoints,n1:n1+n2],scaleAlpha)
    print np.sqrt(np.sum(a1**2)),np.sqrt(np.sum(b1**2)),np.sqrt(np.sum(c2**2))

for i in range(1):
    testKernel()
#testKernel(scaleAlpha=1000.0)
def printSBO(sboObj):
    bike1=sboObj
    print "Points"
    print bike1._XWhist
    print "responses"
    print bike1._yHist
    print "variances"
    print bike1._varianceObservations
    print "kernel"
    print "alpha"
    print bike1._k.alpha
    print "variance"
    print bike1._k.variance
    print "mu"
    print bike1._k.mu

#printSBO(bike1)
#A=bike1._k.A(bike1._XWhist,noise=bike1._varianceObservations)
#L=bike.np.linalg.cholesky(A)
#iRun=numberPoints

#def objectiveFunction(x,grad):
#    return bike1.functionGradientAscentAn(x,grad,bike1,0,L)

#def voiFunction(x,grad):
#    return bike1.functionGradientAscentVn(x,grad,bike1,0)

method="steepest"
args={}
args['n1']=4
#args['projectGradient']=projectGradient
args['xtol']=1e-8
nStart=0


