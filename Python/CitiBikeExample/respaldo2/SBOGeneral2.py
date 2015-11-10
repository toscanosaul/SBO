#!/usr/bin/env python

#!/usr/bin/env python
#Stratified Bayesian Optimization

from math import *



import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import numpy as np
from matplotlib import pyplot as plt
#import GPy
from numpy import multiply
from numpy.linalg import inv
from AffineBreakPoints import *
from scipy.stats import norm
from grid import *
import pylab as plb
from scipy import linalg
#import multiprocessing
#from multiprocessing.pool import ApplyResult
from VOIsboGaussian import aN_grad,Vn
from SBOGaussianOptima import steepestAscent,SteepestAscent_aN
from functionsStartSBOGaussian import g,Fvalues
import SquaredExponentialKernel as SK
import VOIGeneral as VOI
import statGeneral as stat
import optimization as opt
import multiprocessing as mp
import os

####WRITE THE DOCUMENTATION

#f is the function to optimize
#alpha1,alpha2,sigma0 are the parameters of the covariance matrix of the gaussian process
#alpha1,alpha2 should be vectors
#muStart is the mean of the gaussian process (assumed is constant)
#sigma, mu are the parameters of the distribution of w1\inR^d1
#n1,n2 are the dimensions of x and w1, resp.

#c,d are the vectors where the grid is built
#l is the size of the grid


####X,W are column vectors
###Gaussian kernel: Variance same than in paper. LengthScale is 1/2alpha. Input as [alpha1,alpha2,...]
####Set ARD=True. 
##wStart:start the steepest at that point
###N(w1,sigmaW2) parameters of w2|w1
###mPrior samples for the prior

####include conditional dist and density. choose kernels
###python docstring
###RESTART AT ANY STAGE (STREAM)
###Variance(x,w,x,w)=1 (prior)

####kernel should receive only matrices as input.
####B(x,X,W) is a function that computes B(x,i). X=X[i,:],W[i,:]. x is a matrix where each row is a point to be evaluated

####checar multidimensional!!! Poisson already works

####E(fobj)
####trainingData is a dictionary={XWpoints,Ypoints,Noise}, Xpoints is a matrix, Ypoints,Nose are vectors
class SBO:
    def __init__(self, fobj,dimensionKernel,noisyF,gradXSigmaOfunc,gradXBfunc,gradWSigmafunc,
                 dimSeparation=None,trainingData=None,numberEstimateF=15,
                 B=None,kernel=None,numberTrainingData=0,Bhist=None,
                 XWhist=None,yHist=None,varHist=None,pointsVOI=None,
                 constraintA=None,constraintB=None,simulatorW=None,createNewFiles=True,randomSeed=1):
       # np.random.seed(randomSeed)
        self.randomSeed=randomSeed
        self.numberTraining=numberTrainingData
        self.path='%d'%randomSeed+"run"
        if not os.path.exists(path):
            os.makedirs(path)
        os.path.join(path,)
        if createNewFiles is True:
            f=open(os.path.join(self.path,'%d'%randomSeed+"hyperparameters.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"XWHist.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"yhist.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"varHist.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"optimalSolutions.txt"),'w')
            f.close()
        if kernel is None:
            kernel=SK.SEK(dimensionKernel)
       # if acquisitionFunction is None:
        #    acquisitionFunction=VOI
        self._k=kernel
        self._fobj=fobj
        self._infSource=noisyF ###returns also the noise
        self._numberSamples=numberEstimateF
        #self._acquisitionFunction=VOI
        self._B=B
        self._solutions=[]
        self._valOpt=[]
        self._n1=dimSeparation
        self._dimension=dimensionKernel
        self._dimW=self._dimension-self._n1
        self._constraintA=constraintA
        self._constraintB=constraintB
        self._simulatorW=simulatorW
        if XWhist is None:
            ###then trainingData is not None
            kernel.X=trainingData['XWpoints']
            XWhist=trainingData['XWpoints']
            with open('%d'%randomSeed+"XWHist.txt", "a") as f:
                np.savetxt(f,XWhist)
        if yHist is None:
            kernel.y=trainingData['Ypoints']
            yHist=trainingData['Ypoints']
            with open('%d'%randomSeed+"yhist.txt", "a") as f:
                np.savetxt(f,XWhist)
        if varHist is None:
            kernel.noise=trainingData['Noise']
            varHist=trainingData['Noise']
            with open('%d'%randomSeed+"varHist.txt", "a") as f:
                np.savetxt(f,varHist)
        
        self._XWhist=XWhist
        self._yHist=yHist
        self._varianceObservations=varHist
        self._trainingData=trainingData
        
        self.optRuns=[]
        self.optPointsArray=[]

        self._VOI=VOI.VOISBO(kernel=kernel,dimKernel=dimensionKernel,numberTraining=numberTrainingData,
                         gradWSigmafunc=gradWSigmafunc,Bhist=Bhist,pointsApproximation=pointsVOI,
                         gradXSigmaOfunc=gradXSigmaOfunc,gradXBfunc=gradXBfunc,B=B,PointsHist=XWhist,
                         yHist=yHist,noiseHist=varHist,dimW=self._dimW)

    ##m is the number of iterations to take
    def SBOAlg(self,m,nRepeat=10,Train=True,**kwargs):
        if Train is True:
            ###TrainingData is not None
            self.trainModel(numStarts=nRepeat,**kwargs)
        
        for i in range(m):
            self.optVOIParal(i)
            ###opt,valOpt=optimize(aN)
            self.optAnParal(i)
        self.optAnParal(m)
            
    ###start is a matrix of one row
    ###
    def optimizeVOI(self,start, i):
        opt=OptSteepestDescent(start,**kwargs)
        opt.constraintA=self._constraintA
        opt.constraintB=self._constraintB
        def g(x,grad):
            return -self._VOI.VOIfunc(i,x,grad=grad)
        opt.run(f=g)
        self.optRuns.append(opt)
        self.optPointsArray.append(opt.xOpt)
        
    def optVOIParal(self,i,nStart=10,numProcesses=None):
        try:
            n1=self._n1
         #   dim=self.dimension
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            for i in range(nStart):
                x1=np.random.uniform(self._constraintA,self._constraintB,(1,1))
                w1=self._simulatorW((1,1))
                st=np.concatenate((x1,w1),1)
                args2={}
                args2['start']=st
                args2['i']=i
                job = pool.apply_async(misc.VOIOptWrapper, args=(self,), kwds=args2)
                jobs.append(job)
            
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()
        
        numStarts=nStart
        for i in range(numStarts):
            try:
                self.optRuns.append(jobs[i].get())
            except Exception as e:
                print "what"

                
        if len(self.optRuns):
            j = np.argmin([o.fOpt for o in self.optRuns])
            temp=self.optRuns[j].xOpt
            self.optRuns=[]
            self.optPointsArray=[]
            self._XWhist=np.vstack([self._XWhist,temp])
            y,var=self._infSource(self._XWhist,self._numberSamples)
            self._y=np.vstack([self._yHist,y])
            self._varianceObservations=np.append(self._varianceObservations,var)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"varHist.txt"), "a") as f:
                np.savetxt(f,var)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"yHist.txt"), "a") as f:
                np.savetxt(f,y)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"XWHist.txt"), "a") as f:
                np.savetxt(f,XWhist)
        self.optRuns=[]
        self.optPointsArray=[]
            
    def optimizeAn(self,start,i):
        opt=OptSteepestDescent(start,**kwargs)
        opt.constraintA=self._constraintA
        opt.constraintB=self._constraintB
        tempN=i+self.numberTraining
        A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
        L=np.linalg.cholesky(A)
        def g(x,grad):
            return -self._VOI._GP.aN_grad(x,L=L,n=i,gradient=grad)
        opt.run(f=g)
        self.optRuns.append(opt)
        self.optPointsArray.append(opt.xOpt)
    
    def optAnParal(self,i,nStart=10,numProcesses=None):
        try:
            n1=self._n1
         #   dim=self.dimension
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            for i in range(nStart):
                x1=np.random.uniform(self._constraintA,self._constraintB,(1,1))
                args2={}
                args2['start']=x1
                args2['i']=i
                job = pool.apply_async(misc.AnOptWrapper, args=(self,), kwds=args2)
                jobs.append(job)
            
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()
        
        numStarts=nStart
        for i in range(numStarts):
            try:
                self.optRuns.append(jobs[i].get())
            except Exception as e:
                print "what"

                
        if len(self.optRuns):
            j = np.argmin([o.fOpt for o in self.optRuns])
            temp=self.optRuns[j].xOpt
            self._solutions.append(temp)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"optimalSolutions.txt"), "a") as f:
                np.savetxt(f,temp)
            self.optRuns=[]
            self.optPointsArray=[]
            
        self.optRuns=[]
        self.optPointsArray=[]
    
    def trainModel(self,numStarts,**kwargs):
        kernel.train(numStarts=numStarts,**kwargs)
    

    
 