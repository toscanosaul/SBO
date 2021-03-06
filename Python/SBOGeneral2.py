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

import SquaredExponentialKernel as SK
import VOIGeneral as VOI
import statGeneral as stat
import optimization as op
import multiprocessing as mp
import os
import misc

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
    def __init__(self, fobj,dimensionKernel,noisyF,gradXBfunc,gradXWSigmaOfunc,gradXBforAn,
                 dimSeparation=None,trainingData=None,numberEstimateF=15, sampleFromX=None,
                 B=None,kernel=None,numberTrainingData=0,Bhist=None,gradWBfunc=None,dimXsteepest=0,
                 XWhist=None,yHist=None,varHist=None,pointsVOI=None,folder=None,projectGradient=None,
                 constraintA=None,constraintB=None,simulatorW=None,createNewFiles=True,randomSeed=1,
                 functionGradientAscentVn=None,functionGradientAscentAn=None,numberParallel=10,
                 transformationDomainX=None,transformationDomainW=None,estimationObjective=None,
                 folderContainerResults=None):
       # np.random.seed(randomSeed)
        self.transformationDomainX=transformationDomainX
        self.transformationDomainW=transformationDomainW
        self.randomSeed=randomSeed
        self.numberTraining=numberTrainingData
        self.projectGradient=projectGradient
        self.sampleFromX=sampleFromX
        self.functionGradientAscentAn=functionGradientAscentAn
        self.functionGradientAscentVn=functionGradientAscentVn
        self.dimXsteepest=dimXsteepest
        self.estimationObjective=estimationObjective
        if folder is None:
            self.path=os.path.join(folderContainerResults,'%d'%randomSeed+"run")
        else:
            self.path=folder+'%d'%randomSeed+"run"
        self.numberParallel=numberParallel
        if not os.path.exists(self.path):
            os.makedirs(self.path)
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
            f=open(os.path.join(self.path,'%d'%randomSeed+"optimalValues.txt"),'w')
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
        with open(os.path.join(self.path,'%d'%randomSeed+"XWHist.txt"), "a") as f:
            np.savetxt(f,XWhist)
        with open(os.path.join(self.path,'%d'%randomSeed+"yhist.txt"), "a") as f:
            np.savetxt(f,yHist)
        with open(os.path.join(self.path,'%d'%randomSeed+"varHist.txt"), "a") as f:
            np.savetxt(f,varHist)
    
    
        
        self._XWhist=XWhist
        self._yHist=yHist
        self._varianceObservations=varHist
        self._trainingData=trainingData
        
        self.optRuns=[]
        self.optPointsArray=[]

        self._VOI=VOI.VOISBO(kernel=kernel,dimKernel=dimensionKernel,numberTraining=numberTrainingData,
                         gradXWSigmaOfunc=gradXWSigmaOfunc,Bhist=Bhist,pointsApproximation=pointsVOI,
                         gradXBfunc=gradXBfunc,B=B,PointsHist=XWhist,gradWBfunc=gradWBfunc,
                         yHist=yHist,noiseHist=varHist,gradXBforAn=gradXBforAn,dimW=self._dimW)

    ##m is the number of iterations to take
    def SBOAlg(self,m,nRepeat=10,Train=True,**kwargs):
        if Train is True:
            ###TrainingData is not None
            self.trainModel(numStarts=nRepeat,**kwargs)
        points=self._VOI._points
        for i in range(m):
            print i
            self.optVOIParal(i,self.numberParallel)
            
            #####
          #  n1=self._n1
          #  n2=self._dimW
          #  Xst=self.sampleFromX(1)
          #  wSt=self._simulatorW(1)
          #  st=np.concatenate((Xst,wSt),1)
          #  args2={}
          #  args2['start']=Xst
          #  args2['i']=i
          #  misc.AnOptWrapper(self,**args2)
            ####
            print i
            self.optAnParal(i,self.numberParallel)
            print i
        self.optAnParal(m,self.numberParallel)
            
    ###start is a matrix of one row
    ###
    def optimizeVOI(self,start, i):
        opt=op.OptSteepestDescent(n1=self.dimXsteepest,projectGradient=self.projectGradient,xStart=start,xtol=1e-8)
        opt.constraintA=self._constraintA
        opt.constraintB=self._constraintB
      #  self.functionGradientAscentAn
        def g(x,grad):
            return self.functionGradientAscentVn(x,grad,self,i)
            #temp=self._VOI.VOIfunc(i,x,grad=grad)
            #if grad==True:
            #    return temp[0],temp[1]
            #else:
            #    return temp
        opt.run(f=g)
        self.optRuns.append(opt)
        xTrans=self.transformationDomainX(opt.xOpt[0:1,0:self.dimXsteepest])
        self.optPointsArray.append(xTrans)
        
    def optVOIParal(self,i,nStart,numProcesses=None):
        try:
            n1=self._n1
            n2=self._dimW
         #   dim=self.dimension
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            #New
            Xst=self.sampleFromX(nStart)
            wSt=self._simulatorW(nStart)
            ######
            for j in range(nStart):
             #  np.random.seed(seeds[j])
               # x1=np.random.uniform(self._constraintA,self._constraintB,(1,n1))
               # w1=self._simulatorW(1)
                #x1=self.projectGradient(Xst[j,:]).reshape((1,n1))
                #w1=wSt[j,:].reshape((1,n2))
                x1=Xst[j:j+1,:]
                w1=wSt[j:j+1,:]
                st=np.concatenate((x1,w1),1)
              #  print st
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
     #   print jobs
        numStarts=nStart
     #   print jobs[0].get()
        for j in range(numStarts):
            try:
                self.optRuns.append(jobs[j].get())
            except Exception as e:
                print "Error optimizing VOI"
                
        if len(self.optRuns):
            j = np.argmax([o.fOpt for o in self.optRuns])
            temp=self.optRuns[j].xOpt
            xTrans=self.transformationDomainX(self.optRuns[j].xOpt[0:1,0:self.dimXsteepest])
            wTrans=self.transformationDomainW(self.optRuns[j].xOpt[0:1,self.dimXsteepest:self.dimXsteepest+self._dimW])
            ###falta transformar W
            temp=np.concatenate((xTrans,wTrans),1)
            self.optRuns=[]
            self.optPointsArray=[]
            self._XWhist=np.vstack([self._XWhist,temp])
            self._VOI._PointsHist=self._XWhist
            self._VOI._GP._Xhist=self._XWhist
            y,var=self._infSource(temp,self._numberSamples)
            self._yHist=np.vstack([self._yHist,y])
            self._VOI._yHist=self._yHist
            self._VOI._GP._yHist=self._yHist
            self._varianceObservations=np.append(self._varianceObservations,var)
            self._VOI._noiseHist=self._varianceObservations
            self._VOI._GP._noiseHist=self._varianceObservations
            with open(os.path.join(self.path,'%d'%self.randomSeed+"varHist.txt"), "a") as f:
                var=np.array(var).reshape(1)
                np.savetxt(f,var)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"yhist.txt"), "a") as f:
                y=np.array(y).reshape(1)
                np.savetxt(f,y)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"XWHist.txt"), "a") as f:
                np.savetxt(f,temp)
        self.optRuns=[]
        self.optPointsArray=[]
            
    def optimizeAn(self,start,i):
        opt=op.OptSteepestDescent(n1=self.dimXsteepest,projectGradient=self.projectGradient,xStart=start,xtol=1e-8)
        opt.constraintA=self._constraintA
        opt.constraintB=self._constraintB
        tempN=i+self.numberTraining
        A=self._k.A(self._XWhist[0:tempN,:],noise=self._varianceObservations[0:tempN])
        L=np.linalg.cholesky(A)
        def g(x,grad):
            return self.functionGradientAscentAn(x,grad,self,i,L)
      #  def g(x,grad):
      #      temp=self._VOI._GP.aN_grad(x,L,i,grad)
      #      if grad==False:
      #          return temp
      #      else:
      #          return temp[0],temp[1]
        opt.run(f=g)
        self.optRuns.append(opt)
        xTrans=self.transformationDomainX(opt.xOpt[0:1,0:self.dimXsteepest])
        self.optPointsArray.append(xTrans)
    
    def optAnParal(self,i,nStart,numProcesses=None):
        try:
            n1=self._n1
         #   dim=self.dimension
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            Xst=self.sampleFromX(nStart)
            for j in range(nStart):
           #     np.random.seed(seeds[j])
           #     x1=np.random.uniform(self._constraintA,self._constraintB,(1,n1))
                args2={}
               # x1=Xst[j,:]
                args2['start']=Xst[j:j+1,:]
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
        
        for j in range(numStarts):
            try:
                self.optRuns.append(jobs[j].get())
            except Exception as e:
                print "Error optimizing An"

                
        if len(self.optRuns):
            j = np.argmax([o.fOpt for o in self.optRuns])
            temp=self.optRuns[j].xOpt
            xTrans=self.transformationDomainX(self.optRuns[j].xOpt[0:1,0:self.dimXsteepest])
        #    temp2=self.
            self._solutions.append(xTrans)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"optimalSolutions.txt"), "a") as f:
                np.savetxt(f,xTrans)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"optimalValues.txt"), "a") as f:
                result,var=self.estimationObjective(xTrans[0,:])
                res=np.append(result,var)
                np.savetxt(f,res)
            self.optRuns=[]
            self.optPointsArray=[]
            
        self.optRuns=[]
        self.optPointsArray=[]
    
    def trainModel(self,numStarts,**kwargs):
        self._k.train(numStarts=numStarts,**kwargs)
        
        f=open(os.path.join(self.path,'%d'%self.randomSeed+"hyperparameters.txt"),'w')
        f.write(str(self._k.getParamaters()))
        f.close()
       
    

    
 