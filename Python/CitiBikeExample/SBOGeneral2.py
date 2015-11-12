#d!/usr/bin/env python

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
    def __init__(self, fobj,dimensionKernel,noisyF,gradXBfunc,gradXWSigmaOfunc,gradXBforAn, parallel,
                 dimSeparation=None,trainingData=None,numberEstimateF=15, sampleFromX=None,
                 B=None,kernel=None,numberTrainingData=0,Bhist=None,gradWBfunc=None,dimXsteepest=0,
                 XWhist=None,yHist=None,varHist=None,pointsVOI=None,folder=None,projectGradient=None,
                 constraintA=None,constraintB=None,simulatorW=None,createNewFiles=True,randomSeed=1,
                 functionGradientAscentVn=None,functionGradientAscentAn=None,numberParallel=10,
                 transformationDomainX=None,transformationDomainW=None,estimationObjective=None,
                 folderContainerResults=None,scaledAlpha=1.0,xtol=None,functionConditionOpt=None,
		 computeLogProductExpectationsForAn=None):
       # np.random.seed(randomSeed)
	self.computeLogProductExpectationsForAn=computeLogProductExpectationsForAn
	self.parallel=parallel
	if xtol is None:
	    xtol=1e-8
	self.functionConditionOpt=functionConditionOpt
	self.xtol=xtol
        self.scaledAlpha=scaledAlpha
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
#	self.path='%d'%randomSeed+"run"
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
            f=open(os.path.join(self.path,'%d'%randomSeed+"optVOIgrad.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"optAngrad.txt"),'w')
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
    
	self.histSaved=0
	self.Bhist=np.zeros((pointsVOI.shape[0],0))
        
        self._XWhist=XWhist
        self._yHist=yHist
        self._varianceObservations=varHist
        self._trainingData=trainingData
        
        self.optRuns=[]
        self.optPointsArray=[]
	
	self.B=B

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
	    if self.parallel:
		self.optVOIParal(i,self.numberParallel)
	    else:
		self.optVOInoParal(i)

            #####
            n1=self._n1
            n2=self._dimW
            Xst=self.sampleFromX(1)
            wSt=self._simulatorW(1)
            st=np.concatenate((Xst,wSt),1)
            args2={}
            args2['start']=st
            args2['i']=i
           # misc.VOIOptWrapper(self,**args2)
            ####
            args2['start']=self.sampleFromX(1)
           # misc.AnOptWrapper(self,**args2)
            print i
	    if self.parallel:
		self.optAnParal(i,self.numberParallel)
	    else:
		self.optAnnoParal(i)
            print i
        args2={}
        args2['start']=self.sampleFromX(1)
        args2['i']=m
      #  misc.AnOptWrapper(self,**args2)
    #    self.optAnParal(m,self.numberParallel)
    ###start is a matrix of one row
    ###
    def optimizeVOI(self,start, i,L,temp2,a,B,scratch):
        opt=op.OptSteepestDescent(n1=self.dimXsteepest,projectGradient=self.projectGradient,stopFunction=self.functionConditionOpt,xStart=start,xtol=self.xtol)
        opt.constraintA=self._constraintA
        opt.constraintB=self._constraintB
      #  self.functionGradientAscentAn
        def g(x,grad,onlyGradient=False):
            return self.functionGradientAscentVn(x,grad,self,i,L,temp2,a,B,scratch,onlyGradient=onlyGradient)

            #temp=self._VOI.VOIfunc(i,x,grad=grad)
            #if grad==True:
            #    return temp[0],temp[1]
            #else:
            #    return temp
        opt.run(f=g)
        self.optRuns.append(opt)
        xTrans=self.transformationDomainX(opt.xOpt[0:1,0:self.dimXsteepest])
        self.optPointsArray.append(xTrans)
    
    def optVOInoParal(self,i):
	n1=self._n1
	n2=self._dimW
	Xst=self.sampleFromX(1)
	wSt=self._simulatorW(1)
	x1=Xst[0:0+1,:]
	w1=wSt[0:0+1,:]
	tempN=self.numberTraining+i
	st=np.concatenate((x1,w1),1)
	#st=np.array([[ 1500,  1500,  1500,   348,   383,   333,   390]]) ####ELIMINATE!!!!!!!
	A=self._VOI._GP._k.A(self._VOI._GP._Xhist[0:tempN,:],noise=self._VOI._GP._noiseHist[0:tempN])
	L=np.linalg.cholesky(A)
	m=self._VOI._points.shape[0]
	for j in xrange(self.histSaved,tempN):
	    temp=self.B(self._VOI._points,self._VOI._GP._Xhist[j,:],self._n1,self._dimW) ###change my previous function because we have to concatenate X and W
	    self.Bhist=np.concatenate((self.Bhist,temp.reshape((m,1))),1)
	    self.histSaved+=1
	muStart=self._k.mu
	y=self._yHist
	temp2=linalg.solve_triangular(L,(self.Bhist).T,lower=True)
	temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
	a=muStart+np.dot(temp2.T,temp1)
	
	scratch=np.zeros((m,tempN))
	for j in xrange(m):
	    scratch[j,:]=linalg.solve_triangular(L,self.Bhist[j,:].transpose(),lower=True)
           # misc.VOIOptWrapper(self,**args2)
	args2={}
	args2['start']=st
        args2['i']=i
	args2['L']=L
	args2['temp2']=temp2
	args2['a']=a
	args2['B']=self.Bhist
	args2['scratch']=scratch
	self.optRuns.append(misc.VOIOptWrapper(self,**args2))
	j=0
	temp=self.optRuns[j].xOpt
	gradOpt=self.optRuns[j].gradOpt
	numberIterations=self.optRuns[j].nIterations
	gradOpt=np.sqrt(np.sum(gradOpt**2))
	gradOpt=np.array([gradOpt,numberIterations])
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
	with open(os.path.join(self.path,'%d'%self.randomSeed+"optVOIgrad.txt"), "a") as f:
	    np.savetxt(f,gradOpt)

        self.optRuns=[]
        self.optPointsArray=[]


    def optVOIParal(self,i,nStart,numProcesses=None):
        try:
            n1=self._n1
            n2=self._dimW
	    tempN=self.numberTraining+i
	    A=self._VOI._GP._k.A(self._VOI._GP._Xhist[0:tempN,:],noise=self._VOI._GP._noiseHist[0:tempN])
	    L=np.linalg.cholesky(A)
	    m=self._VOI._points.shape[0]
	    for j in xrange(self.histSaved,tempN):
		temp=self.B(self._VOI._points,self._VOI._GP._Xhist[j,:],self._n1,self._dimW) ###change my previous function because we have to concatenate X and W
		self.Bhist=np.concatenate((self.Bhist,temp.reshape((m,1))),1)
		self.histSaved+=1
	    muStart=self._k.mu
	    y=self._yHist
	    temp2=linalg.solve_triangular(L,(self.Bhist).T,lower=True)
	    temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
	    a=muStart+np.dot(temp2.T,temp1)
	    
	    scratch=np.zeros((m,tempN))
	    for j in xrange(m):
		scratch[j,:]=linalg.solve_triangular(L,self.Bhist[j,:].transpose(),lower=True)
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
              # erint st
                args2={}
                args2['start']=st
                args2['i']=i
		args2['L']=L
		args2['temp2']=temp2
		args2['a']=a
		args2['B']=self.Bhist
		args2['scratch']=scratch
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
            gradOpt=self.optRuns[j].gradOpt
            numberIterations=self.optRuns[j].nIterations
            gradOpt=np.sqrt(np.sum(gradOpt**2))
            gradOpt=np.array([gradOpt,numberIterations])
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
            with open(os.path.join(self.path,'%d'%self.randomSeed+"optVOIgrad.txt"), "a") as f:
                np.savetxt(f,gradOpt)

        self.optRuns=[]
        self.optPointsArray=[]
            
    def optimizeAn(self,start,i,L,logProduct):
        opt=op.OptSteepestDescent(n1=self.dimXsteepest,projectGradient=self.projectGradient,xStart=start,xtol=self.xtol,stopFunction=self.functionConditionOpt)
        opt.constraintA=self._constraintA
        opt.constraintB=self._constraintB
        tempN=i+self.numberTraining

        def g(x,grad,onlyGradient=False):
            return self.functionGradientAscentAn(x,grad,self,i,L,onlyGradient=onlyGradient,
						 logproductExpectations=logProduct)

        opt.run(f=g)
        self.optRuns.append(opt)
        xTrans=self.transformationDomainX(opt.xOpt[0:1,0:self.dimXsteepest])
        self.optPointsArray.append(xTrans)
    
    
    def optAnnoParal(self,i):
	n1=self._n1
	tempN=i+self.numberTraining
	A=self._k.A(self._XWhist[0:tempN,:],noise=self._varianceObservations[0:tempN])
	L=np.linalg.cholesky(A)
	######computeLogProduct....only makes sense for the SEK, the function should be optional
	logProduct=self.computeLogProductExpectationsForAn(self._XWhist[0:tempN,n1:self._dimW+n1],
                                                         tempN)
	Xst=self.sampleFromX(1)
	args2={}
	args2['start']=Xst[0:0+1,:]
	args2['i']=i
	args2['L']=L
	args2['logProduct']=logProduct
	self.optRuns.append(misc.AnOptWrapper(self,**args2))
	j = 0
	temp=self.optRuns[j].xOpt
	tempGrad=self.optRuns[j].gradOpt
	tempGrad=np.sqrt(np.sum(tempGrad**2))
	tempGrad=np.array([tempGrad,self.optRuns[j].nIterations])
	xTrans=self.transformationDomainX(self.optRuns[j].xOpt[0:1,0:self.dimXsteepest])
    #    temp2=self.
	self._solutions.append(xTrans)
	with open(os.path.join(self.path,'%d'%self.randomSeed+"optimalSolutions.txt"), "a") as f:
	    np.savetxt(f,xTrans)
	with open(os.path.join(self.path,'%d'%self.randomSeed+"optimalValues.txt"), "a") as f:
	    result,var=self.estimationObjective(xTrans[0,:])
	    res=np.append(result,var)
	    np.savetxt(f,res)
	with open(os.path.join(self.path,'%d'%self.randomSeed+"optAngrad.txt"), "a") as f:
	    np.savetxt(f,tempGrad)
	self.optRuns=[]
	self.optPointsArray=[]
            
        self.optRuns=[]
        self.optPointsArray=[]
     

    
    def optAnParal(self,i,nStart,numProcesses=None):
        try:
            n1=self._n1
	    tempN=i+self.numberTraining
	    A=self._k.A(self._XWhist[0:tempN,:],noise=self._varianceObservations[0:tempN])
	    L=np.linalg.cholesky(A)
	    ######computeLogProduct....only makes sense for the SEK, the function should be optional
	    logProduct=self.computeLogProductExpectationsForAn(self._XWhist[0:tempN,n1:self._dimW+n1],
							       tempN)
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
		args2['L']=L
		args2['logProduct']=logProduct
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
            tempGrad=self.optRuns[j].gradOpt
            tempGrad=np.sqrt(np.sum(tempGrad**2))
            tempGrad=np.array([tempGrad,self.optRuns[j].nIterations])
            xTrans=self.transformationDomainX(self.optRuns[j].xOpt[0:1,0:self.dimXsteepest])
        #    temp2=self.
            self._solutions.append(xTrans)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"optimalSolutions.txt"), "a") as f:
                np.savetxt(f,xTrans)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"optimalValues.txt"), "a") as f:
                result,var=self.estimationObjective(xTrans[0,:])
                res=np.append(result,var)
                np.savetxt(f,res)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"optAngrad.txt"), "a") as f:
                np.savetxt(f,tempGrad)
            self.optRuns=[]
            self.optPointsArray=[]
            
        self.optRuns=[]
        self.optPointsArray=[]

    def trainModel(self,numStarts,**kwargs):
	if self.parallel:
	    self._k.train(scaledAlpha=self.scaledAlpha,numStarts=numStarts,**kwargs)
	else:
	    self._k.trainnoParallel(scaledAlpha=self.scaledAlpha,**kwargs)
        f=open(os.path.join(self.path,'%d'%self.randomSeed+"hyperparameters.txt"),'w')
        f.write(str(self._k.getParamaters()))
        f.close()
       
    

    
 
