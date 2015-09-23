import numpy as np
from math import *
import sys
sys.path.append("..")
import optimization as op
import multiprocessing as mp
import os
import misc
from matplotlib import pyplot as plt
from numpy import multiply
from numpy.linalg import inv
from AffineBreakPoints import *
from scipy.stats import norm
from grid import *
import pylab as plb
from scipy import linalg
import statGeneral as stat
import VOIGeneral as VOI

class EI:
    def __init__(self, fobj,dimensionKernel,noisyG,gradXKern,
                 trainingData=None,numberEstimateG=15, sampleFromX=None,
                 kernel=None,numberTrainingData=0,dimXsteepest=0
                 ,Xhist=None,yHist=None,varHist=None,pointsVOI=None,folder=None,projectGradient=None,
                 constraintA=None,constraintB=None,simulatorW=None,createNewFiles=True,randomSeed=1,
                 functionGradientAscentVn=None,functionGradientAscentMuN=None,numberParallel=10,
                 transformationDomainX=None,estimationObjective=None,
                 folderContainerResults=None,scaledAlpha=1.0,xtol=None,functionConditionOpt=None):
        if xtol is None:
            xtol=1e-8
        self.functionConditionOpt=functionConditionOpt
        self.xtol=xtol
        self.scaledAlpha=scaledAlpha
        self.transformationDomainX=transformationDomainX
        self.randomSeed=randomSeed
        self.numberTraining=numberTrainingData
        self.projectGradient=projectGradient
        self.sampleFromX=sampleFromX
        self.functionGradientAscentMuN=functionGradientAscentMuN
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
            f=open(os.path.join(self.path,'%d'%randomSeed+"XHist.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"yhist.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"varHist.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"optimalSolutions.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"optimalValues.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"optEIgrad.txt"),'w')
            f.close()
            f=open(os.path.join(self.path,'%d'%randomSeed+"optAngrad.txt"),'w')
            f.close()
        if kernel is None:
            kernel=SK.SEK(dimensionKernel)
        self._k=kernel
        self._fobj=fobj
        self._infSource=noisyG ###returns also the noise
        self._numberSamples=numberEstimateG
        self._solutions=[]
        self._valOpt=[]
        self._n1=dimensionKernel
        self._dimension=dimensionKernel
        self._constraintA=constraintA
        self._constraintB=constraintB
        with open(os.path.join(self.path,'%d'%randomSeed+"XHist.txt"), "a") as f:
            np.savetxt(f,Xhist)
        with open(os.path.join(self.path,'%d'%randomSeed+"yhist.txt"), "a") as f:
            np.savetxt(f,yHist)
        with open(os.path.join(self.path,'%d'%randomSeed+"varHist.txt"), "a") as f:
            np.savetxt(f,varHist)
    
        self._Xhist=Xhist
        self._yHist=yHist
        self._varianceObservations=varHist
        self._trainingData=trainingData
        
        self.optRuns=[]
        self.optPointsArray=[]

        self._VOI=VOI.EI(kernel=kernel,dimKernel=dimensionKernel,numberTraining=numberTrainingData,
                         gradXKern=gradXKern,pointsApproximation=pointsVOI,
                         PointsHist=Xhist,yHist=yHist,noiseHist=varHist)

    def EIAlg(self,m,nRepeat=10,Train=True,**kwargs):
        if Train is True:
            ###TrainingData is not None
	    self.trainModel(numStarts=nRepeat,**kwargs)
       # points=self._VOI._points
        for i in range(m):
            print i
            self.optVOIParal(i,self.numberParallel)
           # st=np.array([[2.1]])
	    st=self.sampleFromX(1)
            args2={}
            args2['start']=st
            args2['i']=i
           # misc.VOIOptWrapper(self,**args2)
            print i
            self.optAnParal(i,self.numberParallel)
            print i
        self.optAnParal(m,self.numberParallel)
        
    ###start is a matrix of one row
    ###
    def optimizeVOI(self,start, i):
        opt=op.OptSteepestDescent(n1=self.dimXsteepest,projectGradient=self.projectGradient,xStart=start,xtol=self.xtol,stopFunction=self.functionConditionOpt)
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
          #  n2=self._dimW
         #   dim=self.dimension
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            #New
            Xst=self.sampleFromX(nStart)
           # wSt=self._simulatorW(nStart)
            ######
            for j in range(nStart):
                st=Xst[j:j+1,:]
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
         #   wTrans=self.transformationDomainW(self.optRuns[j].xOpt[0:1,self.dimXsteepest:self.dimXsteepest+self._dimW])
            ###falta transformar W
            temp=xTrans
           # temp=np.concatenate((xTrans,wTrans),1)
            self.optRuns=[]
            self.optPointsArray=[]
            self._Xhist=np.vstack([self._Xhist,temp])
            self._VOI._PointsHist=self._Xhist
            self._VOI._GP._Xhist=self._Xhist
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
            with open(os.path.join(self.path,'%d'%self.randomSeed+"XHist.txt"), "a") as f:
                np.savetxt(f,temp)
            with open(os.path.join(self.path,'%d'%self.randomSeed+"optEIgrad.txt"), "a") as f:
                np.savetxt(f,gradOpt)
        self.optRuns=[]
        self.optPointsArray=[]
        
        
    def optimizeAn(self,start,i):
        opt=op.OptSteepestDescent(n1=self.dimXsteepest,projectGradient=self.projectGradient,xStart=start,xtol=self.xtol,stopFunction=self.functionConditionOpt)
        opt.constraintA=self._constraintA
        opt.constraintB=self._constraintB
        tempN=i+self.numberTraining
        #A=self._k.A(self._XWhist[0:tempN,:],noise=self._varianceObservations[0:tempN])
        #L=np.linalg.cholesky(A)
        def g(x,grad):
            return self.functionGradientAscentMuN(x,grad,self,i)
         #   return self.functionGradientAscentMun(x,grad,self,i,L)
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
        self._k.train(scaledAlpha=self.scaledAlpha,numStarts=numStarts,**kwargs)
        
        f=open(os.path.join(self.path,'%d'%self.randomSeed+"hyperparameters.txt"),'w')
        f.write(str(self._k.getParamaters()))
        f.close()
        
