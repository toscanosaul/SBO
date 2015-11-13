"""
Stratified Bayesian Optimization (SBO) Algorithm.

This is a new algorithm proposed by [Toscano-Palmerin and Frazier][tf].
It's used for simulation optimization problems. Specificially, it's used
for the global optimization of the expectation of continuos functions
(respect to some metric), which depend on big random vectors. We suppose
that a small number of random variables have a much stronger effect on the
variability. In general, the functions are time-consuming to evaluate, and
the derivatives are unavailable.

In mathematical notation, we want to solve the problem:

	max_{x} E[f(x,w,z)]

where the expectation is over w and z, and w is the random vector that have
a much stronger effect on the variability of f.

[tf]: http://toscanosaul.github.io/saul/SBO.pdf

This class takes five class arguments (for details, see [tf]):


(*) Objective class:
-fobj: The simulator or objective function.
-dimSeparation: Dimension of w.
-noisyF: Estimator of the conditional expectation given w, F(x,w)=E[f(x,w,z)|w].
-numberEstimateF: Observations used to estimate F.
-sampleFromX: Chooses a point x at random.
-simulatorW: Simulates a random vector w.
-estimationObjective: Estimates the expectation of fobj. 



(*) Statistical Class:
-kernel: Kernel object for the GP on F.
-dimensionKernel: Dimension of the kernel.
-scaledAlpha: Parameter to scale the alpha parameters of the kernel,
	      alpha/(scaledAlpha^{2})
-B(x,XW,n1,n2,logproductExpectations=None): Computes
	\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w),
	where x can be a vector.
-numberTrainingData: Number of training data.
-XWhist: Training points (x,w).
-yHist: Training observations y.
-varHist: Noise of the observations of the training data.
-computeLogProductExpectationsForAn: Computes the vector with the logarithm
		of the product of the expectations of
		np.exp(-alpha2[j]*((z-W[i,j])**2))
		where W[i,:] is a point in the history.
-gradXBforAn: Computes the gradients with respect to x of
	     B(x,i)=\int\Sigma_{0}(x,w,x_{i},w_{i})dp(w),
	     where (x_{i},w_{i}) is a point in the history observed.




(*) Optimization class:
-numberParallel: Number of starting points for the multistart gradient ascent algorithm.
-dimXsteepest: Dimension of x when the VOI and a_{n} are optimized. We may want to reduce
	       the dimension of the original problem.
-transformationDomainX: Transforms the point x given by the steepest ascent method to the right domain
			of x.
-transformationDomainW: Transforms the point w given by the steepest ascent method to the right domain
			of w.
-projectGradient: Project a point x to the domain of the problem at each step of the gradient
		  ascent method if needed.
-functionGradientAscentVn: Function used for the gradient ascent method. It evaluates the VOI,
			   when grad and onlyGradient are False; it evaluates the VOI and
			   computes its derivative when grad is True and onlyGradient is False,
			   and computes only its gradient when gradient and onlyGradient are both
			   True. 
-functionGradientAscentAn: Function used for the gradient ascent method. It evaluates a_{n},
			   when grad and onlyGradient are False; it evaluates a_{n} and
			   computes its derivative when grad is True and onlyGradient is False,
			   and computes only its gradient when gradient and onlyGradient are both
			   True.
-functionConditionOpt: Gives the stopping rule for the steepest ascent method, e.g. the function
			could be the Euclidean norm. 
-xtol: Tolerance of x for the convergence of the steepest ascent method.



(*) VOI class:
#-pointsVOI: Points of the discretization to compute the VOI.
-VOIobj: VOI object.
#--gradXWSigmaOfunc: Computes the gradient of Sigma_{0}, which is the covariance of the Gaussian
		   Process on F.
#-gradXBfunc: Computes the gradients with respect to x_{n+1} of
	     B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
	     where x_{p} is a point in the discretization of the domain of x.
#-gradWBfunc: Computes the gradients with respect to w_{n+1} of
	     B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
	     where x_{p} is a point in the discretization of the domain of x.


(*) Miscellaneous class:
-randomSeed: Random seed used to run the problem. Only needed for the name of the
	     files with the results.
-parallel: True if we want to run the multistart gradient ascent algorithm and
	   train the kernel of the GP in parallel; otherwise, it's false. 
-folderContainerResults: Direction where the files with the results are saved. 
-createNewFiles: True if we want to create new files for the results; it's false
		 otherwise. If we want to add more results to our previos results,
		 this variable should be false.

"""

from math import *
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import numpy as np
from matplotlib import pyplot as plt
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
import files as fl

class SBO:
    def __init__(self, Objobj,miscObj,
                 VOIobj,optObj,statObj,dataObj):

	self.dataObj=dataObj
	#####
	self.parallel=miscObj.parallel
	self.randomSeed=miscObj.rs
	self.rs=miscObj.rs
	self.path=os.path.join(miscObj.folder,'%d'%self.rs+"run")
	self.createNewFiles=miscObj.create
	######

	self.functionConditionOpt=optObj.functionConditionOpt
	self.xtol=optObj.xtol
	self.transformationDomainX=optObj.transformationDomainX
        self.transformationDomainW=optObj.transformationDomainW
	self.projectGradient=optObj.projectGradient
	self.functionGradientAscentAn=optObj.functionGradientAscentAn
        self.functionGradientAscentVn=optObj.functionGradientAscentVn
        self.dimXsteepest=optObj.dimXsteepest
	self.numberParallel=optObj.numberParallel
        
	##########

        
	####
        self.sampleFromX=Objobj.sampleFromX
	self.estimationObjective=Objobj.estimationObjective
	self._fobj=Objobj.fobj
        self._infSource=Objobj.noisyF ###returns also the noise
        self._numberSamples=Objobj.numberEstimateF
	self._n1=Objobj.dimSeparation
	self._simulatorW=Objobj.simulatorW
	####
	if not os.path.exists(self.path):
            os.makedirs(self.path)


	
	self.stat=statObj
	self.computeLogProductExpectationsForAn=statObj.computeLogProductExpectationsForAn
	self.scaledAlpha=statObj.scaledAlpha
        self.numberTraining=statObj._numberTraining
        self._k=statObj._k
	self.B=statObj.B
	self._XWhist=statObj._Xhist
        self._yHist=statObj._yHist
        self._varianceObservations=statObj._noiseHist

        self._solutions=[]
        self._valOpt=[]
        
        self._dimension=statObj._n
        self._dimW=self._dimension-self._n1
        
	self.histSaved=0
	self.Bhist=np.zeros((VOIobj.sizeDiscretization,0))
        

        
        self.optRuns=[]
        self.optPointsArray=[]
	
	self._VOI=VOIobj


      
  
    ##m is the number of iterations to take
    def SBOAlg(self,m,nRepeat=10,Train=True,**kwargs):
	if self.createNewFiles:
	    fl.createNewFilesFunc(self.path,self.rs)
	fl.writeTraining(self)
        if Train is True:
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
	if self.parallel:
	    self.optAnParal(m,self.numberParallel)
	else:
	    self.optAnnoParal(i)
    ###start is a matrix of one row
    ###
    def optimizeVOI(self,start, i,L,temp2,a,B,scratch):
        opt=op.OptSteepestDescent(n1=self.dimXsteepest,projectGradient=self.projectGradient,
				  stopFunction=self.functionConditionOpt,xStart=start,
				  xtol=self.xtol)
        def g(x,grad,onlyGradient=False):
            return self.functionGradientAscentVn(x,grad,self._VOI,i,L,temp2,a,
						 scratch,onlyGradient=onlyGradient,
						 kern=self.stat._k,XW=self.dataObj.Xhist)

        opt.run(f=g)
        self.optRuns.append(opt)
        xTrans=self.transformationDomainX(opt.xOpt[0:1,0:self.dimXsteepest])
        self.optPointsArray.append(xTrans)


    
    
    def getParametersOptVoi(self,i):
	n1=self._n1
	n2=self._dimW
	tempN=self.numberTraining+i
	A=self._VOI._k.A(self.dataObj.Xhist[0:tempN,:],noise=self.dataObj.noiseHist[0:tempN])
	L=np.linalg.cholesky(A)
	m=self._VOI._points.shape[0]
	for j in xrange(self.histSaved,tempN):
	    temp=self.B(self._VOI._points,self.dataObj.Xhist[j,:],self._n1,self._dimW) ###change my previous function because we have to concatenate X and W
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
	args2={}
        args2['i']=i
	args2['L']=L
	args2['temp2']=temp2
	args2['a']=a
	args2['B']=self.Bhist
	args2['scratch']=scratch
	return args2
    
    def optVOInoParal(self,i):
	n1=self._n1
	n2=self._dimW
	Xst=self.sampleFromX(1)
	wSt=self._simulatorW(1)
	x1=Xst[0:0+1,:]
	w1=wSt[0:0+1,:]
	tempN=self.numberTraining+i
	st=np.concatenate((x1,w1),1)
	args2=self.getParametersOptVoi(i)
	args2['start']=st
	self.optRuns.append(misc.VOIOptWrapper(self,**args2))
	fl.writeNewPointSBO(self,self.optRuns[0])


    def optVOIParal(self,i,nStart,numProcesses=None):
        try:
            n1=self._n1
            n2=self._dimW
	    tempN=self.numberTraining+i
         #   dim=self.dimension
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            #New
            Xst=self.sampleFromX(nStart)
            wSt=self._simulatorW(nStart)
	    args2=self.getParametersOptVoi(i)
            ######
            for j in range(nStart):
                x1=Xst[j:j+1,:]
                w1=wSt[j:j+1,:]
                st=np.concatenate((x1,w1),1)
                args2['start']=st

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
	    fl.writeNewPointSBO(self,self.optRuns[j])

        self.optRuns=[]
        self.optPointsArray=[]
            
    def optimizeAn(self,start,i,L,logProduct):
        opt=op.OptSteepestDescent(n1=self.dimXsteepest,projectGradient=self.projectGradient,
				  xStart=start,xtol=self.xtol,
				  stopFunction=self.functionConditionOpt)
        tempN=i+self.numberTraining

        def g(x,grad,onlyGradient=False):
            return self.functionGradientAscentAn(x,grad,self.stat,i,self.dataObj,L,onlyGradient=onlyGradient,
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
	logProduct=self.stat.computeLogProductExpectationsForAn(self.dataObj.Xhist[0:tempN,n1:self._dimW+n1],
                                                         tempN,self.stat._k)
	Xst=self.sampleFromX(1)
	args2={}
	args2['start']=Xst[0:0+1,:]
	args2['i']=i
	args2['L']=L
	args2['logProduct']=logProduct
	self.optRuns.append(misc.AnOptWrapper(self,**args2))
	fl.writeSolution(self,self.optRuns[0])

            
     

    
    def optAnParal(self,i,nStart,numProcesses=None):
        try:
            n1=self._n1
	    tempN=i+self.numberTraining
	    A=self._k.A(self._XWhist[0:tempN,:],noise=self._varianceObservations[0:tempN])
	    L=np.linalg.cholesky(A)
	    ######computeLogProduct....only makes sense for the SEK, the function should be optional
	    logProduct=self.stat.computeLogProductExpectationsForAn(self.dataObj.Xhist[0:tempN,n1:self._dimW+n1],
							       tempN)
         #   dim=self.dimension
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            Xst=self.sampleFromX(nStart)
            for j in range(nStart):
                args2={}
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
	    fl.writeSolution(self,self.optRuns[j])

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
       
    

    
 
