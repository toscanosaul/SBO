#!/usr/bin/env python

import numpy as np
#from scipy.spatial.distance import cdist
from scipy import linalg
from LBFGS import *
from scipy.optimize import fmin_l_bfgs_b
#from tdot import tdot
from matrixComputations import tripleProduct,inverseComp
from scipy.stats import multivariate_normal
import multiprocessing as mp
import misc
import optimization
from optimization import getOptimizationMethod
from scipy import array, linalg, dot
from nearest_correlation import nearcorr


######add a super class of this, to be able to optimize any object


###exp(-0.5*sum (alpha_i**2 *(x_i-y_i)**2))
##n is the dimension of the kernel
##the kernel is with alpha^2
####optimize respect to log(variance)
class SEK:
    def __init__(self,n,nRestarts=10,X=None,y=None,xStart=None,noise=None,optName='bfgs'):
        self.dimension=n
        self.alpha=np.ones(n)
        self.variance=[1.0]
        self.mu=[0.0]
        self.optimizationMethod=optName
        ####observations for the prior
        self.X=X
        self.y=y
        self.noise=noise
        self.optRuns=[]
        self.optPointsArray=[]
        self.restarts=nRestarts
        
    def getParamaters(self):
        dic={}
        dic['alphaPaper']=0.5*(self.alpha**2)
        dic['variance']=self.variance
        dic['mu']=self.mu
        return dic


    
    ###X,X2 are numpy matrices
    def K(self, X, X2=None,alpha=None,variance=None):
        if alpha is None:
            alpha=self.alpha
        if variance is None:
            variance=self.variance
            
        if X2 is None:
            X=X*alpha
            Xsq=np.sum(np.square(X), 1)
         #   r=-2.*tdot(X) + (Xsq[:, None] + Xsq[None, :])
            r=-2.*np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
            r = np.clip(r, 0, np.inf)
            return variance*np.exp(-0.5*r)
        else:
            X=X*alpha
            X2=X2*alpha
            r=-2.*np.dot(X, X2.T) + (np.sum(np.square(X), 1)[:, None] + np.sum(np.square(X2), 1)[None, :])
            r = np.clip(r, 0, np.inf)
            return variance*np.exp(-0.5*r)
    
    ###Computes the covariance matrix A on the points X, and adds the noise of each observation
    def A(self,X,X2=None,noise=None,alpha=None,variance=None):
        if noise is None:
            K=self.K(X,X2,alpha=alpha,variance=variance)
        else:
            print X
            print X2
            K=self.K(X,X2,alpha=alpha,variance=variance)+np.diag(noise)
        return K
    
    ##X is a matrix
    ###gradient respect to log(var),log(alpha**2)
    def logLikelihood(self,X,y,noise=None,alpha=None,variance=None,mu=None,gradient=False):
        if alpha is None:
            alpha=self.alpha
        if variance is None:
            variance=self.variance
        if mu is None:
            mu=self.mu
        if noise is None:
            K=self.A(X,alpha=alpha,variance=variance)
        else:
            K=self.A(X,alpha=alpha,variance=variance,noise=noise)
        y2=y-mu
        N=X.shape[0]
        try:
            L=np.linalg.cholesky(K)
            alp=inverseComp(L,y2)
            logLike=-0.5*np.dot(y2,alp)-np.sum(np.log(np.diag(L)))-0.5*N*np.log(2.0*np.pi)
            if gradient==False:
                return logLike
            gradient=np.zeros(self.dimension+2)
            
            ###0 to n-1, gradient respect to alpha
            ###n, gradient respect to log(variance)
            ###n+1,gradient respect to mu
            temp=np.dot(alp[:,None],alp[None,:])
            K2=self.A(X,alpha=alpha,variance=variance)
            for i in range(self.dimension):
                derivative=np.zeros((N,N))
                derivative=K2*(-0.5*(alpha[i]**2)*((X[:,i][:,None]-X[:,i][None,:])**2))
                temp3=inverseComp(L,derivative)
                gradient[i]=0.5*np.trace(np.dot(temp,derivative)-temp3)
            
            der=self.K(X,alpha=alpha,variance=variance)
            temp3=inverseComp(L,der)
            gradient[self.dimension]=0.5*np.trace(np.dot(temp,der)-temp3)

            der=np.ones((N,N))
            temp3=inverseComp(L,der)
            gradient[self.dimension+1]=0.5*np.trace(np.dot(temp,der)-temp3)
            return logLike,gradient
        except:
            L=np.linalg.inv(K)
            det=np.linalg.det(L)
            logLike=-0.5*np.dot(y2,np.dot(L,y2))-0.5*N*np.log(2*np.pi)-0.5*np.log(det)
            if gradient==False:
                return logLike
            gradient=np.zeros(self.dimension+2)
            
            alp=np.dot(L,y2)
            temp=np.dot(alp[:,None],alp.T[None,:])
            K2=self.A(X,alpha=alpha,variance=variance)
            for i in range(self.dimension):
                derivative=np.zeros((N,N))
                derivative=K2*(-1.0*alpha[i]*((X[:,i][:,None]-X[:,i][None,:])**2))
                temp2=np.dot(temp-L,derivative)
                gradient[i]=0.5*np.trace(temp2)
            
            temp2=np.dot(temp-L,K2)
            gradient[self.dimension]=0.5*np.trace(temp2)
            
            der=np.ones((N,N))
            temp2=np.dot(temp-L,der)
            gradient[self.dimension+1]=0.5*np.trace(temp2)
            return logLike,gradient
            

    def gradientLogLikelihood(self,X,y,noise=None,alpha=None,variance=None,mu=None):
        return self.logLikelihood(X,y,noise=noise,alpha=alpha,variance=variance,mu=mu,gradient=True)[1]
    
    
    def minuslogLikelihoodParameters(self,t):
        alpha=t[0:self.dimension]
        variance=np.exp(t[self.dimension])
        mu=t[self.dimension+1]
        return -self.logLikelihood(self.X,self.y,self.noise,alpha=alpha,variance=variance,mu=mu)
    
    def minusGradLogLikelihoodParameters(self,t):
        alpha=t[0:self.dimension]
        variance=np.exp(t[self.dimension])
        mu=t[self.dimension+1]
        return -self.gradientLogLikelihood(self.X,self.y,self.noise,alpha=alpha,variance=variance,mu=mu)

    
    
    ##optimizer is the name like 'bfgs'
    def optimizeKernel(self,start=None,optimizer=None,**kwargs):
        if start is None:
            start=np.concatenate((np.log(self.alpha**2),np.log(self.variance),self.mu))
        if optimizer is None:
            optimizer=self.optimizationMethod
        
        optimizer = getOptimizationMethod(optimizer)
        opt=optimizer(start,**kwargs)
        opt.run(f=self.minuslogLikelihoodParameters,df=self.minusGradLogLikelihoodParameters)
        self.optRuns.append(opt)
        self.optPointsArray.append(opt.xOpt)
        
        
    def trainnoParallel(self,scaledAlpha,**kwargs):
        dim=self.dimension
        alpha=np.random.randn(dim)
        variance=np.random.rand(1)
        st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),[0.0]))
        args2={}
        args2['start']=st
        job=misc.kernOptWrapper(self,**args2)
        temp=job.xOpt
        self.alpha=np.sqrt(np.exp(np.array(temp[0:self.dimension])))
        self.variance=np.exp(np.array(temp[self.dimension]))
        self.mu=np.array(temp[self.dimension+1])

    
    
    ###Train the hyperparameters using MLE
    ###noise is an array. X is a matrix y is an array
    def train(self,scaledAlpha,numStarts=None,numProcesses=None,**kwargs):
        if numStarts is None:
            numStarts=self.restarts
        
        try:
            dim=self.dimension
            jobs = []
            pool = mp.Pool(processes=numProcesses)
            for i in range(numStarts):
                alpha=np.random.randn(dim)
                variance=np.random.rand(1)
                st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),[0.0]))
                args2={}
                args2['start']=st
                job = pool.apply_async(misc.kernOptWrapper, args=(self,), kwds=args2)
                jobs.append(job)
            
            pool.close()  # signal that no more data coming in
            pool.join()  # wait for all the tasks to complete
        except KeyboardInterrupt:
            print "Ctrl+c received, terminating and joining pool."
            pool.terminate()
            pool.join()

        for i in range(numStarts):
            try:
                self.optRuns.append(jobs[i].get())
            except Exception as e:
                print "what"

                
        if len(self.optRuns):
            i = np.argmin([o.fOpt for o in self.optRuns])
            temp=self.optRuns[i].xOpt
            
            self.alpha=np.sqrt(np.exp(np.array(temp[0:self.dimension])))
            self.variance=np.exp(np.array(temp[self.dimension]))
            self.mu=np.array(temp[self.dimension+1])

    
    def printPar(self):
        print "alpha is "+self.alpha
        print "variance is "+self.variance
        print "mean is "+ self.mu
        
        

        
