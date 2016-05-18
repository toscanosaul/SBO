#!/usr/bin/env python

"""
This file defines the kernels. We can optimize the hyperparameters,
compute the log-likelihood and the matrix A from the paper [tf].
"""

import numpy as np
from scipy import linalg
from scipy.optimize import fmin_l_bfgs_b
from . matrixComputations import tripleProduct,inverseComp
from scipy.stats import multivariate_normal
import multiprocessing as mp
from . import misc
from . import optimization
from scipy import array, linalg, dot
from scipy.spatial.distance import cdist

SQRT_3 = np.sqrt(3.0)
SQRT_5 = np.sqrt(5.0)

class CORRE:
    def __init__(self,n,nIS,scaleAlpha=None,nRestarts=10,X=None,y=None,
                 noise=None,complete=True,optName='bfgs'):
        """
        Defines the squared exponential kernel,
            variance*exp(-0.5*sum (alpha_i/scaleAlpha)**2 *(x_i-y_i)**2))
        
        Args:
            -n: Dimension of the domain of the kernel.
            -scaleAlpha: The hyperparameters of the kernel are scaled by
                         alpha/(scaledAlpha^{2}).
            -nRestarts: Number of restarts to optimze the hyperparameters.
            -X: Training data.
            -y: Outputs of the training data.
            -noise: Noise of the outputs.
        """
        if scaleAlpha is None:
            scaleAlpha=np.ones(n)
        self.scaleAlpha=scaleAlpha
        self.dimension=n
        self.nIS=nIS
        self.matrix=np.identity(nIS)
        self.alpha=np.ones(n)
        self.variance=np.array([1.0])
        self.mu=[0.0]
        self.optimizationMethod=optName

        self.X=X
        self.y=y
        self.noise=noise
        self.optRuns=[]
        self.optPointsArray=[]
        self.restarts=nRestarts
        
        if complete:
            
            N=len(self.y)/(self.nIS)
            Y=np.zeros((N,self.nIS))
            X2=np.zeros((N,n))
            for i in range(N):
                for j in range(self.nIS):
                    Y[i,j]=self.y[(i-1)*self.nIS+j]
                X2[i,:]=X[(i-1)*self.nIS,0:n]
            self.X2=X2
            self.Y=Y
            self.N=N
        
    def getParamaters(self):
        """
        Returns a dictionary with the hyperparameters and the mean
        of the GP.
        """
        dic={}
        dic['alphaPaper']=(self.alpha**2)/(self.scaleAlpha**2)
        dic['variance']=self.variance
        dic['mu']=self.mu
        return dic
    
    def getParamaters2(self):
        """
        Returns a dictionary with the hyperparameters and the mean
        of the GP.
        """
        dic={}
        dic['alphaPaper']=(self.alpha**2)/(self.scaleAlpha**2)
        dic['variance']=self.variance
        dic['matrix']=self.matrix
        return dic

    def K(self, X, X2=None,alpha=None,variance=None,covM=None,distances=False):
        """
        Computes the covariance matrix cov(X[i,:],X2[j,:]).
        
        Args:
            X: Matrix where each row is a point.
            X2: Matrix where each row is a point.
            alpha: It's the scaled alpha.
            Variance: Sigma hyperparameter.
            cholesky: Decomposition of the matrix of the IS. Lower triangular
                      matrix. 
            
        """
       # covM=np.dot(cholesky,cholesky.transpose())
        if alpha is None:
            alpha=self.alpha
        if variance is None:
            variance=self.variance
        if covM is None:
            covM=self.matrix
        
        

        z=X[:,self.dimension]
        X=X[:,0:self.dimension]
        if X2 is None:
            X=X*alpha/self.scaleAlpha
            X2=X
            z2=z
        else:

            z2=X2[:,self.dimension]
            X2=X2[:,0:self.dimension]
            X=X*alpha/self.scaleAlpha
            X2=X2*alpha/self.scaleAlpha
            
        r2=np.abs(cdist(X,X2,'sqeuclidean'))
        r=np.sqrt(r2)
        cov=(1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp (-SQRT_5* r)
        
        if distances:
            return distance*cov,r,r2
        
        nP=len(z)
        C=np.zeros((nP,nP))
        s,t=np.meshgrid(z,z2)
        s=s.astype(int)
        t=t.astype(int)

        T=covM[s,t].transpose()
        


            
        return (variance*cov)*T
    
    def K2(self, X, X2=None,alpha=None,variance=None,distances=False):
        """
        Computes the covariance matrix cov(X[i,:],X2[j,:]).
        
        Args:
            X: Matrix where each row is a point.
            X2: Matrix where each row is a point.
            alpha: It's the scaled alpha.
            Variance: Sigma hyperparameter.
            cholesky: Decomposition of the matrix of the IS. Lower triangular
                      matrix. 
            
        """
       # covM=np.dot(cholesky,cholesky.transpose())
        if alpha is None:
            alpha=self.alpha
        if variance is None:
            variance=self.variance
 
        
        
        
      #  z=X[:,self.dimension]
        X=X[:,0:self.dimension]
        if X2 is None:
            X=X*alpha/self.scaleAlpha
            X2=X
          #  z2=z
        else:

         #   z2=X2[:,self.dimension]
            X2=X2[:,0:self.dimension]
            X=X*alpha/self.scaleAlpha
            X2=X2*alpha/self.scaleAlpha
            
        r2=np.abs(cdist(X,X2,'sqeuclidean'))
        r=np.sqrt(r2)
        cov=(1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp (-SQRT_5* r)
        
        if distances:
            return distance*cov,r,r2
        
      #  nP=len(z)
      #  C=np.zeros((nP,nP))
      #  s,t=np.meshgrid(z,z2)
      #  s=s.astype(int)
      #  t=t.astype(int)

      #  T=covM[s,t].transpose()
        


            
        return (variance*cov)
    
    def A(self,X,X2=None,noise=None,alpha=None,variance=None,covM=None):
        """
        Computes the covariance matrix A on the points X, and adds
        the noise of each observation.
        
        Args:
            X: Matrix where each row is a point.
            X2: Matrix where each row is a point.
            noise: Noise of the observations.
            alpha: Hyperparameters of the kernel.
            Variance: Sigma hyperparameter.
        """
        if noise is None:
            K=self.K(X,X2,alpha=alpha,variance=variance,covM=covM)
        else:
            K=self.K(X,X2,alpha=alpha,variance=variance,covM=covM)+np.diag(noise)
        return K
    
    def A2(self,X,X2=None,noise=None,alpha=None,variance=None):
        """
        Computes the covariance matrix A on the points X, and adds
        the noise of each observation.
        
        Args:
            X: Matrix where each row is a point.
            X2: Matrix where each row is a point.
            noise: Noise of the observations.
            alpha: Hyperparameters of the kernel.
            Variance: Sigma hyperparameter.
        """
        if noise is None:
            K=self.K2(X,X2,alpha=alpha,variance=variance)
        else:

            K=self.K2(X,X2,alpha=alpha,variance=variance)
        return K
    
    def logLikelihood2(self,X,y,noise=None,alpha=None,variance=None,mu=None,gradient=False):
        """
        Computes the log-likelihood and its gradient. The gradient is respect to  log(var)
        and log(alpha**2).
        
        Args:
            -X: Matrix with the training data.
            -y: Output of the training data.
            -noise: Noise of the outputs.
            -alpha: Hyperparameters of the kernel
            -variance: Hyperparameter of the kernel.
            -mu: Mean parameter of the GP.
            -gradient: True if we want the gradient; False otherwise.
        """
        if alpha is None:
            alpha=self.alpha
        if variance is None:
            variance=self.variance
        if mu is None:
            mu=self.mu
       # print chol
        

       # print chol2
 
        if noise is None:
            K=self.A2(X,alpha=alpha,variance=variance)
        else:
            K=self.A2(X,alpha=alpha,variance=variance,noise=noise)
   
      #  y2=y-mu
        N=X.shape[0]
     #   try:
       # print K
      #  y2=self.Y
        L=np.linalg.cholesky(K)
        Y=self.Y
        alp= linalg.solve_triangular(L,Y,lower=True)
        DET=np.sum(np.log(np.diag(L)))
        prod=np.dot(alp.transpose(),alp)
        logLike=-self.N*np.log(np.linalg.det(prod))-2.0*self.nIS*DET
        

        
       # print np.log(np.linalg.det(prod3))
        ##check derivative

        if gradient==False:
            return logLike
       # entries=int(float(self.nIS)*(float(self.nIS+1))/2.0)
       
    #    print "check"
     #   var2=(np.log(variance))
     #   dh=0.00001
     #   var2=dh+var2
       # print alpha2
     #   var2=(np.exp(var2))
      #  temp1=self.logLikelihood2(X,y,noise=None,alpha=alpha,variance=var2)
      #  print (temp1-logLike)/dh
        
        gradient=np.zeros(self.dimension+1)
        
        temp=prod
        #temp=np.dot(alp[:,None],alp[None,:])
        K2=K
     #   Xcopy=np.array(X)
      #  z=X[:,self.dimension]
   #     X=X[:,0:self.dimension]
        
        X2=X
        dist = np.sqrt(np.sum(np.square((X[:,None,:]-X2[None,:,:])*(alpha/self.scaleAlpha)),-1))
        invdist = 1./np.where(dist!=0.,dist,np.inf)
        dist2M = (np.square(X[:,None,:]-X2[None,:,:])*(alpha**2))/(2.0*(self.scaleAlpha**2))
    
        derivative=np.zeros((N,N))
        dl = -(np.array(variance) * 5./3 * dist * (1 + np.sqrt(5.)*dist ) * np.exp(-np.sqrt(5.)*dist))[:,:,np.newaxis] * dist2M*invdist[:,:,np.newaxis]
    
        
       # nP=len(z)
       
       # C=np.zeros((nP,nP))
       # s,t=np.meshgrid(z,z)
       # s=s.astype(int)
       # t=t.astype(int)
        
       # T=covM[s,t].transpose()
   
      #  Linv=np.linalg.inv(L)
       
        prod2Inv=np.linalg.inv(prod)
        for i in range(self.dimension):
            derivative=dl[:,:,i]
            Linv2=linalg.solve_triangular(L,derivative,lower=True)
            tmp4=np.dot(alp.transpose(),Linv2)
            
            tmp5=linalg.solve_triangular(L,tmp4.transpose(),lower=True)
            tmp6=linalg.solve_triangular(L.transpose(),tmp5,lower=False)
            tmp8=np.dot(Y.transpose(),tmp6)
            tmp8=tmp8.transpose()
            
            aux1=np.dot(prod2Inv,tmp8)
 
            aux2=linalg.solve_triangular(L.transpose(),Linv2,lower=False)
            final=self.N*np.trace(aux1)-self.nIS*np.trace(aux2)
            

            gradient[i]=final

        

        gradient[self.dimension]=0.0

        return logLike,gradient
    
    def logLikelihood(self,X,y,noise=None,alpha=None,variance=None,mu=None,chol=None,gradient=False):
        """
        Computes the log-likelihood and its gradient. The gradient is respect to  log(var)
        and log(alpha**2).
        
        Args:
            -X: Matrix with the training data.
            -y: Output of the training data.
            -noise: Noise of the outputs.
            -alpha: Hyperparameters of the kernel
            -variance: Hyperparameter of the kernel.
            -mu: Mean parameter of the GP.
            -gradient: True if we want the gradient; False otherwise.
        """
        if alpha is None:
            alpha=self.alpha
        if variance is None:
            variance=self.variance
        if mu is None:
            mu=self.mu
       # print chol
        
        chol2=np.zeros((self.nIS,self.nIS))
        cont=0
        for i in range(self.nIS):
            for j in range(i+1):
                chol2[i,j]=chol[cont]
                cont+=1
       # print chol2
        covM=np.dot(chol2,chol2.transpose())
        if noise is None:
            K=self.A(X,alpha=alpha,variance=variance,covM=covM)
        else:
            K=self.A(X,alpha=alpha,variance=variance,noise=noise,covM=covM)
   
        y2=y-mu
        N=X.shape[0]
     #   try:
       # print K
        L=np.linalg.cholesky(K)
        
        alp=inverseComp(L,y2)
        logLike=-0.5*np.dot(y2,alp)-np.sum(np.log(np.diag(L)))-0.5*N*np.log(2.0*np.pi)
        if gradient==False:
            return logLike
        entries=int(float(self.nIS)*(float(self.nIS+1))/2.0)
        gradient=np.zeros(self.dimension+1+entries)
        
        temp=np.dot(alp[:,None],alp[None,:])
        K2=self.A(X,alpha=alpha,variance=variance,covM=covM)
        Xcopy=np.array(X)
        z=X[:,self.dimension]
        X=X[:,0:self.dimension]
        
        X2=X
        dist = np.sqrt(np.sum(np.square((X[:,None,:]-X2[None,:,:])*(alpha/self.scaleAlpha)),-1))
        invdist = 1./np.where(dist!=0.,dist,np.inf)
        dist2M = (np.square(X[:,None,:]-X2[None,:,:])*(alpha**2))/(2.0*(self.scaleAlpha**2))
    
        derivative=np.zeros((N,N))
        dl = -(np.array(variance) * 5./3 * dist * (1 + np.sqrt(5.)*dist ) * np.exp(-np.sqrt(5.)*dist))[:,:,np.newaxis] * dist2M*invdist[:,:,np.newaxis]
    
        
        nP=len(z)
        C=np.zeros((nP,nP))
        s,t=np.meshgrid(z,z)
        s=s.astype(int)
        t=t.astype(int)
        
        T=covM[s,t].transpose()
   

        for i in range(self.dimension):
            derivative=dl[:,:,i]*T

            temp3=inverseComp(L,derivative)
            gradient[i]=0.5*np.trace(np.dot(temp,derivative)-temp3)

        
        der=K2
        temp3=inverseComp(L,der)
        gradient[self.dimension]=0.5*np.trace(np.dot(temp,der)-temp3)
        
     #   der=np.ones((N,N))
     #   temp3=inverseComp(L,der)
     #   gradient[self.dimension+1]=0.5*np.trace(np.dot(temp,der)-temp3)
        
        N2=self.dimension+1
        der=K2
        der2=np.zeros((N,N))
        
        temp3=inverseComp(L,der)
        
       # index=[np.where(z==j)[0] for j in range(self.nIS)]
        cont=0
     #   for j in range(self.nIS):
      #      for i in range(self.nIS):
     #   index1=[[(r,s) if z[r]==z[s] and z[r]==i for r in xrange(N) and s in xrange(N)] for i in range(self.nIS)]
        
        for i in range(self.nIS):
          #  index0=index[j]
         
            for j in range(i+1):
                der2=np.zeros((N,N))
              #  index1=index[i]
                for r in xrange(N):
                    for s in xrange(N):
                        ind=int(i*(i+1)/2)+j
                        p1=z[r]
                        p2=z[s]
                        if p1==p2 and p1==i and j<=p2:
                            der2[r,s]=(K2[r,s]/covM[p1,p2])*(2.0*chol[ind]*chol[ind])
                        elif p1!=p2 and p1==i and j<=min(p1,p2):
                            ind2=int(p2*(p2+1)/2)+j
                            der2[r,s]=(K2[r,s]/covM[p1,p2])*(chol[ind]*chol[ind2])
                        elif p1!=p2 and p2==i and j<=min(p1,p2):
             
                            ind2=int(p1*(p1+1)/2)+j
                        
                            der2[r,s]=(K2[r,s]/covM[p1,p2])*(chol[ind]*chol[ind2])
             #   for r in index0:
             #       for s in index1:
                
                
                temp3=inverseComp(L,der2)
                gradient[N2+cont]=0.5*np.trace(np.dot(temp,der2)-temp3)
                cont+=1
        
    

        return logLike,gradient
       # except:
        #    print "no"

            
    def gradientLogLikelihood(self,X,y,noise=None,alpha=None,variance=None,mu=None,chol=None):
        """
        Computes the gradient of the log-likelihood, respect to log(var)
        and log(alpha**2).
        
        Args:
            -X: Matrix with the training data.
            -y: Output of the training data.
            -noise: Noise of the outputs.
            -alpha: Hyperparameters of the kernel
            -variance: Hyperparameter of the kernel.
            -mu: Mean parameter of the GP.
            -gradient: True if we want the gradient; False otherwise.
        """
        return self.logLikelihood(X,y,noise=noise,alpha=alpha,variance=variance,mu=np.array([0.0]),chol=chol,gradient=True)[1]
    
    def gradientLogLikelihood2(self,X,y,noise=None,alpha=None,variance=None,mu=None):
        """
        Computes the gradient of the log-likelihood, respect to log(var)
        and log(alpha**2).
        
        Args:
            -X: Matrix with the training data.
            -y: Output of the training data.
            -noise: Noise of the outputs.
            -alpha: Hyperparameters of the kernel
            -variance: Hyperparameter of the kernel.
            -mu: Mean parameter of the GP.
            -gradient: True if we want the gradient; False otherwise.
        """
        return self.logLikelihood2(X,y,noise=noise,alpha=alpha,variance=variance,mu=np.array([0.0]),gradient=True)[1]
    
    def minuslogLikelihoodParameters(self,t):
        """
        Computes the minus log-likelihood.
        
        Args:
            t: hyperparameters of the kernel.
        """
    
        alpha=t[0:self.dimension]
        variance=(t[self.dimension])
       # mu=t[self.dimension+1]
        chol=(t[self.dimension+1:])
        return -self.logLikelihood(self.X,self.y,self.noise,alpha=alpha,variance=variance,mu=np.array([0.0]),chol=chol)
    
    def minusGradLogLikelihoodParameters(self,t):
        """
        Computes the gradient of the minus log-likelihood.
        
        Args:
            t: hyperparameters of the kernel.
        """
        alpha=t[0:self.dimension]
        variance=(t[self.dimension])
     #   mu=t[self.dimension+1]
        chol=(t[self.dimension+1:])
        return -self.gradientLogLikelihood(self.X,self.y,self.noise,alpha=alpha,variance=variance,mu=np.array([0.0]),chol=chol)
    
    def minuslogLikelihoodParameters2(self,t):
        """
        Computes the minus log-likelihood.
        
        Args:
            t: hyperparameters of the kernel.
        """
    
        alpha=np.sqrt(np.exp(t[0:self.dimension]))
        variance=np.exp(t[self.dimension])
       # mu=t[self.dimension+1]
     #   chol=(t[self.dimension+1:])
        return -self.logLikelihood2(self.X2,self.Y,self.noise,alpha=alpha,variance=variance,mu=np.array([0.0]))
    
    def minusGradLogLikelihoodParameters2(self,t):
        """
        Computes the gradient of the minus log-likelihood.
        
        Args:
            t: hyperparameters of the kernel.
        """
      
        alpha=np.sqrt(np.exp(t[0:self.dimension]))
        variance=np.exp(t[self.dimension])
     #   mu=t[self.dimension+1]
      #  chol=(t[self.dimension+1:])
        return -self.gradientLogLikelihood2(self.X2,self.Y,self.noise,alpha=alpha,variance=variance,mu=np.array([0.0]))

    def optimizeKernel(self,start=None,optimizer=None,**kwargs):
        """
        Optimize the minus log-likelihood using the optimizer method and starting in start.
        
        Args:
            start: starting point of the algorithm.
            optimizer: Name of the optimization algorithm that we want to use;
                       e.g. 'bfgs'.
            
        """
        if start is None:
            start=np.concatenate((np.log(self.alpha**2),np.log(self.variance)))
        if optimizer is None:
            optimizer=self.optimizationMethod
        
        optimizer = optimization.getOptimizationMethod(optimizer)
        opt=optimizer(start,**kwargs)
        opt.run(f=self.minuslogLikelihoodParameters,df=self.minusGradLogLikelihoodParameters)
        self.optRuns.append(opt)
        self.optPointsArray.append(opt.xOpt)
        
    def optimizeKernel2(self,start=None,optimizer=None,**kwargs):
        """
        Optimize the minus log-likelihood using the optimizer method and starting in start.
        
        Args:
            start: starting point of the algorithm.
            optimizer: Name of the optimization algorithm that we want to use;
                       e.g. 'bfgs'.
            
        """
        if start is None:
            start=np.concatenate((np.log(self.alpha**2),np.log(self.variance)))
        if optimizer is None:
            optimizer=self.optimizationMethod
        
        optimizer = optimization.getOptimizationMethod(optimizer)
        opt=optimizer(start,**kwargs)
        opt.run(f=self.minuslogLikelihoodParameters2,df=self.minusGradLogLikelihoodParameters2)
        self.optRuns.append(opt)
        self.optPointsArray.append(opt.xOpt)
    
    
    def trainnoParallel(self,scaledAlpha,**kwargs):
        """
        Train the hyperparameters starting in only one point the algorithm.
        
        Args:
            -scaledAlpha: The definition may be found above.
        """
        dim=self.dimension
        alpha=np.random.randn(dim)
        variance=np.random.rand(1)
        entries=int(self.nIS*(self.nIS+1)/2.0)

        l=np.random.rand(entries)
        st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),np.exp(l)))

        
        args2={}
        args2['start']=st
        job=misc.kernOptWrapper(self,**args2)
        temp=job.xOpt
        self.alpha=np.sqrt(np.exp(np.array(temp[0:self.dimension])))
        self.variance=np.exp(np.array(temp[self.dimension]))
      #  self.mu=np.array(temp[self.dimension+1])
        
        chol=np.zeros((self.nIS,self.nIS))
        cont=0
        for i in range(self.nIS):
            for j in range(i+1):
                chol[i,j]=np.exp(temp[self.dimension+1+cont])
                cont+=1
                
            
        self.matrix=np.dot(chol,chol.transpose())
        
    ##noiseless and block case
    def trainnoParallel2(self,scaledAlpha,**kwargs):
        """
        Train the hyperparameters starting in only one point the algorithm.
        
        Args:
            -scaledAlpha: The definition may be found above.
        """
        dim=self.dimension
        alpha=np.random.randn(dim)
        variance=np.random.rand(1)
       # entries=int(self.nIS*(self.nIS+1)/2.0)

      #  l=np.random.rand(entries)
       # st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance)))
        st=np.concatenate((alpha,variance))

        
        args2={}
        args2['start']=st
        job=misc.kernOptWrapper2(self,**args2)
        temp=job.xOpt
        self.alpha=np.sqrt(np.exp(np.array(temp[0:self.dimension])))
        self.variance=np.exp(np.array(temp[self.dimension]))
      #  self.mu=np.array(temp[self.dimension+1])
        
       # chol=np.zeros((self.nIS,self.nIS))
       # cont=0
       # for i in range(self.nIS):
       #     for j in range(i+1):
       #         chol[i,j]=np.exp(temp[self.dimension+1+cont])
       #         cont+=1
        Y=self.Y
        K=self.A2(self.X2,None)
        L=np.linalg.cholesky(K)

        alpha=linalg.solve_triangular(L,Y,lower=True)
        self.matrix=np.dot(alpha.transpose(),alpha)/self.N
        
        print self.matrix
        print self.variance
        print self.alpha


       
        
        
    def train2(self,scaledAlpha,numStarts=None,numProcesses=None,**kwargs):
        """
        Train the hyperparameters starting in several different points.
        
        Args:
            -scaledAlpha: The definition may be found above.
            -numStarts: Number of restarting times oft he algorithm.
        """
        if numStarts is None:
            numStarts=self.restarts
        try:
            dim=self.dimension
            jobs = []
            args3=[]
            pool = mp.Pool(processes=numProcesses)
            alpha=np.random.randn(numStarts,dim)
          #  entries=int(self.nIS*(self.nIS+1)/2.0)
            variance=np.random.rand(numStarts,1)
         #   tempZero=np.zeros((numStarts,1))
         #   l=np.random.rand(numStarts,entries)
            st=np.concatenate((alpha,variance),1)
            for i in range(numStarts):
               # alpha=np.random.randn(dim)
               # variance=np.random.rand(1)
               # st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),[0.0]))
               # args2={}
               # args2['start']=st
               # args3.append(args2.copy())
                job = pool.apply_async(misc.kernOptWrapper2, args=(self,st[i,:],))
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
          #  self.mu=np.array(temp[self.dimension+1])
            Y=self.Y
            K=self.A2(self.X2,None)
            L=np.linalg.cholesky(K)
        
            alpha=linalg.solve_triangular(L,Y,lower=True)
            self.matrix=np.dot(alpha.transpose(),alpha)/self.N
           # self.matrix=np.dot(chol,chol.transpose())

    def train(self,scaledAlpha,numStarts=None,numProcesses=None,**kwargs):
        """
        Train the hyperparameters starting in several different points.
        
        Args:
            -scaledAlpha: The definition may be found above.
            -numStarts: Number of restarting times oft he algorithm.
        """
        if numStarts is None:
            numStarts=self.restarts
        try:
            dim=self.dimension
            jobs = []
            args3=[]
            pool = mp.Pool(processes=numProcesses)
            alpha=np.random.randn(numStarts,dim)
            entries=int(self.nIS*(self.nIS+1)/2.0)
            variance=np.random.rand(numStarts,1)
         #   tempZero=np.zeros((numStarts,1))
            l=np.random.rand(numStarts,entries)
            st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),np.exp(l)),1)
            for i in range(numStarts):
               # alpha=np.random.randn(dim)
               # variance=np.random.rand(1)
               # st=np.concatenate((np.sqrt(np.exp(alpha)),np.exp(variance),[0.0]))
               # args2={}
               # args2['start']=st
               # args3.append(args2.copy())
                job = pool.apply_async(misc.kernOptWrapper, args=(self,st[i,:],))
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
          #  self.mu=np.array(temp[self.dimension+1])
            chol=np.zeros((self.nIS,self.nIS))
            cont=0
            for i in range(self.nIS):
                for j in range(i+1):
                    chol[i,j]=np.exp(temp[self.dimension+1+cont])
                    cont+=1
            self.matrix=np.dot(chol,chol.transpose())
            

    
    def printPar(self):
        """
        Print the hyperparameters of the kernel.
        """
        print "alpha is "+self.alpha
        print "variance is "+self.variance
        print "mean is "+ self.mu
        
        
        


        


