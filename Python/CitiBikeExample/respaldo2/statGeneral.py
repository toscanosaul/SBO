#!/usr/bin/env python
import numpy as np
from math import *
from scipy import linalg
from numpy import linalg as LA
from scipy.stats import norm

class GaussianProcess:
    def __init__(self,kernel,dimKernel,Xhist,yHist,noiseHist,numberTraining):
        self._k=kernel
        self._Xhist=Xhist
        self._yHist=yHist
        self._noiseHist=noiseHist
        self._numberTraining=numberTraining ##number of points used to train the kernel
        self._n=dimKernel
        
        
        
class SBOGP(GaussianProcess):
    def __init__(self,*args,**kargs):
        VOI.__init__(self,kernel,B,dimNoiseW,dimPoints,numberPoints,Bhist=None,histSaved=0,**kargs)
        self.SBOGP_name="SBO"
        self.n1=dimPoints
        self.n2=dimNoiseW
        #compute B(x,i). X=X[i,:],W[i,:]. x is a matrix of dimensions nxm where m is the dimension of an element of x.
        ##Remember that B(x,i)=integral_(sigma(x,w,x_i,w_i))dp(w)
        self.B=B
        if Bhist is None:
            Bhist=np.zeros((numberPoints,0))
        self.Bhist=Bhist
        self.histSaved=histSaved
        ####Include Ahist, with Lhist
    
    #computes a and b from the paper
    ##x is a nxdim(x) matrix of points where a_n and sigma_n are evaluated
    ###computed using n past observations
    def aANDb(self,n,x,xNew,wNew):
        x=np.array(x)
        m=x.shape[0]
        tempN=self._numberTraining+n
        A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
        L=np.linalg.cholesky(A)
        for i in xrange((self.histSaved,tempN)):
            temp=self.B(x,self._Xhist[i,:],self.n1,self.n2) ###change my previous function because we have to concatenate X and W
            self.Bhist=np.concatenate((self.Bhist,temp),1)
            self.histSaved+=1
        B=self.Bhist
        BN=np.zeros([m,1])
        n2=self.n2
        BN[:,0]=self.B(x,np.concatenate((xNew,wNew),1),self.n1,n2) #B(x,n+1)
        
        muStart=self._k.mu
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True) 
        a=muStart+np.dot(temp2.T,temp1)
        
        past=self._Xhist[0:tempN,:]
        new=np.concatenate((xNew,wNew),1).reshape((1,n1+n2))
        gamma=np.transpose(self._k.A(new,past,noise=self._noiseHist))
        temp1=linalg.solve_triangular(L,gamma,lower=True)
        b=(BN-np.dot(temp2.T,temp1))
        b2=self._k.K(new)-np.dot(temp1.T,temp1)
        b2=np.clip(b2,0,np.inf)
        try:
            b=b/(sqrt(b2))
        except Exception as e:
            print "use a different point x"
            b=np.zeros((len(b),1))
        return a,b,gamma,BN,L
        
    ##x is point where function is evaluated
    ##L is cholesky factorization of An
    ##X,W past points
    ##y2 are the past observations
    ##n is the time where aN is computed
    ##Output: aN and its gradient if gradient=True
    ##Other parameters are defined in Vn
    def aN_grad(x,L,n,gradient=True):
        y2=self._yHist[0:n+self._numberTraining]-self._k.mu
        muStart=self._k.mu
        n1=self.n1
        n2=self.n2
        B=np.zeros(n+self.histSaved)
        for i in xrange(n+self.histSaved):
            B[i]=self.B(x,self._Xhist[i,:],self.n1,self.n2)
        inv1=linalg.solve_triangular(L,y2,lower=True)
        inv2=linalg.solve_triangular(L,B.transpose(),lower=True)
        aN=muStart+np.dot(inv2.transpose(),inv1)
        if gradient==True:
            gradXB=np.zeros((n1,n+self.histSaved))
            for i in xrange(n+self.histSaved):
                gradXB[:,i]=self._gradXBfunc(x,i)
             #   gradXB[:,i]=B[i]*(-2.0*alpha1*(x-X[i,:]))
            temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
            temp5=linalg.solve_triangular(L,y2,lower=True)
            gradAn=np.dot(temp5.transpose(),temp4)
            return aN,gradAn
        else:
            return aN
    
    
    
    
    