#!/usr/bin/env python
import numpy as np
from math import *
from scipy import linalg
from numpy import linalg as LA
from scipy.stats import norm
import os
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
from matplotlib import pyplot as plt
#####borrar
#import VOIsboGaussian as voi1
#######

class GaussianProcess:
    def __init__(self,kernel,dimKernel,Xhist,yHist,noiseHist,numberTraining):
        self._k=kernel
        self._Xhist=Xhist
        self._yHist=yHist
        self._noiseHist=noiseHist
        self._numberTraining=numberTraining ##number of points used to train the kernel
        self._n=dimKernel
        
        
        
class SBOGP(GaussianProcess):
    def __init__(self,B,dimNoiseW,dimPoints,numberPoints,gradXBforAn,gradXBfunc=None,Bhist=None,histSaved=0,*args,**kargs):
        GaussianProcess.__init__(self,*args,**kargs)
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
        self._gradXBfunc=gradXBfunc
        self.gradXBforAn=gradXBforAn
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
        for i in xrange(self.histSaved,tempN):
            temp=self.B(x,self._Xhist[i,:],self.n1,self.n2) ###change my previous function because we have to concatenate X and W
            self.Bhist=np.concatenate((self.Bhist,temp.reshape((m,1))),1)
            self.histSaved+=1
        B=self.Bhist
        
        BN=np.zeros([m,1])
        n2=self.n2

        BN[:,0]=self.B(x,np.concatenate((xNew,wNew),1),self.n1,n2) #B(x,n+1)
        
        muStart=self._k.mu
        y=self._yHist
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
        a=muStart+np.dot(temp2.T,temp1)
        n1=self.n1
        n2=self.n2
        past=self._Xhist[0:tempN,:]
        new=np.concatenate((xNew,wNew),1).reshape((1,n1+n2))
    #    gamma=np.transpose(self._k.A(new,past,noise=self._noiseHist))
        gamma=np.transpose(self._k.A(new,past))
        temp1=linalg.solve_triangular(L,gamma,lower=True)
        b=(BN-np.dot(temp2.T,temp1))
        b2=self._k.K(new)-np.dot(temp1.T,temp1)

        b2=np.clip(b2,0,np.inf)
        try:
            b=b/(np.sqrt(b2))
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
    def aN_grad(self,x,L,n,gradient=True):
        y2=self._yHist[0:n+self._numberTraining]-self._k.mu
        muStart=self._k.mu
        n1=self.n1
        n2=self.n2
        B=np.zeros(n+self._numberTraining)
        for i in xrange(n+self._numberTraining):
            B[i]=self.B(x,self._Xhist[i,:],self.n1,self.n2)
        inv1=linalg.solve_triangular(L,y2,lower=True)
        inv2=linalg.solve_triangular(L,B.transpose(),lower=True)
        aN=muStart+np.dot(inv2.transpose(),inv1)
        if gradient==True:
           # gradXB=self._gradXBfunc(x,self,BN,keep)
          #  gradXB=np.zeros((n1,n+self.histSaved))
            gradXB=self.gradXBforAn(x,n,B,self,self._Xhist[0:n+self._numberTraining,0:n1])
            temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
            temp5=linalg.solve_triangular(L,y2,lower=True)
            gradAn=np.dot(temp5.transpose(),temp4)
            return aN,gradAn
        else:
            return aN
        
    ####particular to this problem
    def plotAn(self,i,L,points,seed):
        m=points.shape[0]
        z=np.zeros(m)
        for j in xrange(m):
            z[j]=self.aN_grad(points[j,:],L,i,gradient=False)
         #   z2[j]=voi1.aN_grad(points[j,:],L,y2,X,W,n1,n2,variance0,alpha1,alpha2,muStart,tempN,gradient=False)
        
        fig=plt.figure()
        plt.plot(points,-(points**2),label="G(x)")
        plt.plot(points,z,'--',label='$a_%d(x)$'%i)
        
        plt.xlabel('x',fontsize=26)
        plt.legend()
        plt.savefig(os.path.join('%d'%seed+"run",'%d'%i+"a_n.pdf"))
        plt.close(fig)
    

class EIGP(GaussianProcess):
    def __init__(self,dimPoints,gradXKern,*args,**kargs):
        GaussianProcess.__init__(self,*args,**kargs)
        self.SBOGP_name="GP_EI"
        self.n1=dimPoints
        self.gradXKern=gradXKern
    
    def muN(self,x,n,grad=False):
        x=np.array(x)
        m=1
        tempN=self._numberTraining+n
        X=self._Xhist[0:tempN,:]
        A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
        L=np.linalg.cholesky(A)
        x=np.array(x).reshape((1,self.n1))
        B=np.zeros([m,tempN])
        
        for i in xrange(tempN):
            B[:,i]=self._k.K(x,X[i:i+1,:])
            
        y=self._yHist[0:tempN,:]
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        muStart=self._k.mu
        temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
        a=muStart+np.dot(temp2.T,temp1)
        if grad==False:
            return a
        x=np.array(x).reshape((1,self.n1))
       # gradX=np.zeros((n,self.n1))
        gradX=self.gradXKern(x,n,self)
        gradi=np.zeros(self.n1)
        temp3=linalg.solve_triangular(L,y-muStart,lower=True)
        
        for j in xrange(self.n1):
           # for i in xrange(n):
           #     gradX[i,j]=self._k.K(x,X[i,:].reshape((1,self._n1)))*(2.0*self._alpha1[j]*(x[0,j]-X[i,j]))
            temp2=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
            gradi[j]=muStart+np.dot(temp2.T,temp3)
        return a,gradi
    
    
    def varN(self,x,n,grad=False):
        temp=self._k.K(np.array(x).reshape((1,self.n1)))
        tempN=self._numberTraining+n
        sigmaVec=np.zeros((tempN,1))
        for i in xrange(tempN):
            sigmaVec[i,0]=self._k.K(np.array(x).reshape((1,self.n1)),self._Xhist[i:i+1,:])[:,0]
        A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
        L=np.linalg.cholesky(A)
        temp3=linalg.solve_triangular(L,sigmaVec,lower=True)
        temp2=np.dot(temp3.T,temp3)
        temp2=temp-temp2
        if grad==False:
            return temp2
        else:
            gradi=np.zeros(self.n1)
            x=np.array(x).reshape((1,self.n1))

            gradX=self.gradXKern(x,n,self)
            #gradX=np.zeros((n,self._n1))
            for j in xrange(self.n1):
              #  for i in xrange(n):
                  #  gradX[i,j]=self._k.K(x,self._X[i,:].reshape((1,self._n1)))*(2.0*self._alpha1[j]*(x[0,j]-self._X[i,j]))
                temp5=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
                gradi[j]=np.dot(temp5.T,temp3)
            gradVar=-2.0*gradi
            return temp2,gradVar
    
    
