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
    def __init__(self,kernel,dimKernel,dataObj,numberTraining,scaledAlpha=1.0):
        self._k=kernel
        self._Xhist=dataObj.Xhist
        self._yHist=dataObj.yHist
        self._noiseHist=dataObj.varHist
        self._numberTraining=numberTraining ##number of points used to train the kernel
        self._n=dimKernel
        self.scaledAlpha=scaledAlpha
        
        
        
class SBOGP(GaussianProcess):
    def __init__(self,B,dimNoiseW,dimPoints,gradXBforAn, computeLogProductExpectationsForAn,
                 *args,**kargs):
        GaussianProcess.__init__(self,*args,**kargs)
        self.SBOGP_name="SBO"
        self.n1=dimPoints
        self.n2=dimNoiseW
        self.B=B
        self.computeLogProductExpectationsForAn=computeLogProductExpectationsForAn
        self.gradXBforAn=gradXBforAn
    

        
    ##x is point where function is evaluated
    ##L is cholesky factorization of An
    ##X,W past points
    ##y2 are the past observations
    ##n is the time where aN is computed
    ##Output: aN and its gradient if gradient=True
    ##Other parameters are defined in Vn
    def aN_grad(self,x,L,n,dataObj,gradient=True,onlyGradient=False,logproductExpectations=None):
        n1=self.n1
        n2=self.n2
        muStart=self._k.mu
        y2=dataObj.yHist[0:n+self._numberTraining]-self._k.mu
        B=np.zeros(n+self._numberTraining)
        for i in xrange(n+self._numberTraining):
            B[i]=self.B(x,dataObj.Xhist[i,:],self.n1,self.n2,logproductExpectations[i])
        
        inv1=linalg.solve_triangular(L,y2,lower=True)
        
        if onlyGradient:
            gradXB=self.gradXBforAn(x,n,B,self._k,dataObj.Xhist[0:n+self._numberTraining,0:n1])
            temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
            gradAn=np.dot(inv1.transpose(),temp4)
            return gradAn
        
        
        inv2=linalg.solve_triangular(L,B.transpose(),lower=True)
        aN=muStart+np.dot(inv2.transpose(),inv1)
        if gradient==True:
            gradXB=self.gradXBforAn(x,n,B,self,dataObj.Xhist[0:n+self._numberTraining,0:n1])
            temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
            gradAn=np.dot(inv1.transpose(),temp4)
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
    
    
class KG(GaussianProcess):
    def __init__(self,dimPoints,gradXKern,gradXKern2,*args,**kargs):
        GaussianProcess.__init__(self,*args,**kargs)
        self.SBOGP_name="KG"
        self.n1=dimPoints
        self.gradXKern=gradXKern
        self.gradXKern2=gradXKern2

     ##return a_n and b_n
    ##x is a vector of points (x is as matrix) where a_n and sigma_n are evaluated  
    def aANDb(self,n,x,xNew):
        tempN=n+self._numberTraining
        x=np.array(x)
        xNew=xNew.reshape((1,self.n1))
        m=x.shape[0]
        A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
        L=np.linalg.cholesky(A)
        B=np.zeros([m,tempN])
        X=self._Xhist
        y=self._yHist

        for i in xrange(tempN):
            B[:,i]=self._k.K(x,X[i:i+1,:])[:,0]
        muStart=self._k.mu
        temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
        temp4=self._k.K(xNew,X)
        temp5=linalg.solve_triangular(L,temp4.T,lower=True)
        BN=self._k.K(xNew)[:,0]-np.dot(temp5.T,temp5)
        ####esto se debe cambiar en mas dimensiones!!!!!!!!!!!!!!!!!!!!!
       # a=np.zeros((m,1))
       # b=np.zeros((m,1))
        a=np.zeros(m)
        b=np.zeros(m)
        ##############
        for j in xrange(m):
            temp2=linalg.solve_triangular(L,B[j,:].T,lower=True)
            temp3=np.dot(temp2.T,temp5)
           # a[j,0]=muStart+np.dot(temp1.T,temp5)
           # b[j,0]=-temp3+self._k.K(x[j:j+1,:],xNew)[:,0]
           # b[j,0]=b[j,0]/sqrt(float(BN))
            a[j]=muStart+np.dot(temp2.T,temp1)
            b[j]=-temp3+self._k.K(x[j:j+1,:],xNew)[:,0]
            b[j]=b[j]/sqrt(float(BN))
        ######error check again!!!!!!!!!!
        ######
        return a,b,L

    def muN(self,x,n,grad=False):
        tempN=self._numberTraining+n
        x=np.array(x)
        X=self._Xhist[0:tempN,:]
        y=self._yHist[0:tempN,:]
        A=self._k.A(self._Xhist[0:tempN,:],noise=self._noiseHist[0:tempN])
 
        m=1
        L=np.linalg.cholesky(A)
        x=np.array(x).reshape((1,self.n1))
        B=np.zeros([m,tempN])
        
        muStart=self._k.mu
        for i in xrange(tempN):
            # if n>0:
              #  print "x"
               # print x
               # print X[i:i+1,:]
            B[:,i]=self._k.K(x,X[i:i+1,:])
        temp2=linalg.solve_triangular(L,B.T,lower=True)
       # print "ver"
      #  print y
     #   print L
        temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True)
        a=muStart+np.dot(temp2.T,temp1)
        if grad==False:
            return a
        x=np.array(x).reshape((1,self.n1))
#        gradX=np.zeros((n,self._n1))
        gradX=self.gradXKern(x,n,self)
        gradi=np.zeros(self.n1)
        for j in xrange(self.n1):
            temp2=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
            gradi[j]=np.dot(temp2.T,temp1)
        return a,gradi
        
class PIGP(GaussianProcess):
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
    
