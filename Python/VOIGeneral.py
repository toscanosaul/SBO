#!/usr/bin/env python

import numpy as np
from math import *
from AffineBreakPoints import *
import statGeneral as stat
from scipy import linalg
from numpy import linalg as LA
from scipy.stats import norm

class VOI:
    #####x is a nxdim(x) matrix of points where a_n and sigma_n are evaluated
    ###B(x,x_1),...B(x,x_n), B is the function to compute those
    def __init__(self,kernel,dimKernel,numberTraining,gradXWSigmaOfunc=None,Bhist=None,pointsApproximation=None,
                 gradXBfunc=None,B=None,PointsHist=None,yHist=None,noiseHist=None,gradWBfunc=None,BhistSaved=0):
        self._k=kernel
        self._points=pointsApproximation
        self._gradXWSigmaOfunc=gradXWSigmaOfunc
        self._gradXBfunc=gradXBfunc
        self._gradWBfunc=gradWBfunc
        self._PointsHist=PointsHist
        self._yHist=yHist
        self._Bhist=Bhist
        self._B=B
        self._dimKernel=dimKernel
        self._noiseHist=noiseHist
        self._gradXBfunc=gradXBfunc
        self._BhistSaved=BhistSaved
        self._numberTraining=numberTraining
        
    def evalVOI(self,n,pointNew,a,b,grad=False,**args):
        raise NotImplementedError, "this needs to be implemented"
        

class VOISBO(VOI):
    def __init__(self,dimW,gradXBforAn,*args,**kargs):
        VOI.__init__(self,*args,**kargs)
        self.VOI_name="SBO"
        self._dimW=dimW
        self._GP=stat.SBOGP(kernel=self._k,B=self._B,dimNoiseW=dimW,dimPoints=self._dimKernel-dimW,
                       numberPoints=self._points.shape[0],Bhist=self._Bhist,histSaved=self._BhistSaved,
                       Xhist=self._PointsHist, dimKernel=self._dimKernel,
                       yHist=self._yHist,noiseHist=self._noiseHist,numberTraining=self._numberTraining,
                       gradXBfunc=self._gradXBfunc,gradXBforAn=gradXBforAn)

    ##a,b are the vectors of the paper: a=(a_{n}(x_{i}), b=(sigma^tilde_{n})
    def evalVOI(self,n,pointNew,a,b,gamma,BN,L,grad=False,onlyGradient=False):
        #n>=0
        a,b,keep=AffineBreakPointsPrep(a,b)
        keep1,c=AffineBreakPoints(a,b)
        keep1=keep1.astype(np.int64)
        n1=self._dimKernel-self._dimW
        n2=self._dimW
        
        if grad==False:
            h=hvoi(b,c,keep1) ##Vn
            return h
        ####Gradient
        bPrev=b
        a=a[keep1]
        b=b[keep1]
        keep=keep[keep1] #indices conserved
        M=len(keep)
        if M<=1:
            return h,np.zeros(self._dimKernel)
        B=self._GP.Bhist
        cPrev=c
        c=c[keep1+1]
        c2=np.abs(c[0:M-1])
        evalC=norm.pdf(c2)
        
        nTraining=self._GP._numberTraining
        tempN=nTraining+n
        gradXSigma0,gradWSigma0=self._gradXWSigmaOfunc(n,pointNew,self,self._PointsHist[0:tempN,0:n1],self._PointsHist[0:tempN,n1:n1+n2])
     #   gradWSigma0=temp[:,n1:n1+n2]
        gradXB=self._gradXBfunc(pointNew,self,BN,keep) ##check n
        gradWB=self._gradWBfunc(pointNew,self,BN,keep)
    #    gradXB=np.zeros([len(keep1),n1])###ver esto
     #   gradWB=np.zeros([len(keep1),n2])####falta L
     
        
        
        A=self._k.A(self._PointsHist[0:tempN,:],noise=self._noiseHist[0:tempN])
        L=np.linalg.cholesky(A)
        gradientGamma=np.concatenate((gradXSigma0,gradWSigma0),1).transpose()
        inv3=linalg.solve_triangular(L,gamma,lower=True)
        beta1=(self._GP._k.A(pointNew)-np.dot(inv3.T,inv3))
        gradient=np.zeros(M)
        result=np.zeros(n1+n2)

        for i in xrange(n1):
            for j in xrange(M):
               # gradXB[j,i]=-2.0*alpha1[i]*BN[keep[j],0]*(xNew[0,i]-points[keep[j],i])
                inv1=linalg.solve_triangular(L,B[keep[j],:].transpose(),lower=True)
                inv2=linalg.solve_triangular(L,gradientGamma[i,0:tempN].transpose(),lower=True)
                tmp=np.dot(inv2.T,inv1)
                tmp=(beta1**(-.5))*(gradXB[j,i]-tmp)
                beta2=BN[keep[j],:]-np.dot(inv1.T,inv3)
                tmp2=(.5)*(beta1**(-1.5))*beta2*(2.0*np.dot(inv2.T,inv3))
                gradient[j]=tmp+tmp2
            result[i]=np.dot(np.diff(gradient),evalC)
            
        for i in xrange(n2):
            for j in xrange(M):
               # gradWB[j,i]=BN[keep[j],0]*(alpha2[i]*(mu[i]-wNew[0,i]))/((variance[i]*alpha2[i]+.5))
                inv1=linalg.solve_triangular(L,B[keep[j],:].transpose(),lower=True)
                inv2=linalg.solve_triangular(L,gradientGamma[i+n1,0:tempN].transpose(),lower=True)
                tmp=np.dot(inv2.T,inv1)
                tmp=(beta1**(-.5))*(gradWB[j,i]-tmp)
                beta2=BN[keep[j],:]-np.dot(inv1.T,inv3)
                tmp2=(.5)*(beta1**(-1.5))*(2.0*np.dot(inv2.T,inv3))*beta2
                gradient[j]=tmp+tmp2
            result[i+n1]=np.dot(np.diff(gradient),evalC)
            
        if onlyGradient:
            return result
        h=hvoi(bPrev,cPrev,keep1) ##Vn
        return h,result
                    
                    
    def VOIfunc(self,n,pointNew,grad,onlyGradient=False):
        n1=self._dimKernel-self._dimW
        a,b,gamma,BN,L=self._GP.aANDb(n,self._points,pointNew[0,0:n1],pointNew[0,n1:self._dimKernel])
        if onlyGradient:
            return self.evalVOI(n,pointNew,a,b,gamma,BN,L,grad=True,onlyGradient=onlyGradient)
        if grad==False:
            return self.evalVOI(n,pointNew,a,b,gamma,BN,L)
        return self.evalVOI(n,pointNew,a,b,gamma,BN,L,grad=True)

class EI(VOI):
    def __init__(self,gradXKern,*args,**kargs):
        VOI.__init__(self,*args,**kargs)
        self.VOI_name="EI"
        self._GP=stat.EIGP(kernel=self._k,dimPoints=self._dimKernel,
                       Xhist=self._PointsHist, dimKernel=self._dimKernel,
                       yHist=self._yHist,noiseHist=self._noiseHist,numberTraining=self._numberTraining,
                       gradXKern=gradXKern)
      
      
      
    def VOIfunc(self,n,pointNew,grad):
        xNew=pointNew
        nTraining=self._GP._numberTraining
        tempN=nTraining+n
      #  n=n-1
        vec=np.zeros(tempN)
        X=self._PointsHist
        for i in xrange(tempN):
            vec[i]=self._GP.muN(X[i,:],n)
        maxObs=np.max(vec)
        std=np.sqrt(self._GP.varN(xNew,n))
        muNew,gradMu=self._GP.muN(xNew,n,grad=True)
        Z=(muNew-maxObs)/std
        temp1=(muNew-maxObs)*norm.cdf(Z)+std*norm.pdf(Z)
        if grad==False:
            return temp1
        var,gradVar=self._GP.varN(xNew,n,grad=True)
        gradstd=.5*gradVar/std
        gradZ=((std*gradMu)-(muNew-maxObs)*gradstd)/var
        temp10=gradMu*norm.cdf(Z)+(muNew-maxObs)*norm.pdf(Z)*gradZ+norm.pdf(Z)*gradstd+std*(norm.pdf(Z)*Z*(-1.0))*gradZ
        return temp1,temp10
 
class KG(VOI):
    def __init__(self,gradXKern,gradXKern2,*args,**kargs):
        VOI.__init__(self,*args,**kargs)
        self.VOI_name="KG"
        self.gradXKern=gradXKern
        self.gradXKern2=gradXKern2
        self._GP=stat.KG(kernel=self._k,dimPoints=self._dimKernel,
                       Xhist=self._PointsHist, dimKernel=self._dimKernel,
                       yHist=self._yHist,noiseHist=self._noiseHist,numberTraining=self._numberTraining,
                       gradXKern=gradXKern,gradXKern2=gradXKern2)
      
    def evalVOI(self,n,pointNew,a,b,L,grad=False):
       # print "a,b,"
       # print a,b
        a,b,keep=AffineBreakPointsPrep(a,b)
        keep1,c=AffineBreakPoints(a,b)
        keep1=keep1.astype(np.int64)
        n1=self._dimKernel
        h=hvoi(b,c,keep1) ##Vn
        if grad==False:
            return h
        ####Gradient
        a=a[keep1]
        b=b[keep1]
        keep=keep[keep1] #indices conserved
        M=len(keep)
        if M<=1:
            return h,np.zeros(self._dimKernel)
        c=c[keep1+1]
        c2=np.abs(c[0:M-1])
        evalC=norm.pdf(c2)
        nTraining=self._GP._numberTraining
        tempN=nTraining+n
        B=np.zeros((1,tempN))
        X=self._PointsHist
        for i in xrange(tempN):
            B[0,i]=self._k.K(pointNew,X[i:i+1,:])[:,0]
        temp22=linalg.solve_triangular(L,B.T,lower=True)
       # gradX=np.zeros([M,tempN])
        gradX=self.gradXKern(pointNew,n,self._GP)
        temp=np.zeros([tempN,n1])
        tmp100=norm.pdf(c2)
        gradient=np.zeros(n1)
        B2=np.zeros((1,tempN))
        temp54=np.zeros(M)
        sigmaXnew=sqrt(self._k.K(np.array(pointNew).reshape((1,n1)))-np.dot(temp22.T,temp22))
        inv3=linalg.solve_triangular(L,B[0,:],lower=True)
        beta1=(self._GP._k.A(pointNew)-np.dot(inv3.T,inv3))
        for j in xrange(n1):
            for i in xrange(M):
                for p1 in xrange(tempN):
                   # gradX[i,p1]=self._k.K(xNew,X[p1,:].reshape((1,n1)))*(2.0*alpha1[j]*(xNew[0,j]-X[p1,j]))
                    B2[0,p1]=self._k.K(self._points[keep[i]:keep[i]+1,:],X[p1:p1+1,:])[:,0]
                inv1=linalg.solve_triangular(L,B2.T,lower=True)
               # temp2=linalg.solve_triangular(L,gradX[i:i+1,:].T,lower=True)
                inv2=linalg.solve_triangular(L,gradX[:,j],lower=True)
               # temp53=self._k.K(xNew,self._points[keep[i]:keep[i]+1,:])*(2.0*self._alpha1[j]*(xNew[0,j]-self._points[keep[i],j]))
		tmp=np.dot(inv2.T,inv1)
                temp53=self.gradXKern2(pointNew,i,keep,j,self)
		tmp=(beta1**(-.5))*(temp53-tmp)
                beta2=self._k.K(self._points[keep[i]:keep[i]+1,:],pointNew)-np.dot(inv1.T,inv3)
                tmp2=(.5)*(beta1**(-1.5))*beta2*(2.0*np.dot(inv2.T,inv3))
                temp54[i]=tmp+tmp2
            gradient[j]=np.dot(np.diff(temp54),tmp100)
        return h,gradient
            
    def VOIfunc(self,n,pointNew,grad):
        n1=self._dimKernel
        a,b,L=self._GP.aANDb(n,self._points,pointNew)
        if grad==False:
            return self.evalVOI(n,pointNew,a,b,L)
        return self.evalVOI(n,pointNew,a,b,L,grad)

class PI(VOI):
    def __init__(self,gradXKern,*args,**kargs):
        VOI.__init__(self,*args,**kargs)
        self.VOI_name="PI"
        self._GP=stat.PIGP(kernel=self._k,dimPoints=self._dimKernel,
                       Xhist=self._PointsHist, dimKernel=self._dimKernel,
                       yHist=self._yHist,noiseHist=self._noiseHist,numberTraining=self._numberTraining,
                       gradXKern=gradXKern)
      
        
    def VOIfunc(self,n,pointNew,grad):
        xNew=pointNew
        nTraining=self._GP._numberTraining
        tempN=n+nTraining
        X=self._PointsHist[0:tempN,:]
        vec=np.zeros(tempN)
        for i in xrange(tempN):
            vec[i]=self._GP.muN(X[i,:],n)
        maxObs=np.max(vec)
        std=np.sqrt(self._GP.varN(xNew,n))
        muNew,gradMu=self._GP.muN(xNew,n,grad=True)
        Z=(muNew-maxObs)/std
        temp1=norm.cdf(Z)
        if grad==False:
            return temp1
        var,gradVar=self._GP.varN(xNew,n,grad=True)
        gradstd=.5*gradVar/std
        gradZ=((std*gradMu)-(muNew-maxObs)*gradstd)/var
        temp10=norm.pdf(Z)*gradZ
        return temp1,temp10
     
      
##evaluate the function h of the paper
##b has been modified in affineBreakPointsPrep
def hvoi (b,c,keep):
    M=len(keep)
    if M>1:
        c=c[keep+1]
        c2=-np.abs(c[0:M-1])
        tmp=norm.pdf(c2)+c2*norm.cdf(c2) 
        return np.sum(np.diff(b[keep])*tmp)
    else:
        return 0
