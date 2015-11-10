#!/usr/bin/env python

import numpy as np
from math import *
from AffineBreakPoints import *
import statGeneral
from scipy import linalg
from numpy import linalg as LA
from scipy.stats import norm

class VOI:
    #####x is a nxdim(x) matrix of points where a_n and sigma_n are evaluated
    ###B(x,x_1),...B(x,x_n), B is the function to compute those
    def __init__(self,kernel,dimKernel,numberTraining,gradWSigmafunc=None,Bhist=None,pointsApproximation=None,
                 gradXSigmaOfunc=None,gradXBfunc=None,B=None,PointsHist=None,yHist=None,noiseHist=None):
        self._k=kernel
        self._points=pointsApproximation
        self._gradXSigmaOfunc=gradXSigmaOfunc
        self._gradXBfunc=gradXBfunc
        self._gradWBfunc=gradWSigmafunc
        self._PointsHist=PointsHist
        self._yHist=yHist
        self._Bhist=Bhist
        self._B=B
        self._dimKernel=dimKernel
        self._noiseHist=noiseHist
        
    def evalVOI(self,n,pointNew,a,b,grad=False,**args):
        raise NotImplementedError, "this needs to be implemented"
        

class VOISBO(VOI):
    def __init__(self,dimW,*args,**kargs):
        VOI.__init__(self,*args,**kargs)
        self.VOI_name="SBO"
        self._dimW=dimW
        self._GP=SBOGP(kernel=self._k,B=self._B,dimNoiseW=dimW,dimPoints=self._dimKernel-dimNoiseW,
                       numberPoints=self._points.shape[0],Bhist=self._Bhist,histSaved=Bhist.shape[1],
                       Xhist=self._PointsHist,
                       yHist=self._yhist,noiseHist=noiseHist,numberTraining=numberTraining)
        
    ##a,b are the vectors of the paper: a=(a_{n}(x_{i}), b=(sigma^tilde_{n})
    def evalVOI(self,n,pointNew,a,b,gamma,BN,L,grad=False):
        #n>=0
        a,b,keep=AffineBreakPointsPrep(a,b)
        keep1,c=AffineBreakPoints(a,b)
        keep1=keep1.astype(np.int64)
        n1=self._dimKernel-self._dimW
        n2=self._dimW
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
        gradXSigma0=self._gradXSigmaOfunc(pointNew[0:n1])
        gradWSigma0=self._gradWSigmaOfunc(pointNew[n1:n1+n2])
     #   gradWSigma0=temp[:,n1:n1+n2]
      
        gradXB=self._gradXBfunc(keep1,nTraining+n+1) ##check n
        gradWB=self._gradWBfunc(keep1,nTraining+n+1)
        
    #    gradXB=np.zeros([len(keep1),n1])###ver esto
     #   gradWB=np.zeros([len(keep1),n2])####falta L
     
        
        tempN=nTraining+n
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
                inv2=linalg.solve_triangular(L,gradientGamma[i,0:n].transpose(),lower=True)
                tmp=np.dot(inv2.T,inv1)
                tmp=(beta1**(-.5))*(gradXB[j,i]-tmp)
                beta2=BN[keep[j],:]-np.dot(inv1.T,inv3)
                tmp2=(.5)*(beta1**(-1.5))*beta2*(2.0*np.dot(inv2.T,inv3))
                gradient[j]=tmp+tmp2
            result[i]=-np.dot(np.diff(gradient),evalC)
            
        for i in xrange(n2):
            for j in xrange(M):
               # gradWB[j,i]=BN[keep[j],0]*(alpha2[i]*(mu[i]-wNew[0,i]))/((variance[i]*alpha2[i]+.5))
                inv1=linalg.solve_triangular(L,B[keep[j],:].transpose(),lower=True)
                inv2=linalg.solve_triangular(L,gradientGamma[i+n1,0:n].transpose(),lower=True)
                tmp=np.dot(inv2.T,inv1)
                tmp=(beta1**(-.5))*(gradWB[j,i]-tmp)
                beta2=BN[keep[j],:]-np.dot(inv1.T,inv3)
                tmp2=(.5)*(beta1**(-1.5))*(2.0*np.dot(inv2.T,inv3))*beta2
                gradient[j]=tmp+tmp2
            result[i+n1]=-np.dot(np.diff(gradient),evalC)
        
        
        return h,result
                    
                    
    def VOIfunc(self,n,pointNew,grad):
        n1=self._dimKernel-self._dimW
        a,b,gamma,BN,L=self._GP.aANDb(n,self._points,pointNew[0:n1],pointNew[n1:self._dimKernel])
        if grad==False:
            return self.evalVOI(n,pointNew,a,b,gamma,BN,L)
        return self.evalVOI(n,pointNew,a,b,gamma,BN,L,grad)

    
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