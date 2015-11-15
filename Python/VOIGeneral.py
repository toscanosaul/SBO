#!/usr/bin/env python

"""
This file defines several Value of Information functions (VOI).
It includes the VOI used for SBO; the Knowledge Gradient;
Expected Improvement and Probability Improvement.
"""

import numpy as np
from math import *
from AffineBreakPoints import *
import statGeneral as stat
from scipy import linalg
from numpy import linalg as LA
from scipy.stats import norm

class VOI:
    def __init__(self,numberTraining):
        """
        This class defines the Value of Information Function.
        
        Arguments:
            -numberTraining: Numer of training data.
        """
        self._numberTraining=numberTraining
        
    def evalVOI(self,n,pointNew,onlyGradient=False, grad=False,**args):
        """
        Output:
            Evaluates the VOI and it can compute its derivative. It evaluates
            the VOI, when grad and onlyGradient are False; it evaluates the
            VOI and computes its derivative when grad is True and onlyGradient
            is False, and computes only its gradient when gradient and
            onlyGradient are both True.
            
        Args:
            -n: Iteration of the algorithm
            -pointNew: The VOI will be evaluated at this point.
            -grad: True if we want to compute the gradient; False otherwise.
            -onlyGradient: True if we only want to compute the gradient; False otherwise.
        """
        raise NotImplementedError, "this needs to be implemented"
        

class VOISBO(VOI):
    def __init__(self,dimW,dimX,gradXBfunc,gradWBfunc,
                 pointsApproximation,gradXWSigmaOfunc,
                 *args,**kargs):
        """
        Value of Information used for SBO.
        
        Arguments:
            -dimW: Dimension of the vectorial space of w.
            -dimX: Dimension of the vectorial space of x.
            -pointsApproximation: Points used to approximate the VOI.
            -gradXBfunc: Computes the gradients with respect to x_{n+1} of
                         B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
                         where x_{p} is a point in the discretization
                         of the domain of x. Its arguments are:
                            -new: Point (x_{n+1},w_{n+1})
                            -kern: Kernel
                            -keep: Indexes of the points keeped of the
                                   discretization of the domain of x,
                                   after using AffineBreakPoints
                             -BN: Vector B(x_{p},n+1), where x_{p} is
                                  a point in the discretization of
                                  the domain of x.
                             -points: Discretization of the domain of x
            -gradWBfunc: Computes the gradients with respect to w_{n+1} of
                         B(x_{p},n+1)=\int\Sigma_{0}(x_{p},w,x_{n+1},w_{n+1})dp(w),
                         where x_{p} is a point in the discretization of
                         the domain of x. Its arguments are:
                            -new: Point (x_{n+1},w_{n+1})
                            -kern: Kernel
                            -keep: Indexes of the points keeped of the
                                   discretization of the domain of x,
                                   after using AffineBreakPoints
                            -BN: Vector B(x_{p},n+1), where x_{p} is a point
                                 in the discretization of the domain of x.
                            -points: Discretization of the domain of x
            -gradXWSigmaOfunc: Computes the gradient of Sigma_{0}, which is
                               the covariance of the GP on F.
                               Its arguments are:
                                -n: Number of iteration
                                -new: Point where Sigma_{0} is evaluated
                                -kern: Kernel
                                -Xtrain2: Past observations of X
                                -Wtrain2: Past observations of W
                                -N: Number of observations
        """
        VOI.__init__(self,*args,**kargs)
        self.VOI_name="SBO"
        self._dimW=dimW
        self.n2=dimW
        self.n1=dimX
        self._gradXBfunc=gradXBfunc
        self._gradWBfunc=gradWBfunc
        self._gradXWSigmaOfunc=gradXWSigmaOfunc
        self._points=pointsApproximation
        self.sizeDiscretization=self._points.shape[0]
        
    def aANDb(self,n,x,xNew,wNew,L,temp2,past,kernel,B):
        """
        Output:
            -b:Vector of posterior variances of G(x)=E[f(x,w,z)] if
               we choose (xNew,wNew) at this iteration. The variances
               are evaluated at all the points of x.
            -gamma: Vector of Sigma_{0}(x_{i},w_{i},xNew,wNew) where
                    (x_{i},w_{i}) are the past observations.
            -BN: Vector B(x_{p},n+1), where x_{p} is a point
                 in the discretization of the domain of x.
            -temp1: Solution to the system Ly=gamma, where L
                    is the Cholesky decomposition of A.
            -aux4: Square of the norm of temp1.
        
        Args:
            -n: Iteration of the algorithm
            -x: nxdim(x) matrix where b is evaluated.
            -(xNew,wNew): The VOI will be evaluated at this point.
            -L: Cholesky decomposition of the matrix A, where A is the covariance
                matrix of the past obsevations (x,w).
            -temp2:temp2=inv(L)*B.T, where B is a matrix such that B(i,j) is
                   \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
                   where points x_{p} is a point of the discretization of
                   the space of x; and (x_{j},w_{j}) is a past observation.
            -past: Past observations.
            -kernel: kernel.
            -B: Computes B(x,XW)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
                Its arguments are:
                    -x: Vector of points where B is evaluated
                    -XW: Point (x,w)
                    -n1: Dimension of x
                    -n2: Dimension of w
        """
        x=np.array(x)
        m=x.shape[0]
        tempN=self._numberTraining+n
        BN=np.zeros([m,1])
        n2=self.n2
        BN[:,0]=B(x,np.concatenate((xNew,wNew),1),self.n1,n2) #B(x,n+1)
 
        n1=self.n1
        n2=self.n2
        new=np.concatenate((xNew,wNew),1).reshape((1,n1+n2))

        gamma=np.transpose(kernel.A(new,past))
        temp1=linalg.solve_triangular(L,gamma,lower=True)
        b=(BN-np.dot(temp2.T,temp1))
        aux4=np.dot(temp1.T,temp1)
        b2=kernel.K(new)-aux4
        b2=np.clip(b2,0,np.inf)
        try:
            b=b/(np.sqrt(b2))
        except Exception as e:
            print "use a different point x"
            b=np.zeros((len(b),1))
        return b,gamma,BN,temp1,aux4

    def evalVOI(self,n,pointNew,a,b,c,keep,keep1,M,gamma,BN,L,inv,aux4,kern,XW,
                scratch=None,grad=False,onlyGradient=False):
        """
        Output:
            Evaluates the VOI and it can compute its derivative. It evaluates
            the VOI, when grad and onlyGradient are False; it evaluates the
            VOI and computes its derivative when grad is True and onlyGradient
            is False, and computes only its gradient when gradient and
            onlyGradient are both True.
        
        Args:
            -n: Iteration of the algorithm.
            -pointNew: The VOI will be evaluated at this point.
            -a: Vector of the means of the GP on g(x)=E(f(x,w,z)).
                The means are evaluated on the discretization of
                the space of x.
            -b: Vector of posterior variances of G(x)=E[f(x,w,z)] if
                we choose (xNew,wNew) at this iteration. The variances
                are evaluated at all the points of x.
            -c: Vector returned by AffineBreakPoints.
            -keep: Indexes returned by AffineBreakPointsPrep. They represent
                   the new order of the elements of a and b.
            -keep1: Indexes returned by AffineBreakPoints. Those are the
                    indexes of the elements keeped.
            -M: Number of points keeped.
            -gamma: Vector of Sigma_{0}(x_{i},w_{i},xNew,wNew) where
                    (x_{i},w_{i}) are the past observations.
            -BN: Vector B(x_{p},n+1), where x_{p} is a point
                 in the discretization of the domain of x. 
            -L: Cholesky decomposition of the matrix A, where A is the covariance
                matrix of the past obsevations (x,w).
            -inv: Solution to the system Ly=gamma, where L
                  is the Cholesky decomposition of A.
            -aux4: Square of the norm of inv.
            -kern: Kernel.
            -XW: Past observations.
            -scratch: Matrix where scratch[i,:] is the solution of the linear system
                      Ly=B[j,:].transpose() (See above for the definition of B and L)
            -grad: True if we want to compute the gradient; False otherwise.
            -onlyGradient: True if we only want to compute the gradient; False otherwise.
        """
        n1=self.n1
        n2=self.n2
        
        if grad==False:
            h=hvoi(b,c,keep1) ##Vn
            return h
        bPrev=b
        a=a[keep1]
        b=b[keep1]
        keep=keep[keep1] #indices conserved
        
        if M<=1:
            return h,np.zeros(n1+n2)

        cPrev=c
        c=c[keep1+1]
        c2=np.abs(c[0:M-1])
        evalC=norm.pdf(c2)
        
        nTraining=self._numberTraining
        tempN=nTraining+n
        gradXSigma0,gradWSigma0=self._gradXWSigmaOfunc(n,pointNew,
                                                       kern,XW[0:tempN,0:n1],
                                                       XW[0:tempN,n1:n1+n2])

        gradXB=self._gradXBfunc(pointNew,kern,BN,keep,self._points)
        gradWB=self._gradWBfunc(pointNew,kern,BN,keep,self._points)

        gradientGamma=np.concatenate((gradXSigma0,gradWSigma0),1).transpose()

        inv3=inv
        beta1=(kern.A(pointNew)-aux4)
        gradient=np.zeros(M)
        result=np.zeros(n1+n2)
        
        for i in xrange(n1):
            inv2=linalg.solve_triangular(L,gradientGamma[i,0:tempN].transpose(),lower=True)
            aux5=np.dot(inv2.T,inv3)
            for j in xrange(M):
                tmp=np.dot(inv2.T,scratch[j,:])
                tmp=(beta1**(-.5))*(gradXB[j,i]-tmp)
                beta2=BN[keep[j],:]-np.dot(scratch[j,:].T,inv3)
                tmp2=(.5)*(beta1**(-1.5))*beta2*(2.0*aux5)
                gradient[j]=tmp+tmp2
            result[i]=np.dot(np.diff(gradient),evalC)
            
        for i in xrange(n2):
            inv2=linalg.solve_triangular(L,gradientGamma[i+n1,0:tempN].transpose(),lower=True)
            aux5=np.dot(inv2.T,inv3)
            for j in xrange(M):
                tmp=np.dot(inv2.T,scratch[j,:])
                tmp=(beta1**(-.5))*(gradWB[j,i]-tmp)
                beta2=BN[keep[j],:]-np.dot(scratch[j,:].T,inv3)
                tmp2=(.5)*(beta1**(-1.5))*(2.0*aux5)*beta2
                gradient[j]=tmp+tmp2
            result[i+n1]=np.dot(np.diff(gradient),evalC)
            
        if onlyGradient:
            return result
        h=hvoi(bPrev,cPrev,keep1) 
        return h,result

    def VOIfunc(self,n,pointNew,grad,L,temp2,a,scratch,kern,XW,B,onlyGradient=False):
        """
        Output:
            Evaluates the VOI and it can compute its derivative. It evaluates
            the VOI, when grad and onlyGradient are False; it evaluates the
            VOI and computes its derivative when grad is True and onlyGradient
            is False, and computes only its gradient when gradient and
            onlyGradient are both True.
        
        Args:
            -n: Iteration of the algorithm.
            -pointNew: The VOI will be evaluated at this point.
            -grad: True if we want to compute the gradient; False otherwise.
            -L: Cholesky decomposition of the matrix A, where A is the covariance
                matrix of the past obsevations (x,w).
            -temp2: temp2=inv(L)*B.T, where B is a matrix such that B(i,j) is
                   \int\Sigma_{0}(x_{i},w,x_{j},w_{j})dp(w)
                   where points x_{p} is a point of the discretization of
                   the space of x; and (x_{j},w_{j}) is a past observation.
            -a: Vector of the means of the GP on g(x)=E(f(x,w,z)).
                The means are evaluated on the discretization of
                the space of x.
            -scratch: Matrix where scratch[i,:] is the solution of the linear system
                      Ly=B[j,:].transpose() (See above for the definition of B and L)
            -kern: Kernel.
            -XW: Past observations.
            -B: Computes B(x,XW)=\int\Sigma_{0}(x,w,XW[0:n1],XW[n1:n1+n2])dp(w).
                Its arguments are:
                    -x: Vector of points where B is evaluated
                    -XW: Point (x,w)
                    -n1: Dimension of x
                    -n2: Dimension of w
            -onlyGradient: True if we only want to compute the gradient; False otherwise.
        """
        n1=self.n1
        b,gamma,BN,temp1,aux4=self.aANDb(n,self._points,pointNew[0,0:n1],pointNew[0,n1:n1+self.n2],L,
                                    temp2=temp2,past=XW,kernel=kern,B=B)
        a,b,keep=AffineBreakPointsPrep(a,b)
        keep1,c=AffineBreakPoints(a,b)
        keep1=keep1.astype(np.int64)
        M=len(keep1)
        nTraining=self._numberTraining
        tempN=nTraining+n
        keep2=keep[keep1]
        if grad:
            scratch1=np.zeros((M,tempN))
            for j in xrange(M):
                scratch1[j,:]=scratch[keep2[j],:]
        if onlyGradient:
            return self.evalVOI(n,pointNew,a,b,c,keep,keep1,M,gamma,BN,L,scratch=scratch1,
                                inv=temp1,aux4=aux4,grad=True,onlyGradient=onlyGradient,
                                kern=kern,XW=XW)
        if grad==False:
            return self.evalVOI(n,pointNew,a,b,c,keep,keep1,M,gamma,BN,L,aux4=aux4,inv=temp1,
                                kern=kern,XW=XW)
      
        return self.evalVOI(n,pointNew,a,b,c,keep,keep1,M,gamma,BN,L,aux4=aux4,
                            inv=temp1,scratch=scratch1,grad=True,
                            kern=kern,XW=XW)

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
