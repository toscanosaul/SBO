#!/usr/bin/env python

#####We compute the value of information for the CitiBike case (poisson distribution) and also for the normal distribution case

import numpy as np
from scipy.stats import poisson
from scipy import linalg
from numpy import linalg as LA
from scipy.stats import norm
from AffineBreakPoints import *
from math import *

logFactorial=[np.sum([log(i) for i in range(1,j+1)]) for j in range(1,701)]
logFactorial.insert(1,0)
logFactorial=np.array(logFactorial)

#Computes log*sum(exp(x)) for a vector x, but in numerically careful way
def logSumExp(x):
    xmax=np.max(x)
    y=xmax+np.log(np.sum(x-xmax))
    return y


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

#compute B(x,i). X=X[i,:],W[i,:]. x is a matrix of dimensions nxm where m is the dimension of an element of x.
##This function is used by update. Remember that B(x,i)=integral_(sigma(x,w,x_i,w_i))dp(w)
##n1 is the dimension of X[i,:]
##n2 is the dimension of W[i,:]
##variance0 is the parameter of kernel (the variance)
##alpha1 parameter of the kernel. It is related to x
##alpha2 parameter of the kernel. It is related to w
##poisson is true if we want to use it for the citibike problem
##lambdaParameter are the parameters of the Poisson processes for W. It can be empty if poisson=False
def Bparameters(x,X,W,n1,n2,variance0,alpha1,alpha2,lambdaParameter,poisson2=True):
    x=np.array(x).reshape((x.shape[0],n1))
    results=np.zeros(x.shape[0])
    if poisson2==True:
        for i in xrange(x.shape[0]):
            temp=variance0*np.exp(np.sum(-alpha1*((x[i,:]-X)**2)))
            temp2=1.0
            for j in xrange(n2):
                temp2=temp2*poisson.expect(lambda z: np.exp(-alpha2[j]*(z-W[j])**2), args=(np.sum(lambdaParameter[j]),))
            results[i]=temp2*temp
    return results

##X,W are the past observations
##An is the matrix of the paper
##k is the kernel
def An(n,X,W,k,varianceObservations):
    An=k.K(np.concatenate((X,W),1))+np.diag(varianceObservations[0:n])
    return An

##return a_n and b_n, in the paper. This function is used for Vn.
##x is a nxdim(x) matrix of points where a_n and sigma_n are evaluated
##xNew, wNew are the point in sigma(x,xNew,wNew)
##muStart is the mean of the kernel
##k is the kernel
##variance0 is the parameter of kernel (the variance)
####alpha1 parameter of the kernel. It is related to x
##alpha2 parameter of the kernel. It is related to w
##n1 is the dimension of X[i,:]
##n2 is the dimension of W[i,:]
##lambdaParameter are the parameters of the Poisson processes for W. It can be empty if poisson=False
def update (x,n,y,X,W,xNew,wNew,muStart,kernel,variance0,alpha1,alpha2,n1,n2,lambdaParameter,varianceObservations,Bobs):
    if n==0:
        return muStart
    x=np.array(x)
    m=x.shape[0]
    A=An(n,X,W,kernel,varianceObservations)
    L=np.linalg.cholesky(A)
    B=np.zeros([m,n])
    B=Bobs
  #  for i in xrange(n):
   #     B[:,i]=Bparameters(x,X[i,:],W[i,:],n1,n2,variance0,alpha1,alpha2,lambdaParameter)
    BN=np.zeros([m,1])
    BN[:,0]=Bparameters(x,xNew[0,:],wNew[0,:],n1,n2,variance0,alpha1,alpha2,lambdaParameter) #B(x,n+1)
    temp2=linalg.solve_triangular(L,B.T,lower=True)
    temp1=linalg.solve_triangular(L,np.array(y)-muStart,lower=True) 
    a=muStart+np.dot(temp2.T,temp1)
    past=np.concatenate((X,W),1)
    new=np.concatenate((xNew,wNew),1).reshape((1,n1+n2))
    gamma=np.transpose(kernel.K(new,past))
    temp1=linalg.solve_triangular(L,gamma,lower=True)
    b=(BN-np.dot(temp2.T,temp1))
    b2=kernel.K(new)-np.dot(temp1.T,temp1)
    if( b2>0):
        b=b/(sqrt(b2))
    else:
        ###this means we already know the point and so has variance zero.
        print "error!!!!!!!"
        b=np.zeros((len(b),1))
    return a,b,gamma,BN,L,B

#compute Vn and its gradient at xNew,wNew, evaluated in the points
###X,W are the past observations. They are matrices
###n is the index of V_{n-1}. So, V_{0} is computed with n=1
###y are the past observations
#if grad=True, the gradient is computed
##muStart is the mean of the kernel
##k is the kernel
##variance0 is the parameter of kernel (the variance)
##alpha1 parameter of the kernel. It is related to x
##Precondition: n>=1
##n1 is the dimension of X[i,:]
##n2 is the dimension of W[i,:]
##alpha2 parameter of the kernel. It is related to w
##lambdaParameter are the parameters of the Poisson processes for W. It can be empty if poisson=False
##points is the discretization of the space
def Vn (xNew,wNew,X,W,n,y,muStart,kernel,variance0,alpha1,alpha2,n1,n2,lambdaParameter,points,varianceObservations,Bobs,grad=True):
    if n>1:
        n=n-1
        a,b,gamma,BN,L,B=update(points,n,y,X,W,np.array(xNew).reshape((1,n1)),np.array(wNew).reshape((1,n2)),muStart,kernel,variance0,alpha1,alpha2,n1,n2,lambdaParameter,varianceObservations,Bobs)
        if np.all(b==np.zeros((len(b),1))):
            if grad==False:
                return 0
            else:
                return 0,0
        a,b,keep=AffineBreakPointsPrep(a,b)
        keep1,c=AffineBreakPoints(a,b)
        keep1=keep1.astype(np.int64)
        h=hvoi(b,c,keep1) ##Vn
        ###Now we compute the gradient
        a=a[keep1]
        b=b[keep1]
        keep=keep[keep1] #indices conserved
        M=len(keep)
        if grad==True:
            #########DELETE#######

            #####################
            if M<=1:
                return 0,0
            else:
                c=c[keep1+1]
                c2=np.abs(c[0:M-1])
                evalC=norm.pdf(c2)
                gradXSigma0=np.zeros([n+1,n1])
                gradWSigma0=np.zeros([n+1,n2])
                gradXB=np.zeros([len(keep1),n1])
                gradWB=np.zeros([len(keep1),n2])
                for i in xrange(n):
                    gradXSigma0[i,:]=-2.0*gamma[i]*alpha1*(xNew-X[i,:])
                    gradWSigma0[i,:]=-2.0*gamma[i]*alpha2*(wNew-W[i,:])
                gradientGamma=np.concatenate((gradXSigma0,gradWSigma0),1).transpose()
                inv3=linalg.solve_triangular(L,gamma,lower=True)
                beta1=(kernel.K(np.concatenate((xNew,wNew),1))-np.dot(inv3.T,inv3))
                gradient=np.zeros(M)
                result=np.zeros(n1+n2)
                for i in xrange(n1):
                    for j in xrange(M):
                        gradXB[j,i]=-2.0*alpha1[i]*BN[keep[j],0]*(xNew[0,i]-points[keep[j],i])
                        inv1=linalg.solve_triangular(L,B[keep[j],:].transpose(),lower=True)
                        inv2=linalg.solve_triangular(L,gradientGamma[i,0:n].transpose(),lower=True)
                        tmp=np.dot(inv2.T,inv1)
                        tmp=(beta1**(-.5))*(gradXB[j,i]-tmp)
                        beta2=BN[keep[j],:]-np.dot(inv1.T,inv3)
                        tmp2=(.5)*(beta1**(-1.5))*(2.0*np.dot(inv2.T,inv3))*beta2
                        gradient[j]=tmp+tmp2
                    result[i]=-np.dot(np.diff(gradient),evalC)
                    
                for i in xrange(n2):
                    for j in xrange(M):
                        temp100=BN[keep[j],0]*poisson.expect(lambda x: (x-wNew[0,i])*np.exp(-alpha2[i]*(x-wNew[0,i])**2), args=(np.sum(lambdaParameter[i]),))
                        temp100=temp100*2.0*alpha2[i]
                        temp101=poisson.expect(lambda x: np.exp(-alpha2[i]*(x-wNew[0,i])**2), args=(np.sum(lambdaParameter[i]),))
                        temp100=temp100/temp101
                        gradWB[j,i]=temp100
                        inv1=linalg.solve_triangular(L,B[keep[j],:].transpose(),lower=True)
                        inv2=linalg.solve_triangular(L,gradientGamma[i+n1,0:n].transpose(),lower=True)
                        tmp=np.dot(inv2.T,inv1)
                        tmp=(beta1**(-.5))*(gradWB[j,i]-tmp)
                        beta2=BN[keep[j],:]-np.dot(inv1.T,inv3)
                        tmp2=(.5)*(beta1**(-1.5))*(2.0*np.dot(inv2.T,inv3))*beta2
                        gradient[j]=tmp+tmp2
                    result[i+n1]=-np.dot(np.diff(gradient),evalC)
                return h,result
        else:
            return h
    else:
        m=points.shape[0]
        BN=np.zeros([m,1])
        BN[:,0]=Bparameters(points,xNew.reshape(n1),wNew.reshape(n2),n1,n2,variance0,alpha1,alpha2,lambdaParameter) #B(x,n+1)
        f=open("Bver.txt","w")
        np.savetxt(f,BN)
        f.close()
        b=BN[:,0]
        a=np.repeat(muStart,m)
        a,b,keep=AffineBreakPointsPrep(a,b)
        keep1,c=AffineBreakPoints(a,b)
        keep1=keep1.astype(np.int64)
        M=len(keep1)
        h=hvoi(b,c,keep1)
        keep=keep[keep1]
        if M<=1:
            if grad==True:
                return 0,0
            else:
                return 0
        else:
            if grad==True:
                c=c[keep1+1]
                c2=np.abs(c[0:M-1])
                evalC=norm.pdf(c2)
                gradXB=np.zeros([len(keep1),n1])
                gradWB=np.zeros([len(keep1),n2])
                Gradient=np.zeros(n1+n2)
                for i in xrange(n1):
                    for j in xrange(M):
                        gradXB[j,i]=-2.0*alpha2[i]*BN[keep[j],0]*(xNew[0,i]-points[keep[j],i])
                    Gradient[i]=-np.dot(np.diff(gradXB[:,i]).T,evalC)

                for i in xrange(n2):
                    for j in xrange(M):
                        temp100=BN[keep[j],0]*poisson.expect(lambda x: (x-wNew[0,i])*np.exp(-alpha2[i]*(x-wNew[0,i])**2), args=(np.sum(lambdaParameter[i]),))
                        temp100=temp100*2.0*alpha2[i]
                        temp101=poisson.expect(lambda x: np.exp(-alpha2[i]*(x-wNew[0,i])**2), args=(np.sum(lambdaParameter[i]),))
                        temp100=temp100/temp101
                        gradWB[j,i]=temp100/sqrt(kernel.K(np.concatenate((xNew,wNew),1)))
                    Gradient[i+n1]=-np.dot(np.diff(gradWB[:,i]).T,evalC)
                return h, Gradient
            else:
                return h


##x is point where function is evaluated
##L is cholesky factorization of An
##X,W past points
##y2 are the past observations
##n is the time where aN is computed
##Output: aN and its gradient if gradient=True
##Other parameters are defined in Vn
def aN_grad(x,L,y2,X,W,n1,n2,variance0,alpha1,alpha2,lambdaParameter,muStart,n,Bobs,gradient=True):
    B=np.zeros(n)
    for i in xrange(n):
        B[i]=Bparameters(np.array(x).reshape((1,n1)),X[i,:],W[i,:],n1,n2,variance0,alpha1,alpha2,lambdaParameter)
    inv1=linalg.solve_triangular(L,y2,lower=True)
    inv2=linalg.solve_triangular(L,B.transpose(),lower=True)
    aN=muStart+np.dot(inv2.transpose(),inv1)
    if gradient==True:
        gradXB=np.zeros((n1,n))
        for i in xrange(n):
            gradXB[:,i]=B[i]*(-2.0*alpha1*(x-X[i,:]))
        temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
        temp5=linalg.solve_triangular(L,y2,lower=True)
        gradAn=np.dot(temp5.transpose(),temp4)
        return aN,gradAn
    else:
        return aN