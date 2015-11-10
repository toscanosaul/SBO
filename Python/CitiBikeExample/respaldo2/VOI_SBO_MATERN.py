#!/usr/bin/env python

#####We compute the value of information for the CitiBike case (poisson distribution) and also for the normal distribution case

import numpy as np
from scipy.stats import poisson
from scipy import linalg
from numpy import linalg as LA
from scipy.stats import norm
from AffineBreakPoints import *
from math import *
import itertools
import multiprocessing

logFactorial=[np.sum([log(i) for i in range(1,j+1)]) for j in range(1,501)]
logFactorial.insert(1,0)

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

####NOTE!!!!! THIS ONLY WORKS FOR n2=4, for other values the function has to be changed. It's faster in this way
#compute B(x,i). X=X[i,:],W[i,:]. x is a matrix of dimensions nxm where m is the dimension of an element of x.
##This function is used by update. Remember that B(x,i)=integral_(sigma(x,w,x_i,w_i))dp(w)
##n1 is the dimension of X[i,:]
##n2 is the dimension of W[i,:]
##variance0 is the parameter of kernel (the variance)
##alpha1 parameter of the kernel. It is related to x
##alpha2 parameter of the kernel. It is related to w
##poisson is true if we want to use it for the citibike problem
##lambdaParameter are the parameters of the Poisson processes for W. It can be empty if poisson=False
def auxBparameters(x,X,W,n1,n2,variance0,alpha1,alpha2,lambdaParameter,nPartition,parameterLamb,tempVect,poisson2=True):
   # nPartition=5
    s1=[-np.sum(parameterLamb)+np.dot(z,np.log(parameterLamb))-
        np.sum([logFactorial[z[j]] for j in range(n2)])
        -sqrt(3)*np.sqrt(np.sum(alpha1*((x-X)**2))+np.sum(alpha2*((z-W)**2)))
        +np.log(1+sqrt(3)*np.sqrt(np.sum(alpha1*((x-X)**2))+np.sum(alpha2*((z-W)**2))))
        for z in tempVect]
   # s=np.sqrt([np.sum(alpha1*((x[i,:]-X)**2))+np.sum(alpha2*((z-W)**2)) for z in tempVect])
   # s2=[(1+sqrt(3)*r)*np.exp(-sqrt(3)*r) for r in s]
    results=variance0*np.sum(np.exp(s1))
    return results

def Bparameters(x,X,W,n1,n2,variance0,alpha1,alpha2,lambdaParameter,poisson2=True):
    x=np.array(x).reshape((x.shape[0],n1))
    results=np.zeros(x.shape[0])
    if poisson2==True:
      #  print x.shape[0]
        parameterLamb=np.zeros(n2)
        for j in xrange(n2):
            parameterLamb[j]=np.sum(lambdaParameter[j])
        nPartition=4
        tempVect=[(i1,i2,i3,i4) for i1 in range(max(0,int(W[0])-nPartition),int(W[0])+nPartition) for i2 in range(max(0,int(W[1])-nPartition),int(W[1])+nPartition)
            for i3 in range(max(0,int(W[2])-nPartition),int(W[2])+nPartition) for i4 in range(max(0,int(W[3])-nPartition),int(W[3])+nPartition)]
        pool=multiprocessing.Pool()
        results_async = [pool.apply_async(auxBparameters,args=(x[i,:],X,W,n1,n2,variance0,alpha1,alpha2,lambdaParameter,nPartition,parameterLamb,tempVect,)) for i in range(x.shape[0])]
        output = [p.get() for p in results_async]
        pool.close()
        pool.join()
   # print output
    return output

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
    print "matrixA done"
    L=np.linalg.cholesky(A)
    print "factorization done"
    B=np.zeros([m,n])

    B=Bobs
 #   for i in xrange(n):
  #      B[:,i]=Bparameters(x,X[i,:],W[i,:],n1,n2,variance0,alpha1,alpha2,lambdaParameter)

    BN=np.zeros([m,1])
    print "BN"
    BN[:,0]=Bparameters(x,xNew[0,:],wNew[0,:],n1,n2,variance0,alpha1,alpha2,lambdaParameter) #B(x,n+1)
    print "BN Done"
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
        print "uddate"
        a,b,gamma,BN,L,B=update(points,n,y,X,W,np.array(xNew).reshape((1,n1)),np.array(wNew).reshape((1,n2)),muStart,kernel,variance0,alpha1,alpha2,n1,n2,lambdaParameter,varianceObservations,Bobs)
        print "updateDone"
        B=Bobs
  
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
                    r=np.sum(alpha1*((xNew-X[i,:])**2))+np.sum(alpha2*((wNew-W[i,:])**2))
                    r=np.sqrt(r)
                    gradXSigma0[i,:]=-variance0*3.0*np.exp(-sqrt(3)*r)*alpha1*(xNew-X[i,:])
                    gradWSigma0[i,:]=-variance0*3.0*np.exp(-sqrt(3)*r)*alpha2*(wNew-W[i,:])
                gradientGamma=np.concatenate((gradXSigma0,gradWSigma0),1).transpose()
                inv3=linalg.solve_triangular(L,gamma,lower=True)
                beta1=(kernel.K(np.concatenate((xNew,wNew),1))-np.dot(inv3.T,inv3))
                gradient=np.zeros(M)
                result=np.zeros(n1+n2)
                ########How many terms we should add to approximate expectations
                parameterLamb=np.zeros(n2)
                for j2 in xrange(n2):
                    parameterLamb[j2]=np.sum(lambdaParameter[j2])
                quantil=int(poisson.ppf(.99999999,max(parameterLamb)))
                ######
                print "derivatives"
                nPartition=5
                tempVect=[(i1,i2,i3,i4) for i1 in range(max(0,int(wNew[0,0])-nPartition),int(wNew[0,0])+nPartition) for i2 in range(max(0,int(wNew[0,1])-nPartition),int(wNew[0,1])+nPartition)
                    for i3 in range(max(0,int(wNew[0,2])-nPartition),int(wNew[0,2])+nPartition) for i4 in range(max(0,int(wNew[0,3])-nPartition),int(wNew[0,3])+nPartition)]
                for i in xrange(n1):
                    for j in xrange(M):
                        ####integral of the derivative
                       # tempVect=itertools.product(range(quantil),repeat=n2)
                        s1=[-np.sum(parameterLamb)+np.dot(z,np.log(parameterLamb))-
                            np.sum([np.sum([log(i1) for i1 in range(1,z[j1]+1)]) for j1 in range(n2)])
                            -sqrt(3)*np.sqrt(np.sum(alpha1*((points[keep[j],:]-xNew[0,:])**2))+np.sum(alpha2*((z-wNew[0,:])**2)))
                            for z in tempVect]
                        s2=-3.0*variance0*alpha1[i]*(xNew[0,i]-points[keep[j],i])*np.sum(np.exp(s1))
                        ##############
                        gradXB[j,i]=s2
                        inv1=linalg.solve_triangular(L,B[keep[j],:].transpose(),lower=True)
                        inv2=linalg.solve_triangular(L,gradientGamma[i,0:n].transpose(),lower=True)
                        tmp=np.dot(inv2.T,inv1)
                        tmp=(beta1**(-.5))*(gradXB[j,i]-tmp)
                        beta2=BN[keep[j],:]-np.dot(inv1.T,inv3)
                        tmp2=(.5)*(beta1**(-1.5))*(2.0*np.dot(inv2.T,inv3))*beta2
                        gradient[j]=tmp+tmp2
                    result[i]=-np.dot(np.diff(gradient),evalC)
                
                print "derivativeX done"
                for i in xrange(n2):
                    for j in xrange(M):
                      #  tempVect=itertools.product(range(quantil),repeat=n2)
                        s1=[[np.exp(-np.sum(parameterLamb)+np.dot(z,np.log(parameterLamb))-
                            np.sum([np.sum([log(i1) for i1 in range(1,z[j2]+1)]) for j2 in range(n2)])
                            -sqrt(3)*np.sqrt(np.sum(alpha1*((points[keep[j],:]-xNew[0,:])**2))+np.sum(alpha2*((z-wNew[0,:])**2)))),z[i]]
                            for z in tempVect]
                        s2=[t[0]*alpha2[i]*(wNew[0,i]-t[1]) for t in s1]
                        temp=-3.0*variance0*np.sum(s2)
                        gradWB[j,i]=temp
                        inv1=linalg.solve_triangular(L,B[keep[j],:].transpose(),lower=True)
                        inv2=linalg.solve_triangular(L,gradientGamma[i+n1,0:n].transpose(),lower=True)
                        tmp=np.dot(inv2.T,inv1)
                        tmp=(beta1**(-.5))*(gradWB[j,i]-tmp)
                        beta2=BN[keep[j],:]-np.dot(inv1.T,inv3)
                        tmp2=(.5)*(beta1**(-1.5))*(2.0*np.dot(inv2.T,inv3))*beta2
                        gradient[j]=tmp+tmp2
                    result[i+n1]=-np.dot(np.diff(gradient),evalC)
                print "derivativeW done"
                return h,result
        else:
            return h
    else:
        ###this part has to be changed!!!!!!!
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
    B=Bobs
#    for i in xrange(n):
 #       B[i]=Bparameters(np.array(x).reshape((1,n1)),X[i,:],W[i,:],n1,n2,variance0,alpha1,alpha2,lambdaParameter)
    inv1=linalg.solve_triangular(L,y2,lower=True)
    inv2=linalg.solve_triangular(L,B.transpose(),lower=True)
    aN=muStart+np.dot(inv2.transpose(),inv1)
    if gradient==True:
        parameterLamb=np.zeros(n2)
        for j in xrange(n2):
            parameterLamb[j]=np.sum(lambdaParameter[j])
        quantil=int(poisson.ppf(.99999999,max(parameterLamb)))
        tempVect=itertools.product(range(quantil),repeat=n2)
        s1=[-np.sum(parameterLamb)+np.dot(z,np.log(parameterLamb))-
            np.sum([np.sum([log(i) for i in range(1,z[j]+1)]) for j in range(n2)])
            -sqrt(3)*np.sqrt(np.sum(alpha1*((x[i,:]-X)**2))+np.sum(alpha2*((z-W)**2)))
            for z in tempVect]
       # s=np.sqrt([np.sum(alpha1*((x[i,:]-X)**2))+np.sum(alpha2*((z-W)**2)) for z in tempVect])
       # s2=[(1+sqrt(3)*r)*np.exp(-sqrt(3)*r) for r in s]
        results=variance0*np.sum(np.exp(s1))*-3.0
        gradXB=np.zeros((n1,n))
        for i in xrange(n):
            gradXB[:,i]=results*(alpha1*(x-X[i,:]))
        temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
        temp5=linalg.solve_triangular(L,y2,lower=True)
        gradAn=np.dot(temp5.transpose(),temp4)
        return aN,gradAn
    else:
        return aN