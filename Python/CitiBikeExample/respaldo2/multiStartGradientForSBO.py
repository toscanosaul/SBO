#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import multiprocessing
from VOI_SBO import An
from scipy import linalg
from numpy import linalg as LA
from math import *
from projectionSimplex import *


#Golden Section
# Here q=(x,w)=(xNew,wNew) are vectors of inputs into Vn, (ql,qr)=xl,xr,wl,wr are upper and lower
# values for x,w resp., also given as vectors. However, the function only works
# on the dimension dim
#fn is the function to optimize
def goldenSection(fn,q,ql,qr,dim,tol=1e-8,maxit=100):
    gr=(1+sqrt(5))/2
    ql=np.array(ql)
    ql=ql.reshape(ql.size)
    qr=np.array(qr).reshape(qr.size)
    q=np.array(q).reshape(q.size)
    pl=q
    pl[dim]=ql[dim]
    pr=q
    pr[dim]=qr[dim]
    pm=q
    pm[dim]=pl[dim]+(pr[dim]-pl[dim])/(1+gr)
    FL=fn(pl)
    FR=fn(pr)
    FM=fn(pm)
    tolMet=False
    iter=0
    while tolMet==False:
        iter=iter+1
        if pr[dim]-pm[dim] > pm[dim]-pl[dim]:
            z=pm+(pr-pm)/(1+gr)
            FY=fn(z)
            if FY>FM:
                pl=pm
                FL=FM
                pm=z
                FM=FY
            else:
                pr=z
                FR=FY
        else:
            z=pm-(pm-pl)/(1+gr)
            FY=fn(z)
            if FY>FM:
                pr=pm
                FR=FM
                pm=z
                FM=FY
            else:
                pl=z
                FL=FY
        if pr[dim]-pm[dim]< tol or iter>maxit:
            tolMet=True
    return pm

#we are optimizing in X+alpha*g2
##guarantee that the point is in the compact set ?
def goldenSectionLineSearch (fns,tol,maxtry,X,g2):
     # Compute the limits for the Golden Section search
    ar=np.array([0,2*tol,4*tol])
    fval=np.array([fns(0),fns(2*tol),fns(4*tol)])
    tr=2
    while fval[tr]>fval[tr-1] and tr<maxtry:
        ar=np.append(ar,2*ar[tr])
        tr=tr+1
        fval=np.append(fval,fns(ar[tr]))
    if tr==maxtry:
        al=ar[tr-1]
        ar=ar[tr]
    else:
        al=ar[tr-2]
        ar=ar[tr]
    ##Now call goldensection for line search
    if fval[tr]==-float('inf'):
        return 0
    else:
        return goldenSection(fns,al,al,ar,0,tol=tol)

###steepestAscent uses this function. See its description to understand the parameters.
def Iter (n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,lambdaParameter,points,varianceObservations,Bobs,tol=1e-8,maxit=1000,maxtry=25,repeat=15):
    w1=np.zeros((1,n2))
    for j2 in xrange(n2):
        w1[0,j2]=np.random.random_integers(300,400,size=1)
    sampleSize=points.shape[0]
    randomIndexes=np.random.random_integers(0,sampleSize-1,1)
    x=points[randomIndexes,:]
    tolMet=False
    iter=0
    X=np.concatenate((x,w1),1)
    g1=-100
    while tolMet==False:
        iter=iter+1
        oldEval=g1
        oldPoint=X
        print "Vn1"
        g1,g2=Vn(X[0,0:n1].reshape((1,n1)),X[0,n1:n2+n1].reshape((1,n2)),Xpast,Wpast,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,lambdaParameter,points,varianceObservations,Bobs)
        print g1
        print g2
        ####if g1=0, we already know this observation
        while g1==0:
            if iter==1:
                x=np.array(np.random.uniform(c,d)).reshape((1,n1))
                w1=np.array(np.random.uniform(c,d)).reshape((1,n2))
                X=np.concatenate((x,w1),1)
                g1,g2=Vn(X[0,0:n1].reshape((1,n1)),X[0,n1:n2+n1].reshape((1,n2)),Xpast,Wpast,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,lambdaParameter,points,varianceObservations,Bobs)
            else:
                val[i,0]=oldEval
                val[i,1:]=oldPoint
                tolMet=True
                break
        if (tolMet==True):
            break
        def fns(alpha,X_=oldPoint,g2=g2):
            tmp=X_+alpha*g2
            x_=tmp[0,0:n1]
            w_=tmp[0,n1:n2+n1]
            return Vn(x_,w_,Xpast,Wpast,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,lambdaParameter,points,varianceObservations,Bobs,grad=False)
        
        lineSearch=goldenSectionLineSearch(fns,tol,maxtry,X,g2)
        X=X+lineSearch*g2
        print "linesearch"
        print lineSearch
                
       # if (any(X[0,0:n1]<c)):
       #     index2=np.where(X[0,0:n1]<c)
       #     X[0,index2]=c[index2]
           
        
       # if (any(X[0,0:n1]>d)):
       #     index2=np.where(X[0,0:n1]>d)
       #     X[0,index2]=d[index2]
            
        if (np.sum(X[0,0:n1])!=600 or any(X[0,0:n1]<c)):
            r=project(X[0,0:n1],600)
            X[0,0:n1]=r
     #       MAX=np.argmax(X[0,:])
      #      temp123=np.array(range(n1))
       #     rest=np.delete(temp123,MAX)
        #    X[0,MAX]=600-np.sum(X[0,rest])
            
        if LA.norm(X[0,:]-oldPoint[0,:])<tol or iter > maxit:
            tolMet=True
            g1=Vn(X[0,0:n1].reshape((1,n1)),X[0,n1:n2+n1].reshape((1,n2)),Xpast,Wpast,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,lambdaParameter,points,varianceObservations,Bobs,grad=False)
            return g1,X

###Optimize value of information function Vn(x,w)
###repeat is the number that we start the multi-start gradient algorithm
###n1,n2 are the dimensions of x and w, respectively.
###c,d are the vector where we are looking for the maximum (c<=x<=d)
###Vn is the value of information function
###Xpast,Wpast are the past points used. They are matrices, where each row represents each point
###n is where we evaluate Vn
###yPast are the past observations.
###If discrete is true, the maximum is rounded using the ceil function. Otherwise, it just returns the value
###Output: two vectors: X,W
##Bobs is B(x,i) for i until n
def steepestAscent (n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,lambdaParameter,points,varianceObservations,Bobs,tol=1e-8,maxit=1000,maxtry=25,repeat=3,discrete=True):
    val=np.zeros((repeat,1+n1+n2))
    print "iter0"
    Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,lambdaParameter,points,varianceObservations,Bobs)
   # print "iter1"
    pool=multiprocessing.Pool()
    results_async = [pool.apply_async(Iter,args=(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,lambdaParameter,points,varianceObservations,Bobs,)) for i in range(repeat)]
    output = [p.get() for p in results_async]
    pool.close()
    pool.join()
    print "iter1"
  #  g1,X=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,lambdaParameter,points)
    for i in xrange(repeat):
        val[i,0]=output[i][0]
        val[i,1:]=output[i][1][0,:]
    MAX=np.argmax(val[:,0])
    X=np.array(val[MAX,1:]).reshape((1,n1+n2))
    if discrete==True:
        temp1=np.floor(X[0,0:n1])
        aux=600-np.sum(temp1)
        index=0
        while aux>0:
            temp1[index]=temp1[index]+1
            index=index+1
            index=index%n1
            aux=aux-1
        return temp1,np.ceil(X[0,n1:n2+n1])
    else:
        return X[0,0:n1],X[0,n1:n2+n1]
    

###SteepestAscent_aN uses this function
###algorithm starts at x
def Iter_aN(L,y2,c,d,aN,n1,n2,variance0,alpha1,alpha2,lambdaParameter,muStart,n,Xpast,Wpast,Bobs,poisson=True,tol=1e-8,maxit=1000,maxtry=25):
    x=np.zeros((1,n1))
    if poisson==True:
        for i in xrange(n1):
            x[0,i]=np.random.random_integers(c[i],d[i],1)
    tolMet=False
    iter=0
    while tolMet==False:
        iter=iter+1
        oldPoint=x
        
        g1,g2=aN(x,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,lambdaParameter,muStart,n,Bobs,gradient=True)
        def fns(alpha,x_=oldPoint,g2=g2,L=L,y2=y2):
            tmp=x_+alpha*g2
            return aN(tmp,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,lambdaParameter,muStart,n,Bobs,gradient=False)
        lineSearch=goldenSectionLineSearch(fns,tol,maxtry,x,g2)
        x=x+lineSearch*g2
        
        if (np.sum(x[0,0:n1])!=600 or any(x[0,0:n1]<c)):
            r=project(x[0,0:n1],600)
            x[0,0:n1]=r
        
    #    if (any(x[0,0:n1]<c)):
    #        index2=np.where(x[0,0:n1]<c)
    #        x[0,index2]=c[index2]
           
     #   if (any(x[0,0:n1]>d)):
      #      index2=np.where(x[0,0:n1]>d)
       #     x[0,index2]=d[index2]
            
      #  if (np.sum(x[0,0:n1])>600):
      #      MAX=np.argmax(x[0,:])
      #      temp123=np.array(range(n1))
      #      rest=np.delete(temp123,MAX)
      #      x[0,MAX]=600-np.sum(x[0,rest])

        if LA.norm(x-oldPoint)<tol or iter > maxit:
            tolMet=True
        
    return x,aN(x,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,lambdaParameter,muStart,n,Bobs,gradient=False)

####Optimize a_n(x). aN is the function of an and it also gives the gradient
###n1 dimension of x
##k is the kernel
def SteepestAscent_aN (n,n1,n2,Xpast,Wpast,aN,c,d,yObs,muStart,variance0,alpha1,alpha2,lambdaParameter,k,varianceObservations,Bobs,tol=1e-8,maxit=1000,maxtry=25,repeat=3,discrete=True):
    A=An(n,Xpast,Wpast,k,varianceObservations)
    L=np.linalg.cholesky(A)
    y2=np.array(yObs)-muStart
    val=np.zeros((repeat,1+n1))
    pool=multiprocessing.Pool()
    results_async = [pool.apply_async(Iter_aN,args=(L,y2,c,d,aN,n1,n2,variance0,alpha1,alpha2,lambdaParameter,muStart,n,Xpast,Wpast,Bobs,)) for i in range(repeat)]
    output = [p.get() for p in results_async]
    pool.close()
    pool.join()
    for i in xrange(repeat):
        val[i,0]=output[i][1]
        val[i,1:]=output[i][0][0,:]
    MAX=np.argmax(val[:,0])
    X=np.array(val[MAX,1:])
    X=np.floor(X)
    aux=600-np.sum(X)
    index=0
    while aux>0:
        X[index]=X[index]+1
        index=index+1
        index=index%n1
        aux=aux-1
    
    
  #  if (np.sum(X)>600):
  #      ind=np.argmax(X)
   #     temp123=np.array(range(n1))
   #     rest=np.delete(temp123,ind)
   #     X[ind]=600-np.sum(X[rest])
        
   # if (np.sum(X)<600):
   #     ind=np.argmin(X)
   #     temp123=np.array(range(n1))
   #     rest=np.delete(temp123,ind)
    #    X[ind]=600-np.sum(X[rest])
            
    if discrete==True:
        return np.ceil(X), aN(np.ceil(X),L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,lambdaParameter,muStart,n,Bobs,gradient=False)
    else:
        return X,aN(X,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,lambdaParameter,muStart,n,Bobs,gradient=False)

