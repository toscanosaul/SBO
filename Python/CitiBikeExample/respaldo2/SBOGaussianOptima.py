#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import multiprocessing
from scipy import linalg
from numpy import linalg as LA
from math import *
from VOIsboGaussian import An


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

###borrar esta funcion
def steepestAscent(f,xStart,tol=1e-8,maxit=1000,maxtry=25):
    tolMet=False
    iter=0
    X=xStart
  #  x=np.array(np.random.uniform(c,d)).reshape((1,n1))
   # w1=np.array(np.random.uniform(c,d)).reshape((1,n2))
    #tolMet=False
    #iter=0
    #X=np.concatenate((x,w1),1)
    g1=-100
    while tolMet==False:
        iter=iter+1
        oldEval=g1
        oldPoint=X
            
        g1,g2=f(X)

        if (tolMet==True):
            break
        def fns(alpha,X_=oldPoint,g2=g2):
            tmp=X_+alpha*g2
            x_=tmp[0,0:n1]
            w_=tmp[0,n1:n2+n1]
            return Vn(x_,w_,Xobs,Wobs,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,points,varianceObservations,False)
        lineSearch=goldenSectionLineSearch(fns,tol,maxtry,X,g2)
        X=X+lineSearch*g2

        if (any(X[0,0:n1]<c)):
            temp1=np.array(X[0,0:n1]).reshape(n1)
            index2=np.where(temp1<c)
            X[0,index2[0]]=c[index2[0]]
           
        if (any(X[0,0:n1]>d)):
            index2=np.where(X[0,0:n1]>d)
            X[0,index2[0]]=d[index2[0]]

        if LA.norm(X[0,:]-oldPoint[0,:])<tol or iter > maxit:
            g1=Vn(X[0,0:n1].reshape((1,n1)),X[0,n1:n2+n1].reshape((1,n2)),Xobs,Wobs,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,points,varianceObservations,grad=False)
            return g1,X
            tolMet=True
    

###steepestAscent uses this function. See its description to understand the parameters.
def Iter (n,n1,n2,c,d,Vn,Xobs,Wobs,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points,tol=1e-8,maxit=1000,maxtry=25,repeat=1):
    tolMet=False
    iter=0
    x=np.array(np.random.uniform(c,d)).reshape((1,n1))
    w1=np.array(np.random.uniform(c,d)).reshape((1,n2))
    tolMet=False
    iter=0
    X=np.concatenate((x,w1),1)
    g1=-100
    while tolMet==False:
        iter=iter+1
        oldEval=g1
        oldPoint=X
            
        g1,g2=Vn(X[0,0:n1].reshape((1,n1)),X[0,n1:n2+n1].reshape((1,n2)),Xobs,Wobs,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,points,varianceObservations)
      #  while g1==0:
       #     if iter==1:
       #       #  print n
       #         x=np.array(np.random.uniform(c,d)).reshape((1,n1))
       #         w1=np.array(np.random.uniform(c,d)).reshape((1,n2))
       #         X=np.concatenate((x,w1),1)
       #         g1,g2=Vn(X[0,0:n1].reshape((1,n1)),X[0,n1:n2+n1].reshape((1,n2)),Xobs,Wobs,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,points,varianceObservations)
       #     else:
       #        # print n
       #         val[i,0]=oldEval
       #         val[i,1:]=oldPoint
       #         tolMet=True
       #         break

        if (tolMet==True):
            break
        def fns(alpha,X_=oldPoint,g2=g2):
            tmp=X_+alpha*g2
            x_=tmp[0,0:n1]
            w_=tmp[0,n1:n2+n1]
            return Vn(x_,w_,Xobs,Wobs,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,points,varianceObservations,False)
        lineSearch=goldenSectionLineSearch(fns,tol,maxtry,X,g2)
        X=X+lineSearch*g2

        if (any(X[0,0:n1]<c)):
            temp1=np.array(X[0,0:n1]).reshape(n1)
            index2=np.where(temp1<c)
            X[0,index2[0]]=c[index2[0]]
           
        if (any(X[0,0:n1]>d)):
            index2=np.where(X[0,0:n1]>d)
            X[0,index2[0]]=d[index2[0]]

        if LA.norm(X[0,:]-oldPoint[0,:])<tol or iter > maxit:
            g1=Vn(X[0,0:n1].reshape((1,n1)),X[0,n1:n2+n1].reshape((1,n2)),Xobs,Wobs,n,yPast,muStart,kernel,variance0,alpha1,alpha2,n1,n2,points,varianceObservations,grad=False)
            return g1,X
            tolMet=True

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
def steepestAscent (n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,points,varianceObservations,tol=1e-8,maxit=1000,maxtry=25,repeat=7,discrete=True):
    val=np.zeros((repeat,1+n1+n2))
    output=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points)
    output1=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points)
    output2=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points)
    output3=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points)
    output4=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points)
    output5=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points)
    output6=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points)
    val[0,0]=output[0]
    val[0,1:]=output[1][0,:]
    val[1,0]=output1[0]
    val[1,1:]=output1[1][0,:]
    val[2,0]=output2[0]
    val[2,1:]=output2[1][0,:]
    val[3,0]=output3[0]
    val[3,1:]=output3[1][0,:]
    val[4,0]=output4[0]
    val[4,1:]=output4[1][0,:]
    val[5,0]=output5[0]
    val[5,1:]=output5[1][0,:]
    val[6,0]=output6[0]
    val[6,1:]=output6[1][0,:]
  #  pool=multiprocessing.Pool()
 #   results_async = [pool.apply_async(Iter,args=(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,varianceObservations,points,)) for i in range(repeat)]
  #  output = [p.get() for p in results_async]
  #  pool.close()
  #  pool.join()
  #  g1,X=Iter(n,n1,n2,c,d,Vn,Xpast,Wpast,yPast,muStart,kernel,variance0,alpha1,alpha2,lambdaParameter,points)
  #  for i in xrange(repeat):
   #     val[i,0]=output[i][0]
    #    val[i,1:]=output[i][1][0,:]
    MAX=np.argmax(val[:,0])
    X=np.array(val[MAX,1:]).reshape((1,n1+n2))
    return X
    


###SteepestAscent_aN uses this function
###algorithm starts at x
def Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations,tol=1e-8,maxit=1000,maxtry=25):
    x=np.zeros((1,n1))
    for i in xrange(n1):
        x[0,i]=np.random.uniform(c[i],d[i],1)
    A=An(n,Xpast,Wpast,kernel,varianceObservations)
    L=np.linalg.cholesky(A)
    y2=np.array(yObs)-muStart
    tolMet=False
    iter=0
    while tolMet==False:
        iter=iter+1
        oldPoint=x
        g1,g2=aN(x,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,muStart,n,gradient=True)

        def fns(alpha,x_=oldPoint,g2=g2,L=L,y2=y2):
            tmp=x_+alpha*g2
            return aN(tmp,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,muStart,n,gradient=False)
        lineSearch=goldenSectionLineSearch(fns,tol,maxtry,x,g2)
        x=x+lineSearch*g2
        
        if (any(x[0,0:n1]<c)):
            index2=np.where(x[0,0:n1]<c)
            x[0,index2[0]]=c[index2[0]]
           
        if (any(x[0,0:n1]>d)):
            index2=np.where(x[0,0:n1]>d)
            x[0,index2[0]]=d[index2[0]]
        
        
        if LA.norm(x-oldPoint)<tol or iter > maxit:
            g1=aN(x,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,muStart,n,gradient=False)

            tolMet=True
    return x,g1


def SteepestAscent_aN (n,n1,n2,Xpast,Wpast,aN,c,d,yObs,muStart,variance0,alpha1,alpha2,kernel,varianceObservations,tol=1e-10,maxit=1000,maxtry=25,repeat=9,discrete=True):
    A=An(n,Xpast,Wpast,kernel,varianceObservations)
    L=np.linalg.cholesky(A)
    y2=np.array(yObs)-muStart
    val=np.zeros((repeat,1+n1))
    output=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    output1=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    output2=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    output3=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    output4=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    output5=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    output6=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    output7=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    output8=Iter_aN(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations)
    val[0,0]=output[1]
    val[0,1:]=output[0][0,:]
    val[1,0]=output1[1]
    val[1,1:]=output1[0][0,:]
    val[2,0]=output2[1]
    val[2,1:]=output2[0][0,:]
    val[3,0]=output3[1]
    val[3,1:]=output3[0][0,:]
    val[4,0]=output4[1]
    val[4,1:]=output4[0][0,:]
    val[5,0]=output5[1]
    val[5,1:]=output5[0][0,:]
    val[6,0]=output6[1]
    val[6,1:]=output6[0][0,:]
    val[7,0]=output7[1]
    val[7,1:]=output7[0][0,:]
    val[8,0]=output8[1]
    val[8,1:]=output8[0][0,:]
    
  #  pool=multiprocessing.Pool()
  #  results_async = [pool.apply_async(Iter_aN,args=(L,yObs,c,d,aN,n1,n2,variance0,alpha1,alpha2,muStart,n,Xpast,Wpast,kernel,varianceObservations,)) for i in range(repeat)]
  #  output = [p.get() for p in results_async]
  #  pool.close()
  #  pool.join()
  #  for i in xrange(repeat):
   #     val[i,0]=output[i][1]
   #     val[i,1:]=output[i][0][0,:]
    MAX=np.argmax(val[:,0])
    X=np.array(val[MAX,1:])
  #  print "Aver"
 #   print aN(X,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,muStart,n,gradient=False)
  #  print aN(np.array([0]),L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,muStart,n,gradient=False)
    return X,aN(X,L,y2,Xpast,Wpast,n1,n2,variance0,alpha1,alpha2,muStart,n,gradient=False)

