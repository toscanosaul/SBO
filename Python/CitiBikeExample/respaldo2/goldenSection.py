#!/usr/bin/env python

import numpy as np
from math import *

# Here mu is a vector of inputs into fn, mul and mur are upper and lower
# values for mu, also given as vectors. However, the function only works
# on the dimension dim.

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
    #line

    return pm

def goldenSectionLineSearch (fns,tol,maxtry,X,g2):
      # Compute the limits for the Golden Section search
     ar=np.array([0,2*tol,4*tol])
     fval=np.array([fns(0),fns(2*tol),fns(4*tol)])
     #print "line"
     #print ar
     #print fval
     #print "grad"
     #print g2
     #print "point"
     #print X
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
     return goldenSection(fns,al,ar,al,0,tol=tol)
    
def steepestAscent (g,dg,x,y,tol=1e-8,maxit=1000,maxtry=25):
    tolMet=False
    iter=0

    
    
    while tolMet==False:
        iter=iter+1
        oldPoint1=x
        oldPoint2=y
        g2=dg(x,y)
        X=np.append(np.array(x),np.array(y))
        oldPoint=X
       # print g2
        def fns(alpha,g2=g2):
            tmp=X+alpha*np.array(g2)
            return g(tmp[0],tmp[1])
        print g2
     #   print "veamos"
      #  print X
       # print g2
       # print g1
        lineSearch=goldenSectionLineSearch(fns,tol,maxtry,X,g2)
        print lineSearch
        print "\n"
       # print lineSearch
        X=X+lineSearch*g2
        x=X[0]
        y=X[1]
        #print X
      #  print "steepest"
       # print X
       # print oldPoint
       # print abs(X-oldPoint).reshape(X.shape[1])
        if max(abs(X-oldPoint))<tol or iter > maxit:
            tolMet=True
    
 
    return X


def g(x,y):
    return -(x**2)-y**2

def dg(x,y):
    return np.array([-2*x,-2*y])

if __name__ == '__main__':
    print steepestAscent(g,dg,10,10)