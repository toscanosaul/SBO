#!/usr/bin/env python

import numpy as np
from scipy.optimize import *
#from misc import getOptimizationMethod
from warnings import warn
from math import *
from scipy import linalg
from numpy import linalg as LA
import SBOGeneral2 as SB


class Optimization:
    def __init__(self,xStart,maxfun=1e4,maxIters=1e3,ftol=None,xtol=None,gtol=None,bfgsFactor=None):
        self.optMethod=None
        self.xStart=xStart
        self.maxFun=maxfun
        self.maxIters=maxIters
        self.ftol=ftol
        self.xtol=xtol
        self.gtol=gtol
        self.xOpt=None
        self.fOpt=None
        self.status=None
        self.bfgsFactor=bfgsFactor
        self.gradOpt=None
        self.nIterations=None

    def run(self, **kwargs):
        self.opt(**kwargs)
    
        
    def opt(self,f=None,fp=None):
        raise NotImplementedError, "optimize needs to be implemented"
    

class OptBFGS(Optimization):
    def __init__(self, *args, **kwargs):
        Optimization.__init__(self,*args,**kwargs)
        self.Name="bfgs"
    
    #fd is the derivative
    def opt(self,f=None,df=None):
        assert df!=None, "Derivative is necessary"
        
        statuses = ['Converged', 'Maximum number of f evaluations reached', 'Error']
    
        dictOpt={}
        if self.gtol is not None:
            dictOpt['pgtol']=self.gtol
        if self.bfgsFactor is not None:
            dictOpt['factr']=self.bfgsFactor
            
        optResult=fmin_l_bfgs_b(f,self.xStart,df,maxfun=self.maxFun,**dictOpt)
  
        self.xOpt=optResult[0]
        self.fOpt = optResult[1]
        self.status=statuses[optResult[2]['warnflag']]
        
        
class OptSteepestDescent(Optimization):
    def __init__(self,n1,projectGradient=None, *args, **kwargs):
        Optimization.__init__(self,*args,**kwargs)
        self.Name="steepest"
        self.maxtry=25
        self.constraintA=-3.0
        self.constraintB=3.0
        self.n1=n1
        self.projectGradient=projectGradient
        
        #Golden Section
    # Here q=(x,w)=(xNew,wNew) are vectors of inputs into Vn, (ql,qr)=xl,xr,wl,wr are upper and lower
    # values for x,w resp., also given as vectors. However, the function only works
    # on the dimension dim
    #fn is the function to optimize
    def goldenSection(self,fn,q,ql,qr,dim,tol=1e-8,maxit=100):
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
    
    def goldenSectionLineSearch (self,fns,tol,maxtry,X,g2):
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
            return self.goldenSection(fns,al,al,ar,0,tol=tol)
    
    ###borrar esta funcion
    def steepestAscent(self,f):
        xStart=self.xStart
        tol=self.xtol
        maxit=self.maxIters
        maxtry=self.maxtry
        tolMet=False
        iter=0
        X=xStart
        g1=-100
        
        n1=self.n1
        c=self.constraintA
        d=self.constraintB
        
        while tolMet==False:
            iter=iter+1
            oldEval=g1
            oldPoint=X
                
            g1,g2=f(X,grad=True)
            if (tolMet==True):
                break
            def fns(alpha,X_=oldPoint,g2=g2):
                tmp=X_+alpha*g2
                return f(tmp,grad=False)
            lineSearch=self.goldenSectionLineSearch(fns,tol,maxtry,X,g2)
            X=X+lineSearch*g2
            X[0,0:n1]=self.projectGradient(X[0,0:n1])
        #    if (any(X[0,0:n1]<c)):
        #        temp1=np.array(X[0,0:n1]).reshape(n1)
        #        index2=np.where(temp1<c)
        #        X[0,index2[0]]=c[index2[0]]
           
         #   if (any(X[0,0:n1]>d)):
         #       index2=np.where(X[0,0:n1]>d)
         #       X[0,index2[0]]=d[index2[0]]

            
            if LA.norm(X[0,:]-oldPoint[0,:])<tol or iter > maxit:
                tolMet=True
                g1,g2=f(X,grad=True)
                return X,g1,g2,iter
                
    
    #f gives both the function and the derivative
    def opt(self,f=None,df=None):
        x,g,g1,it=self.steepestAscent(f)
        self.xOpt=x
        self.fOpt =g
        self.gradOpt=g1
        self.nIterations=it
        
##x is a string with the name of the method we want to use.
##x: 'bfgs',..
def getOptimizationMethod(x):
    optimizers={'bfgs': OptBFGS,'steepest':OptSteepestDescent}
    
    for optMethod in optimizers.keys():
        if optMethod.lower().find(x.lower()) != -1:
            return optimizers[optMethod]
        
    raise KeyError('No optimizer was found matching the name: %s' % x)



        
    
    
    
        
        
