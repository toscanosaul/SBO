#!/usr/bin/env python
#Stratified Bayesian Optimization

from math import *

import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,6)
import numpy as np
from matplotlib import pyplot as plt
import GPy
from numpy import multiply
from numpy.linalg import inv
from AffineBreakPoints import *
from scipy.stats import norm
from grid import *
import pylab as plb
from scipy import linalg
#from pylab import *
#from pylab import *

####WRITE THE DOCUMENTATION

#f is the function to optimize
#alpha1,alpha2,sigma0 are the parameters of the covariance matrix of the gaussian process
#alpha1,alpha2 should be vectors
#muStart is the mean of the gaussian process (assumed is constant)
#sigma, mu are the parameters of the distribution of w1\inR^d1
#n1,n2 are the dimensions of x and w1, resp.

#c,d are the vectors where the grid is built
#l is the size of the grid


####X,W are column vectors
###Gaussian kernel: Variance same than in paper. LengthScale is 1/2alpha. Input as [alpha1,alpha2,...]
####Set ARD=True. 
##wStart:start the steepest at that point
###N(w1,sigmaW2) parameters of w2|w1
###mPrior samples for the prior

####include conditional dist and density. choose kernels
###python docstring
###RESTART AT ANY STAGE (STREAM)
###Variance(x,w,x,w)=1 (prior)
class SBO:
    def __init__(self,f,alpha1,alpha2,sigma0,sigma,mu,muStart,n1,n2,l,c,d,wStart,muW2,sigmaW2,M,mPrior=100,old=False,Xold=-1,Wold=-1,yold=-1):
        self._k=GPy.kern.RBF(input_dim=n1+n2, variance=sigma0**2, lengthscale=np.concatenate(((0.5/np.array(alpha1))**(.5),(0.5/np.array(alpha2))**(.5))),ARD=True)+GPy.kern.White(n1+n2,0.1)
        self._variance=np.array(sigma)**2
        self._mu=np.array(mu)
        self._muStart=np.array(muStart)
        self._f=f
        self._n1=n1
        self._n2=n2
        self._alpha2=np.array(alpha2)
        self._variance0=sigma0**2
        self._alpha1=np.array(alpha1)
        self._sizeGrid=l
        self._c=c
        self._d=d
        self._points=grid(c,d,l)
        self._wStart=wStart
        self._old=old
        self._mPrior=mPrior
        self._noise=0.1
        self._varianceObservations=np.zeros(0)
        ###old is true if we are including data to the model
        if old==False:
            self._X=np.zeros([0,n1]) #points used
            self._W=np.zeros([0,n2]) #points used
            self._y=np.zeros([0,1]) #evaluations of the function
            self._n=0 #stage of the algorithm
            self._Xprior=np.random.uniform(c,d,mPrior).reshape((self._mPrior,self._n1)) ##to estimate the kernel
            self._wPrior=np.random.normal(self._wStart,1,mPrior).reshape((self._mPrior,self._n2))
            self._mPrior=mPrior
        else:
            self._X=Xold
            self._W=Wold
            self._y=yold
            self._n=yold.shape[0]
            
            
            
       # self._wStart=wStart
        self._muW2=muW2
        self._sigmaW2=sigmaW2
        self._M=M
        
     #   self._mPrior=mPrior
    #    self._Xprior=np.random.uniform(c,d,mPrior).reshape((self._mPrior,self._n1))
     #   self._wPrior=np.random.normal(self._wStart,1,mPrior).reshape((self._mPrior,self._n2))
        
       # self._pointsPlotAn=np.zeros([0,self._points.shpape[0]])
        
    def kernel(self):
        return self._k
    
    def evaluatef(self,x,w1,w2):
        return self._f(x,w1,w2)
    
    ##X,W are the past observations
    ##An is the matrix of the paper
    def An(self,n,X,W):
        An=self._k.K(np.concatenate((X,W),1))+np.diag(self._varianceObservations[0:n])
        return An
    
    ##compute the kernel in the points of each entry of the vector (x,w) against all the past observations
    ##returns a matrix where each row corresponds to each point of the vector (x,w)
    ##no la uso
    def b(self,n,x,w,X,W):
        past=np.concatenate((X,W),1)
        new=np.concatenate((x,w),1)
        bn=self._k.K(new,past)
        return bn
    
    #compute B(x,i). X=X[i,:],W[i,:]
    def B(self,x,X,W):
     
        tmp=-(((self._mu/self._variance)+2.0*(self._alpha2)*np.array(W))**2)/(4.0*(-self._alpha2-(1/(2.0*self._variance))))
      
        tmp2=-self._alpha2*(np.array(W)**2)
        tmp3=-(self._mu**2)/(2.0*self._variance)
        tmp=np.exp(tmp+tmp2+tmp3)
        tmp=tmp*(1/(sqrt(2.0*self._variance)))*(1/(sqrt((1/(2.0*self._variance))+self._alpha2)))
        
        x=np.array(x).reshape((x.size,self._n1))
        tmp1=self._variance0*np.exp(np.sum(-self._alpha1*((np.array(x)-np.array(X))**2),axis=1))

        return np.prod(tmp)*tmp1
    
    
    ##return a_n and b_n
    ##x is a vector of points where a_n and sigma_n are evaluated  
    def update (self,x,n,y,X,W,xNew,wNew):
        if n==0:
            return self._muStart
        x=np.array(x)
        m=x.shape[0]
        A=self.An(n,X,W)
        L=np.linalg.cholesky(A)
      #  Ainv=linalg.inv(A)
        B=np.zeros([m,n])
        for i in xrange(n):
            B[:,i]=self.B(x,X[i,:],W[i,:])
        BN=np.zeros([m,1])
    
        BN[:,0]=self.B(x,xNew,wNew) #B(x,n+1)
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        temp1=linalg.solve_triangular(L,np.array(y)-self._muStart,lower=True)
       # tmp=np.array(y)-self._muStart
       # tmp2=np.dot(Ainv,tmp)
       # tmp3=np.dot(B,tmp2)
        
        a=self._muStart+np.dot(temp2.T,temp1)

        past=np.concatenate((X,W),1)

        new=np.concatenate((xNew,wNew),1).reshape((1,self._n1+self._n2))
        gamma=np.transpose(self._k.K(new,past))
       # b=(BN-np.dot(B,np.dot(Ainv,gamma)))
        temp1=linalg.solve_triangular(L,gamma,lower=True)
    #    temp2=linalg.solve_triangular(L,B.T,lower=True)
        b=(BN-np.dot(temp2.T,temp1))

      #  print np.dot(temp2.T,temp1)
  
     
     #   return
        b2=self._k.K(new,new)-np.dot(temp1.T,temp1)
        
      #  bTest=(b**2)/(self._k.K(new,new)-np.dot(np.dot(np.transpose(gamma),Ainv),gamma))
        #print self._k.K(new,,1))

        
        if( b2>0):
            #b=abs(b)/(sqrt(self._k.K(new,new)-np.dot(np.dot(np.transpose(gamma),Ainv),gamma)))
            b=b/(sqrt(b2))
        else:
            ###this means we already know the point and so has variance zero.
            print "error!!!!!!!"
            b=np.zeros((len(b),1))
          #  b[0,0]=float("inf")
      #  print log(abs(bTest))
      #  print (self._k.K(new,new)-np.dot(np.dot(np.transpose(gamma),Ainv),gamma))
        return a,b,gamma,BN,L,B
        #only if X and W have been updated
        #Ainv is inv(An)
        #y2 is y-muStart
        ##past are all the observations
    def getMu(self, n,x,w,L,y2,X,W):
        past=np.concatenate((X,W),1)
        new=np.concatenate((x,w),1)
        bn=self._k.K(new,past)
        temp1=linalg.solve_triangular(L,np.array(y2)-self._muStart,lower=True)
        temp2=linalg.solve_triangular(L,bn.T,lower=True)
        return self._muStart+np.dot(temp2.T,temp1)
    
    ##posterior parameter sigma
    ###L is the Cholesky factorization
    def getSigmaN(self,n,x,w,x1,w1,L,y2,X,W):
        pas=np.concatenate((X,W),1)
        new=np.concatenate((x,w),1)
        new1=np.concatenate((x1,w1),1)
        temp1=linalg.solve_triangular(L,self._k.K(new1,pas).T,lower=True)
        temp2=linalg.solve_triangular(L,self._k.K(new,pas).T,lower=True)
        #print self._muStart
        temp1=self._k.K(new,new1)-np.dot(temp2.T,temp1)
      #  print "prueba de simetria"
        #print Ainv-Ainv.T
       # print "up"
       # print self._k.K(new1,new)-np.dot(np.dot(self._k.K(new1,pas),Ainv),self._k.K(new,pas).T)
       # print temp1
       # return
        return temp1
        
    #######
    ##evaluate the function h
    ##b has been modified in affineBreakPointsPrep
    def h (self,b,c,keep):
        M=len(keep)
        if M>1:
            c=c[keep+1]
            c2=-np.abs(c[0:M-1])
            tmp=norm.pdf(c2)+c2*norm.cdf(c2) 
            return np.sum(np.diff(b[keep])*tmp)
        else:
            return 0
    
    ###only for n>0
    def VarF(self,n,x):
        X=self._X
        W=self._W
        if n==0:
            return 
        x=np.array(x)
        m=x.shape[0]
        A=self.An(n,X,W)
        L=np.linalg.cholesky(A)
      #  Ainv=linalg.inv(A)
        B=np.zeros([m,n])
        for i in xrange(n):
            B[:,i]=self.B(x,X[i,:],W[i,:])
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        return self._variance0*.5*(1.0/((.25+self._alpha2)**.5))-np.dot(temp2.T,temp2)
      #  bTest=(b**2)/(self._k.K(new,new)-np.dot(np.dot(np.transpose(gamma),Ainv),gamma))
        #print self._k.K(new,,1))
    
    #compute Vn and its gradient at xNew,wNew, evaluated in the points
    #if grad=True, the gradient is computed
    ###xNew,wNew are matrices of dimension 1xn1,and 1xn2 resp.
    def Vn (self,xNew,wNew,X,W,n,y,grad=True):
        if n>1:
            n=n-1
            a,b,gamma,BN,L,B=self.update(self._points,n,y,X,W,xNew,wNew)
            if np.all(b==np.zeros((len(b),1))):
                if grad==False:
                    return 0
                else:
                    return 0,0
       
       #         if grad==False:
        #            return float("-inf")
         #       else:
          #          return float("-inf"),0
            a,b,keep=AffineBreakPointsPrep(a,b)
            keep1,c=AffineBreakPoints(a,b)
            ###ask for index which is infinity and remove if ()
            ###Test/borrar
          #  h10=1e-2
         #   a2,b2,gamma2,BN2,L2,B2=self.update(self._points,n,y,X,W,xNew,wNew-h10)
          
          #  a3,b3,gamma3,BN3,L3,B3=self.update(self._points,n,y,X,W,xNew,wNew+h10)
            #############
            
            #######
            keep1=keep1.astype(np.int64)
            h=self.h(b,c,keep1) ##Vn
            ###Now we compute the gradient
            a=a[keep1]
            b=b[keep1]
            keep=keep[keep1] #indices conserved
            M=len(keep)
            if grad==True:
                if M<=1:
                    return 0,0
                else:
                    gradXSigma0=np.zeros([n+1,self._n1])
                    gradWSigma0=np.zeros([n+1,self._n2])
                    gradXB=np.zeros([len(keep1),self._n1])
                    gradWB=np.zeros([len(keep1),self._n2])

                    ####borrar
                    for i in xrange(n):
                        gradXSigma0[i,:]=-2.0*gamma[i]*self._alpha1*(xNew-X[i,:])
                        gradWSigma0[i,:]=-2.0*gamma[i]*self._alpha2*(wNew-W[i,:])

                    for i in xrange(self._n1):
                        gradXB[:,i]=-2.0*self._alpha1[i]*BN[keep,0]*(xNew[0,i]-self._points[keep,i])
                    for i in xrange(self._n2):
                        temp100=(self._alpha2[i]*(self._mu[i]-wNew[0,i]))
                        temp101=self._variance[i]*self._alpha2[i]+.5
                        gradWB[:,i]=BN[keep,0]*temp100/temp101
                    gradB=np.concatenate((gradXB,gradWB),1).transpose()
                    gradientGamma=np.concatenate((gradXSigma0,gradWSigma0),1).transpose()
                ######    #######prubea
         #           print "gradient"
          
          #          print np.exp(np.log(abs(gamma2-gamma3))-log(2*h10))
           #         print "teorical"
            #        print gradWSigma0
                    #print self._k.K(np.concatenate((xNew,wNew),1))
                 #   print "variacion"
                  #  return
                #    print BN
                  #  print "son iguales"
                  #  print -2.0*self._alpha1[0]*BN[keep[0],0]*(xNew[0,0]-self._points[keep[0],0])
                 #   return
                    #############
                    inv1=linalg.solve_triangular(L,B[keep,:].transpose(),lower=True)
                    inv2=linalg.solve_triangular(L,gradientGamma[:,0:n].transpose(),lower=True)
                    tmp=np.dot(inv2.T,inv1)
                    inv3=linalg.solve_triangular(L,gamma,lower=True)
                    beta1=(self._k.K(np.concatenate((xNew,wNew),1))-np.dot(inv3.T,inv3))
                    tmp=(beta1**(-.5))*(gradB-tmp)
                    
                    beta2=BN[keep,:]-np.dot(inv1.T,inv3)

                    tmp2=(.5)*(beta1**(-1.5))*(2.0*np.dot(inv2.T,inv3))
                    tmp2=np.dot(tmp2,beta2.transpose())
                   # tmp2=tmp2*beta2
  
                    #tmp2=np.dot(beta2,tmp2)
                    gradient=tmp+tmp2
                    #######prubea 2
              #      print "gradient"
                  #  print tmp2
                 #   print b2
              #      print np.exp(np.log(abs(b2[keep]-b3[keep]))-log(2*h10))
               #     print "teorical"
                #    print gradient
                 #   return 
                    #print self._k.K(np.concatenate((xNew,wNew),1))
                 #   print "variacion"
                  #  return
                    ############
                    c=c[keep1+1]
                    c2=np.abs(c[0:M-1])
                    tmp=norm.pdf(c2)
                    Gradient=-np.dot(np.diff(gradient),tmp)
                    return h,Gradient
            else:
                return h
        else:
            ###CAMBIAR!!!
            m=self._points.shape[0]
            BN=np.zeros([m,1])
            BN[:,0]=self.B(self._points,xNew,wNew) #B(x,n+1)
            
            b=BN[:,0]
            a=np.repeat(self._muStart,m)
            a,b,keep=AffineBreakPointsPrep(a,b)
            keep1,c=AffineBreakPoints(a,b)
            keep1=keep1.astype(np.int64)
            M=len(keep1)
            h=self.h(b,c,keep1)
            keep=keep[keep1]
            if M<=1:
                if grad==True:
                    return 0,0
                else:
                    return 0
            else:
                if grad==True:
                    gradXB=np.zeros([len(keep1),self._n1])
                    gradWB=np.zeros([len(keep1),self._n2])
                    for i in xrange(self._n1):
                        gradXB[:,i]=-2.0*self._alpha1[i]*BN[keep,0]*(xNew[0,i]-self._points[keep,i])
                    for i in xrange(self._n2):
                        temp100=(self._alpha2[i]*(self._mu[i]-wNew[0,i]))
                        temp101=self._variance[i]*self._alpha2[i]+.5
                        gradWB[:,i]=BN[keep,0]*temp100/temp101
                    gradB=np.concatenate((gradXB,gradWB),1).transpose()
                    c=c[keep1+1]
                    c2=np.abs(c[0:M-1])
                    tmp=norm.pdf(c2)
                    Gradient=-np.dot(np.diff(gradB),tmp)
                    return h, Gradient
                else:
                    return h
                
            
            

    
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
    
    # Steepest Ascent Function
    #(x,w1) is where the algorithm starts
    
    ##Using M samples for W2
    ###tal vez no usare self._M
    def updateY (self,n,X,W):
        t=np.array(np.random.normal(W[n,:],self._sigmaW2,self._M))
        t=self._f(X[n,:],W[n,:],t)
        
        return np.mean(t),float(np.var(t))/self._M
        #var=np.var(t)
        #m1=floor((1536.0)*var)
      #  print m1
        n1=(self._M)
        n2=self._M
        i=0
        t2=np.array(np.random.normal(W[n,:],self._sigmaW2,n2))
        t2=self._f(X[n,:],W[n,:],t2)
       # while ((np.var(t))>((.1**2)*n1)/(4*(1.96**2))):
       #    # print i
       #     n2=2*n2
       #     n1=n2+n1
       #     i=i+1
       #     t2=np.array(np.random.normal(W[n,:],self._sigmaW2,n2))
       #     
       #     t2=self._f(X[n,:],W[n,:],t2)
       #     t=np.concatenate((t,t2))
     #   return np.mean(t)
        
    
    def aN_grad(self,x,L,y2,gradient=True):
        B=np.zeros(self._n)
        for i in xrange(self._n):
            B[i]=self.B(x,self._X[i,:],self._W[i,:])
        inv1=linalg.solve_triangular(L,y2,lower=True)
        inv2=linalg.solve_triangular(L,B.transpose(),lower=True)
        aN=self._muStart+np.dot(inv2.transpose(),inv1)
        if gradient==True:
            gradXB=np.zeros((self._n1,self._n))
            for i in xrange(self._n):
                gradXB[:,i]=B[i]*(-2.0*self._alpha1*(x-self._X[i,:]))
            temp4=linalg.solve_triangular(L,gradXB.transpose(),lower=True)
            temp5=linalg.solve_triangular(L,y2,lower=True)
            gradAn=np.dot(temp5.transpose(),temp4)
           # gradAn=np.dot(tmp4,gradXB.transpose())
            return aN,gradAn
        else:
            return aN
    
        ##naive multi-start!
    
    #(self,n,x,w1,tol=1e-8,maxit=1000,maxtry=25)
    def steepestAscent (self,n,tol=1e-8,maxit=1000,maxtry=25):
        
        tolMet=False
        iter=0
   #     x=np.random.uniform(self._c,self._d)
    #    w1=np.random.uniform(self._c,self._d)
        repeat=3 #multi-Start

        val=np.zeros((repeat,1+self._n1+self._n2))
      #  X=np.concatenate((x,w1),1)
       # X2=np.concatenate((self._X,self._W),1)
        
     #   while (len(np.where(np.all(X2==x,axis=1))[0])>0):
      #      print "arreglando"
       #     X=X+0.5
        
        for i in xrange(repeat):
            x=np.array(np.random.uniform(self._c,self._d)).reshape((1,self._n1))
            w1=np.array(np.random.uniform(self._c,self._d)).reshape((1,self._n2))
            tolMet=False
            iter=0
            X=np.concatenate((x,w1),1)
            g1=-100
            while tolMet==False:
                iter=iter+1
                oldEval=g1
                oldPoint=X
    
                g1,g2=self.Vn(X[0,0:self._n1].reshape((1,self._n1)),X[0,self._n1:self._n2+self._n1].reshape((1,self._n2)),self._X,self._W,n,self._y)
                while g1==0:
                    if iter==1:
                      #  print n
                        x=np.array(np.random.uniform(self._c,self._d)).reshape((1,self._n1))
                        w1=np.array(np.random.uniform(self._c,self._d)).reshape((1,self._n2))
                        X=np.concatenate((x,w1),1)
                        g1,g2=self.Vn(X[0,0:self._n1].reshape((1,self._n1)),X[0,self._n1:self._n2+self._n1].reshape((1,self._n2)),self._X,self._W,n,self._y)
                    else:
                       # print n
                        val[i,0]=oldEval
                        val[i,1:]=oldPoint
                        tolMet=True
                        break
                    
                if (tolMet==True):
                    break
                def fns(alpha,X_=oldPoint,g2=g2):
                    tmp=X_+alpha*g2
                    #s1=tmp[0,0:self._n1].reshape((1,self._n1))<=self._d
                    #s2=self._c<=tmp[0,0:self._n1].reshape((1,self._n1))
                    #s3=np.squeeze(np.concatenate((s1,s2),1))
                    #if len([1 for l in s3 if l==False])==0:
                    x_=tmp[0,0:self._n1]
                    w_=tmp[0,self._n1:self._n2+self._n1]
                    return self.Vn(x_,w_,self._X,self._W,n,self._y,grad=False)
                   # else:
                   #     return -float('inf')
                
    
                lineSearch=self.goldenSectionLineSearch(fns,tol,maxtry,X,g2)
                
    
                
                X=X+lineSearch*g2

    
                    
                if max(np.reshape(abs(X-oldPoint),X.shape[1]))<tol or iter > maxit:
                    val[i,0]=g1
                    val[i,1:]=X
                    
                    tolMet=True
        
        MAX=np.argmax(val[:,0])
     #   print MAX
       # print max (val[:,0])
        if max (val[:,0])<1e-6:
            return -1,-1,-1,-1,-1
        
        X=np.array(val[MAX,1:]).reshape((1,self._n1+self._n2))
        self._X=np.vstack([self._X,X[0,0:self._n1]])
        self._W=np.vstack([self._W,X[0,self._n1:self._n2+self._n1]])
        ###change for a funtion that updates y
        temp34=self.updateY(self._n,self._X,self._W)
        self._varianceObservations=np.append(self._varianceObservations,temp34[1])
        self._y=np.vstack([self._y,temp34[0]])
        self._n=self._n+1
        
        f=open("oldX.txt",'w')
        g=open("oldW.txt",'w')
        f2=open("oldY.txt",'w')
        np.savetxt(f,self._X)
        np.savetxt(g,self._W)
        np.savetxt(f2,self._y)
        f2.close()
        f.close()
        g.close()
      #  m = GPy.models.GPRegression(np.concatenate((self._X,self._W),1),self._y,self._k)
      #  m.optimize()
      #  self._alpha1=np.array(0.5/(np.array(m['.*rbf'].values()[1:self._n1+1])**2))
      #  self._alpha2=np.array(0.5/(np.array(m['.*rbf'].values()[1+self._n1:self._n1+self._n2+1])**2))
      #  self._variance0=np.array(m['.*rbf'].values()[0])
      #  self._muStart=np.array(m['.*var'].values()[1])
      #  print val[MAX,0]
        
        return X[0,0:self._n1],X[0,self._n1:self._n2+self._n1],self._y[n-1,0],n,max(val[:,0])

    ###this function is very particular to this example
    def Fvalues(self,x,w1):
        return -x**2-w1
        
    
    def muN(self,C,D,n):
        X=self._X
        W=self._W
        y=self._y[:,0]
        m=C.shape[0]
        A=self.An(n,X,W)
        L=np.linalg.cholesky(A)
      #  Ainv=linalg.inv(A)
        B=np.zeros([m*m,n])
        muN=np.zeros((m,m))
        temp1=linalg.solve_triangular(L,np.array(y)-self._muStart,lower=True)
        for j in xrange(m):
            for k in xrange(m):
                for i in xrange(n):
                    B[j+k,i]=self._k.K(np.concatenate((np.array([[C[j,k]]]),np.array([[D[j,k]]])),1),np.concatenate((X[i:i+1,:],W[i:i+1,:]),1))[:,0]
                temp2=linalg.solve_triangular(L,B[j+k:j+k+1,:].T,lower=True)
                muN[j,k]=self._muStart+np.dot(temp2.T,temp1)

      #  temp2=linalg.solve_triangular(L,B.T,lower=True)
 
 #       muN=self._muStart+np.dot(temp2.T,temp1)
        return muN
    
    def graphAn (self,n):
        A=self.An(n,self._X[0:n,:],self._W[0:n,:])
       # Ainv=linalg.inv(A)
        L=np.linalg.cholesky(A)
        y2=np.array(self._y[0:n])-self._muStart
        m=self._points.shape[0]
        z=np.zeros(m)
        var=np.zeros(m)
        
        ###countours for F
        w1=np.linspace(-3,3,m)
        points=self._points[:,0]
        X,W=np.meshgrid(points,w1)
        Z=self.Fvalues(X,W)
        Z2=self.muN(X,W,n)
        fig=plt.figure()
        CS=plt.contour(X,W,Z2)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('Contours of estimation of F(x,w)')
        plt.xlabel('x')
        plt.ylabel('w')
        plt.savefig('%d'%n+"Contours of estimation of F.pdf")
        plt.close(fig)
        fig=plt.figure()
        CS=plt.contour(X,W,Z)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('Contours of F(x,w)')
        plt.xlabel('x')
        plt.ylabel('w')
        plt.savefig('%d'%n+"F.pdf")
        plt.close(fig)
        
        #####contors for VOI
        VOI=np.zeros((m,m))
        for i in xrange(m):
            for j in xrange(m):
                VOI[i,j]=self.Vn(np.array([[X[i,j]]]),np.array([[W[i,j]]]),self._X[0:n-1,:],self._W[0:n-1,:],n,self._y[0:n-1,:],False)
        fig=plt.figure()
        CS=plt.contour(X,W,VOI)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('Contours of VOI')
        plt.xlabel('x')
        plt.ylabel('w')
        plt.savefig('%d'%n+"VOI.pdf")
        plt.close(fig)
        ######
        for i in xrange(m):
            z[i]=self.aN_grad(self._points[i,:],L,y2,gradient=False)
            var[i]=self.VarF(n,self._points[i,:])
   #     a,b,gamma,BN,L,B=self.update(self._points,n,self._y,self._X,self._W,xNew,wNew)
        
        fig=plt.figure()
        plt.plot(self._points,-(self._points**2))
        plt.plot(self._points,z,'--')
        confidence=z+1.96*(var**.5)
        plt.plot(self._points,confidence,'--',color='r')
        confidence2=z-1.96*(var**.5)
        plt.plot(self._points,confidence2,'--',color='r')
       # plt.plot(self._points,z+1.96*b,'--',color='r')
       # plt.plot(self._points,z-1.96*b,'--',color='r')
        plt.title("Graph of $a_n(x)$ vs G(x)")
        plt.xlabel('x')
        plt.savefig('%d'%n+"a_n.pdf")
        plt.close(fig)
        
        
        
        
      #  print z
       # print -self._points**2
       
        ####if old=true, this produces n more points
    def getNPoints(self,n,tol=1e-8,maxit=1000,maxtry=25):
        if self._old==False:
            w=self._wStart
          #  m=100
            ####m is the number of samples for the prior
           # X=np.zeros([0,self._n1])
           # W=np.zeros([0,self._n2]) #
            y=np.zeros([0,1]) 
           # X=np.vstack([X,x])
           # W=np.vstack([W,w])
            #print self._Xprior
            ##print self._wPrior
            for i in xrange(self._mPrior):
                y=np.vstack([y,self.updateY(i,self._Xprior,self._wPrior)[0]])
            
            #y=np.vstack([y,self.updateY(0,self._Xp,W)])
            #self._n=self._n+1
            
            self._model = GPy.models.GPRegression(np.concatenate((self._Xprior,self._wPrior),1),y,self._k)
            self._model.optimize()
            m=self._model
            #self._plots=[]
            #self._plots.append(m)
            self._alpha1=np.array(0.5/(np.array(m['.*rbf'].values()[1:self._n1+1]))**2)
            self._alpha2=np.array(0.5/(np.array(m['.*rbf'].values()[self._n1+1:self._n1+1+self._n2]))**2)
            self._variance0=np.array(m['.*rbf'].values()[0])
            self._muStart=np.array(m['.*var'].values()[1])
            
            f=open("hyperparameters.txt",'w')
            np.savetxt(f,self._alpha1)
            np.savetxt(f,self._alpha2)
            np.savetxt(f,np.array([self._variance0]))
            np.savetxt(f,np.array([self._muStart]))
            f.close()
            
           # x=np.mean(self._Xprior,axis=0)
            
            
            for i in xrange(1,n+1):
                
                
                print i
               # t=np.array(np.random.normal(x)).reshape((1,self._n2))
                
                #t2=(self._d-np.array(t))/2
              #  x=t
               # w=np.array(np.random.normal(w)+1).reshape((1,self._n2))
                
                x1,w1,y1,n1,maxVn=self.steepestAscent(i)
                ##checar multidimensional
                
                #self._points=np.delete(self._points,np.where(self._points==x1[0]))
                self.graphAn(i)
              #  np.savetxt(f,x1)

                print i
             #   if (maxVn<1e-2):
              #      break
                if maxVn==-1:
                    print "great approximation!"
                    return
                w=w1
                x=x1
            
            
        else:
            n1=self._n
            for i in xrange(1,n+1):
                print i
               # t=np.array(np.random.normal(x)).reshape((1,self._n2))
                
                #t2=(self._d-np.array(t))/2
              #  x=t
               # w=np.array(np.random.normal(w)+1).reshape((1,self._n2))
                
                x1,w1,y1,n1,maxVn=self.steepestAscent(n1+i)
                ##checar multidimensional
                
                #self._points=np.delete(self._points,np.where(self._points==x1[0]))
                self.graphAn(n1+i)
                print i
                w=w1
                x=x1
             #   print maxVn
                if (maxVn==-1):
                    print "great approximation!"
                    return 
             #   if (maxVn<1e-2):
              #      break
                w=w1
                x=x1
    
    ##for an
    

      ##for an  



    
    def optimize(self,n):
        self.getNPoints(n)
        x,val=self.SteepestAscent2(self._X[self._n-1,:])
        return x,val
    
    def SteepestAscent2 (self,x,tol=1e-8,maxit=1000,maxtry=25):
        
        A=self.An(self._n,self._X,self._W)
        L=np.linalg.cholesky(A)
      #  Ainv=linalg.inv(A)
        y2=np.array(self._y)-self._muStart
        tolMet=False
        iter=0
        
        while tolMet==False:

            iter=iter+1
            oldPoint=x
            g1,g2=self.aN_grad(x,L,y2)
   
            def fns(alpha,x_=oldPoint,g2=g2,L=L,y2=y2):
                tmp=x_+alpha*g2
                #s1=tmp<=self._d
                #s2=self._c<=tmp
                #s3=np.squeeze(np.concatenate((s1,s2),1))
                #if len([1 for l in s3 if l==False])==0:
                return self.aN_grad(tmp,L,y2,gradient=False)
                #else:
                #    return -float('inf')
            #print oldPoint
            lineSearch=self.goldenSectionLineSearch(fns,tol,maxtry,x,g2)
    
            

            x=x+lineSearch*g2
           # print x
           # print "\n"
       
            
            if max(np.reshape(abs(x-oldPoint),x.shape[1]))<tol or iter > maxit:
                tolMet=True
       # print maxit
        return x,self.aN_grad(x,L,y2,gradient=False)

        
    def getPoints(self):
        A=self.An(self._n,self._X,self._W)
        return self._X,self._W,self._y,self._variance0,self._muStart,self._alpha1,self._alpha2, linalg.inv(A)
    


def g(x,w1,w2):
    val=(w2+8.0)/(w1+8.0)
    return -(val)*(x**2)-w1



    
    
if __name__ == '__main__':
    np.random.seed(10)
    alpha1=[.5]
    alpha2=[.5]
    sigma0=10
    sigma=[1]
    mu=[0]
    muStart=0
    f=g
    a=[-3]
    b=[3]
    n1=1
    n2=1
    l=50
    w=np.array([1])
    muW2=np.array([0])
    sigmaW2=np.array([1])
    M=1000
    alg=SBO(f,alpha1,alpha2,sigma0,sigma,mu,muStart,n1,n2,l,a,b,w,muW2,sigmaW2,M)
    opt=alg.optimize(15)
    print opt[0]
    
    print "optimal value"
    print opt[1]
 #   print alg._X
 #   print alg._W
 #   print alg._y
    
   # alg.graphAn(2)
 #   t=alg.getPoints()
   # print t[0]
   # print t[2]
   # print t[1]
   # print t[3:]
  #  print alg.aN_grad(np.array([3]),t[7],np.array(t[2])-t[4],gradient=True)

    #####Read Files
    X=np.loadtxt("oldX.txt",ndmin=2)
    W=np.loadtxt("oldW.txt",ndmin=2)
    Y=np.loadtxt("oldY.txt",ndmin=2)
    T=np.loadtxt("hyperparameters.txt",ndmin=2)
    alpha1=T[0,:]
    alpha2=T[1,:]
    sigma0=T[2,:][0]
    muStart=T[3,:][0]
  #  alg=SBO(f,alpha1,alpha2,sigma0,sigma,mu,muStart,n1,n2,l,a,b,w,muW2,sigmaW2,M,old=True,Xold=X,Wold=W,yold=Y)
  #  print alg._X
  #  print alg._W
  #  print alg._y
  #  print alg.optimize(1)
  #  print alg._X
  #  print alg._W
  #  print alg._y

