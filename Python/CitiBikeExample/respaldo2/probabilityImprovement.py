#!/usr/bin/env python

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
from numpy import linalg as LA
from scipy.stats import norm
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


####X is column vectors
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
    def __init__(self,f,alpha1,sigma0,sigma,mu,muStart,n1,n2,l,c,d,wStart,muW2,sigmaW2,M,mPrior=20,old=False,Xold=-1,Wold=-1,yold=-1):
        self._k=GPy.kern.RBF(input_dim=n1, variance=sigma0**2, lengthscale=(0.5/np.array(alpha1))**(.5),ARD=True)+GPy.kern.White(n1,0.1)
        self._variance=np.array(sigma)**2
        self._mu=np.array(mu)
        self._muStart=np.array(muStart)
        self._f=f
        self._n1=n1
        self._n2=n2
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
    def An(self,n,X):
        An=self._k.K(X)+np.diag(self._varianceObservations[0:n])

        return An
    
    ##compute the kernel in the points of each entry of the vector (x,w) against all the past observations
    ##returns a matrix where each row corresponds to each point of the vector (x,w)
    ##no la uso
    def b(self,n,x,w,X,W):
        past=np.concatenate((X,W),1)
        new=np.concatenate((x,w),1)
        bn=self._k.K(new,past)
        return bn
    
    ##x is a single point
    def muN(self,x,n,grad=False):
        x=np.array(x)
        m=1
        X=self._X[0:n,:]
        A=self.An(n,X)
        L=np.linalg.cholesky(A)
        x=np.array(x).reshape((1,self._n1))
        B=np.zeros([m,n])
        
        for i in xrange(n):
            B[:,i]=self._k.K(x,X[i:i+1,:])
        y=self._y[0:n,:]
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        temp1=linalg.solve_triangular(L,np.array(y)-self._muStart,lower=True)
        a=self._muStart+np.dot(temp2.T,temp1)
        if grad==False:
            return a
        x=np.array(x).reshape((1,self._n1))
        gradX=np.zeros((n,self._n1))
        gradi=np.zeros(self._n1)
        temp3=linalg.solve_triangular(L,y-self._muStart,lower=True)
        
        for j in xrange(self._n1):
            for i in xrange(n):
                gradX[i,j]=self._k.K(x,X[i,:].reshape((1,self._n1)))*(2.0*self._alpha1[j]*(x[0,j]-X[i,j]))
            temp2=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
            gradi[j]=self._muStart+np.dot(temp2.T,temp3)
        return a,gradi
    
    
    def varN(self,x,n,grad=False):
        temp=self._k.K(np.array(x).reshape((1,self._n1)))
        sigmaVec=np.zeros((n,1))
        for i in xrange(n):
            sigmaVec[i,0]=self._k.K(np.array(x).reshape((1,self._n1)),self._X[i:i+1,:])[:,0]
        A=self.An(n,self._X[0:n,:])
        L=np.linalg.cholesky(A)
        temp3=linalg.solve_triangular(L,sigmaVec,lower=True)
        temp2=np.dot(temp3.T,temp3)
        temp2=temp-temp2
        if grad==False:
            return temp2
        else:
            gradi=np.zeros(self._n1)
            x=np.array(x).reshape((1,self._n1))
            gradX=np.zeros((n,self._n1))
            for j in xrange(self._n1):
                for i in xrange(n):
                    gradX[i,j]=self._k.K(x,self._X[i,:].reshape((1,self._n1)))*(2.0*self._alpha1[j]*(x[0,j]-self._X[i,j]))
                temp5=linalg.solve_triangular(L,gradX[:,j].T,lower=True)
                gradi[j]=np.dot(temp5.T,temp3)
            gradVar=-2.0*gradi
            return temp2,gradVar
    
    def PI (self,xNew,X,n,y,grad=False):
        n=n-1
        vec=np.zeros(n)
        for i in xrange(n):
            vec[i]=self.muN(X[i,:],n)
        maxObs=np.max(vec)
        std=np.sqrt(self.varN(xNew,n))
        muNew,gradMu=self.muN(xNew,n,grad=True)
        Z=(muNew-maxObs)/std
        temp1=norm.cdf(Z)
        if grad==False:
            return temp1
        var,gradVar=self.varN(xNew,n,grad=True)

        gradstd=.5*gradVar/std
        gradZ=((std*gradMu)-(muNew-maxObs)*gradstd)/var
        temp10=norm.pdf(Z)*gradZ
        return temp1,temp10
        
    
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
    def updateY (self,n,X):
        mean=[0,0]
        cov=[[1,1],[1,2]]
        L=np.linalg.cholesky(cov)
        t=np.zeros(self._M)
        for i in xrange(self._M):
            x=np.random.normal(0,1,(1,1))
            y=np.random.normal(0,1,(1,1))
            z=np.concatenate((x,y),1)
            z=np.dot(L,z.T).reshape(2)
            t[i]=self._f(X[n,:],z[0],z[1])
        return np.mean(t),float(np.var(t))/self._M
        
    
    
    #(self,n,x,w1,tol=1e-8,maxit=1000,maxtry=25)
    def steepestAscent (self,n,tol=1e-8,maxit=1000,maxtry=25):
        tolMet=False
        iter=0
        repeat=3 #multi-Start
        val=np.zeros((repeat,1+self._n1))
        for i in xrange(repeat):
            x=np.array(np.random.uniform(self._c,self._d)).reshape((1,self._n1))
            tolMet=False
            iter=0
            X=x
            g1=-100
            while tolMet==False:

                
                iter=iter+1
                oldEval=g1
                oldPoint=X
                g1,g2=self.PI(X[0,0:self._n1].reshape((1,self._n1)),self._X,n,self._y,grad=True)

                while g1==0:
                    print "mm"
                    if iter==1:
                        x=np.array(np.random.uniform(self._c,self._d)).reshape((1,self._n1))
                        X=x
                        g1,g2=self.PI(X[0,0:self._n1].reshape((1,self._n1)),self._X,n,self._y,grad=True)
                    else:
                        val[i,0]=oldEval
                        val[i,1:]=oldPoint
                        tolMet=True
                        break
                    
                if (tolMet==True):
                    break
                
                def fns(alpha,X_=oldPoint,g2=g2):
                    tmp=X_+alpha*g2
                    x_=tmp[0,0:self._n1]
                    return self.PI(x_,self._X,n,self._y,grad=False)
                lineSearch=self.goldenSectionLineSearch(fns,tol,maxtry,X,g2)
                X=X+lineSearch*g2
                if max(np.reshape(abs(X-oldPoint),X.shape[1]))<tol or iter > maxit:
                    g1=self.PI(X[0,0:self._n1].reshape((1,self._n1)),self._X,n,self._y,grad=False)
                    val[i,0]=g1
                    val[i,1:]=X
                    tolMet=True
        MAX=np.argmax(val[:,0])
     #   print max (val[:,0])
     #   if max (val[:,0])<1e-6:
     #       print "interesante"
     #       return -1,-1,-1,-1
     #   print "ya"
        X=np.array(val[MAX,1:]).reshape((1,self._n1))
        self._X=np.vstack([self._X,X[0,0:self._n1]])
        temp34=self.updateY(self._n,self._X)
        self._varianceObservations=np.append(self._varianceObservations,temp34[1])
        self._y=np.vstack([self._y,temp34[0]])
        self._n=self._n+1
        return X[0,0:self._n1],self._y[n-1,0],n,max(val[:,0])


        
        

        
    def graphAn (self,n):
        ###plot of muN
        points=self._points[:,0]
        m=self._points.shape[0]
        muN=np.zeros(m)
        VOI=np.zeros(m)
        variance=np.zeros(m)
        for i in xrange(m):
            muN[i]=self.muN(points[i],n)
            VOI[i]=self.PI(points[i],self._X[0:n-1,:],n,self._y[0:n-1,:],False)
            variance[i]=self.varN(points[i],n)
            
        plt.plot(points,VOI)
        plt.xlabel('x',fontsize=26)
        plt.ylabel('VOI',fontsize=24)
        plt.savefig('%d'%n+"informationPIStandardGaussian.pdf")
     #   plt.title('VOI for n=%d'%n)
        plt.close("all")
        
        plt.plot(points,muN,'--')
        confidence=muN+1.96*(variance**.5)
        confidence2=muN-1.96*(variance**.5)
        plt.plot(self._points,-(self._points**2),color='b',label="G(x)")
        plt.plot(self._points,muN,'--',color='g',label='$\mu_%d(x)$'%n)
        plt.plot(self._points,confidence,'--',color='r',label="95% CI")
        plt.plot(self._points,confidence2,'--',color='r')
        plt.legend()
        plt.xlabel('x',fontsize=26)
      #  plt.title('$\mu_%d(x)$'%n+' vs G(x)')
        plt.savefig('%d'%n+"PIStandardGaussian.pdf")
        plt.close("all")
        
        ####plot of VOI



        ####if old=true, this produces n more points
    def getNPoints(self,n,i1,tol=1e-8,maxit=1000,maxtry=25):
        if self._old==False:
            w=self._wStart
            y=np.zeros([0,1]) 
            for i in xrange(self._mPrior):
                temp100=self.updateY(i,self._Xprior)
                y=np.vstack([y,temp100[0]])
                self._varianceObservations=np.append(self._varianceObservations,temp100[1])
            self._X=self._Xprior
            self._y=y
            self._model = GPy.models.GPRegression(self._Xprior,y,self._k)
            self._model.optimize()
            m=self._model
            self._alpha1=np.array(0.5/(np.array(m['.*rbf'].values()[1:self._n1+1]))**2)
            self._variance0=np.array(m['.*rbf'].values()[0])
            self._muStart=np.array(m['.*var'].values()[1])
            f=open('%d'%i1+"G(xn)PIAnalytic.txt","w")
            self._n=self._mPrior
            for i in xrange(1,n):
                print i
                x1,y1,n1,maxVn=self.steepestAscent(i+self._mPrior)
                x21,val=self.SteepestAscent2(self._X[self._n-1,:])
                np.savetxt(f,-x21**2)
             #   self.graphAn(i)
                print i
                if maxVn==-1:
                    print "great approximation!"
                    return
                x=x1
            f.close()
        else:
            n1=self._n
            for i in xrange(1,n+1):
                print i
                x1,w1,y1,n1,maxVn=self.steepestAscent(n1+i)
                self.graphAn(n1+i)
                print i
                x=x1
                if (maxVn==-1):
                    print "great approximation!"
                    return 
                x=x1
                
    def optimize(self,n,i):
        self.getNPoints(n,i)
       # x,val=self.SteepestAscent2(self._X[self._n-1,:])
        return 0,0
    
    def SteepestAscent2 (self,x,tol=1e-8,maxit=1000,maxtry=25):
        n=self._n
        A=self.An(self._n,self._X)
        L=np.linalg.cholesky(A)
        y2=np.array(self._y)-self._muStart
        tolMet=False
        iter=0
        
        while tolMet==False:
            iter=iter+1
            oldPoint=x
            g1,g2=self.muN(x,n,True)
   
            def fns(alpha,x_=oldPoint,g2=g2,L=L,y2=y2):
                tmp=x_+alpha*g2
                return self.muN(tmp,n)
            lineSearch=self.goldenSectionLineSearch(fns,tol,maxtry,x,g2)
            x=x+lineSearch*g2
            if LA.norm(x-oldPoint)<tol or iter > maxit:
                tolMet=True
        return x,self.muN(x,n)

        
    def getPoints(self):
        A=self.An(self._n,self._X,self._W)
        return self._X,self._W,self._y,self._variance0,self._muStart,self._alpha1,self._alpha2, linalg.inv(A)
    


def g(x,w1,w2):
    val=(w2+8.0)/(w1+8.0)
    return -(val)*(x**2)-w1



    
    
if __name__ == '__main__':
    for i in range(1,101):
        np.random.seed(i)
        alpha1=[.5]
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
        M=100
        alg=SBO(f,alpha1,sigma0,sigma,mu,muStart,n1,n2,l,a,b,w,muW2,sigmaW2,M)
        opt=alg.optimize(9,i)
    print opt[0]
    
    print "optimal value"
    print opt[1]
 #   print alg._X
 #   print alg._W
 #   print alg._y
    


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

