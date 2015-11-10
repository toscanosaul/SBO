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
import multiprocessing
from multiprocessing.pool import ApplyResult
from VOIsboGaussian import aN_grad,Vn
from SBOGaussianOptima import steepestAscent,SteepestAscent_aN




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

####kernel should receive only matrices as input.
####B(x,X,W) is a function that computes B(x,i). X=X[i,:],W[i,:]. x is a matrix where each row is a point to be evaluated

####checar multidimensional!!! Poisson already works 
class SBO:
    def __init__(self,f,graphAn,kernel,B,alpha1,alpha2,sigma0,sigma,mu,muStart,n1,n2,l,c,d,wStart,muW2,sigmaW2,M,mPrior=5,old=False,Xold=-1,Wold=-1,yold=-1):
        self._k=kernel
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
        self.B=B
        self.graphAn=graphAn
        ###old is true if we are including data to the model
        if old==False:
            self._X=np.zeros([0,n1]) #points used
            self._W=np.zeros([0,n2]) #points used
            self._y=np.zeros([0,1]) #evaluations of the function
            self._n=0 #stage of the algorithm
            self._Xprior=np.random.uniform(c,d,mPrior).reshape((self._mPrior,self._n1)) ##to estimate the kernel
            self._wPrior=np.random.normal(0,1,mPrior).reshape((self._mPrior,self._n2))
            self._mPrior=mPrior
        else:
            self._X=Xold
            self._W=Wold
            self._y=yold
            self._n=yold.shape[0]
        self._muW2=muW2
        self._sigmaW2=sigmaW2
        self._M=M
 

    ##X,W are the past observations
    ##An is the matrix of the paper
    def An(self,n,X,W,varianceObservations):
        An=self._k.K(np.concatenate((X,W),1))+np.diag(self._varianceObservations[0:n])
        return An
    
    
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
    

    ###only for n>0
    def VarF(self,n,x):
        X=self._X
        W=self._W
        if n==0:
            return 
        x=np.array(x)
        m=x.shape[0]
        A=self.An(n,X,W,self._varianceObservations)
        L=np.linalg.cholesky(A)
        B=np.zeros([m,n])
        for i in xrange(n):
            B[:,i]=self.B(x,X[i,:],W[i,:])
        temp2=linalg.solve_triangular(L,B.T,lower=True)
        return self._variance0*.5*(1.0/((.25+self._alpha2)**.5))-np.dot(temp2.T,temp2)
    

    def updateY (self,n,X,W):
        t=np.array(np.random.normal(W[n,:],self._sigmaW2,self._M))
        t=self._f(X[n,:],W[n,:],t)
        return np.mean(t),float(np.var(t))/self._M
    


    ###this function is very particular to this example
    def Fvalues(self,x,w1):
        return -x**2-w1
        
    
    def muN(self,C,D,n):
        X=self._X
        W=self._W
        y=self._y[:,0]
        m=C.shape[0]
        A=self.An(n,X,W,self._varianceObservations)
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
        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 15}
        A=self.An(n,self._X[0:n,:],self._W[0:n,:],self._varianceObservations)
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
        plt.xlabel('x',fontsize=26)
        plt.ylabel('w',fontsize=24)
        plt.savefig('%d'%n+"Contours of estimation of F.pdf")
        plt.close(fig)
        fig=plt.figure()
        CS=plt.contour(X,W,Z)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel('x',fontsize=26)
        plt.ylabel('w',fontsize=24)
        plt.savefig('%d'%n+"F.pdf")
        plt.close(fig)
        
        #####contors for VOI
        VOI=np.zeros((m,m))
        for i in xrange(m):
            for j in xrange(m):
                VOI[i,j]=Vn(np.array([[X[i,j]]]),np.array([[W[i,j]]]),self._X[0:n-1,:],self._W[0:n-1,:],n,self._y[0:n-1,:],self._muStart,self._k,self._variance0,self._alpha1,self._alpha2,self._n1,self._n2,self._points,self._varianceObservations,False)
        fig=plt.figure()
        CS=plt.contour(X,W,VOI)
        plt.clabel(CS, inline=1, fontsize=10)
        n1=n-1
        plt.xlabel('x',fontsize=26)
        plt.ylabel('w',fontsize=24)
        plt.savefig('%d'%n+"VOI.pdf")
        plt.close(fig)
        ######
        for i in xrange(m):
            z[i]=aN_grad(self._points[i,:],L,y2,self._X,self._W,self._n1,self._n2,self._variance0,self._alpha1,self._alpha2,self._muStart,n,gradient=False)
            var[i]=self.VarF(n,self._points[i,:])
        
        fig=plt.figure()
        plt.plot(self._points,-(self._points**2),label="G(x)")
        plt.plot(self._points,z,'--',label='$a_%d(x)$'%n)
        confidence=z+1.96*(var**.5)
        plt.plot(self._points,confidence,'--',color='r',label="95% CI")
        confidence2=z-1.96*(var**.5)
        plt.plot(self._points,confidence2,'--',color='r')
        plt.legend()
        plt.xlabel('x',fontsize=26)
        plt.savefig('%d'%n+"a_n.pdf")
        plt.close(fig)
        

        ####if old=true, this produces n more points
    def getNPoints(self,n,i1,tol=1e-8,maxit=1000,maxtry=25):
        if self._old==False:
            
            w=self._wStart
            y=np.zeros([0,1])
            for i in xrange(self._mPrior):
                temp100=self.updateY(i,self._Xprior,self._wPrior)
                y=np.vstack([y,temp100[0]])
                self._varianceObservations=np.append(self._varianceObservations,temp100[1])
            self._X=self._Xprior
            self._W=self._wPrior
            self._y=y
            self._n=self._mPrior
            self._model = GPy.models.GPRegression(np.concatenate((self._Xprior,self._wPrior),1),y,self._k)
            self._model.optimize()
            print (self._model)
            self._model.optimize_restarts(num_restarts = 10)
            m=self._model
            print (m)
            self._alpha1=np.array(0.5/(np.array(m['.*rbf'].values()[1:self._n1+1]))**2)
            self._alpha2=np.array(0.5/(np.array(m['.*rbf'].values()[self._n1+1:self._n1+1+self._n2]))**2)
            self._variance0=np.array(m['.*rbf'].values()[0])
            self._muStart=np.array(m['.*var'].values()[1])
            print self._alpha1
            print self._alpha2
            f=open("hyperparameters.txt",'w')
            np.savetxt(f,self._alpha1)
            np.savetxt(f,self._alpha2)
            np.savetxt(f,np.array([self._variance0]))
            np.savetxt(f,np.array([self._muStart]))
            f.close()
            x21,val=SteepestAscent_aN(self._mPrior,self._n1,self._n2,self._X,self._W,aN_grad,self._c,self._d,self._y,self._muStart,self._variance0,self._alpha1,self._alpha2,self._k,self._varianceObservations)
            f=open("2NewResults"+'%d'%self._M+"samples5Prior/"+'%d'%i1+"G(xn)SBOAnalytic"+'%d'%self._M+".txt","w")
            np.savetxt(f,-x21**2)
            for i in xrange(1,n+1):
                x1=steepestAscent(i+self._mPrior,self._n1,self._n2,self._c,self._d,Vn,self._X,self._W,self._y,self._muStart,self._k,self._variance0,self._alpha1,self._alpha2,self._points,self._varianceObservations)
                self._X=np.vstack([self._X,x1[0,0:self._n1]])
                self._W=np.vstack([self._W,x1[0,self._n1:self._n1+self._n2]])
                temp34=self.updateY(self._n,self._X,self._W)
                self._y=np.vstack([self._y,temp34[0]])
                self._varianceObservations=np.append(self._varianceObservations,temp34[1])
                x21,val=SteepestAscent_aN(i+self._mPrior,self._n1,self._n2,self._X,self._W,aN_grad,self._c,self._d,self._y,self._muStart,self._variance0,self._alpha1,self._alpha2,self._k,self._varianceObservations)
                self._n=self._n+1
                print x21
                np.savetxt(f,-x21**2)
                self.graphAn(i+self._mPrior)
                print i
            f.close()
        else:
            n1=self._n
            for i in xrange(1,n+1):
                print i
                x1,w1,y1,n1,maxVn=self.steepestAscent(n1+i)
                ##checar multidimensional
                
                #self._points=np.delete(self._points,np.where(self._points==x1[0]))
                self.graphAn(n1+i)
                print i
                w=w1
                x=x1
                if (maxVn==-1):
                    print "great approximation!"
                    return 
                w=w1
                x=x1
 


    def optimize(self,n,i):
        self.getNPoints(n,i)
      #  x,val=self.SteepestAscent2(self._X[self._n-1,:])
        return 0,0
    



###Objective function
def g(x,w1,w2):
    val=(w2)/(w1)
    return -(val)*(x**2)-w1

####Function F
def Fvalues(x,w1):
    return -x**2-w1

###Plots of F,F_{n},a_{n},g
###Xhist,y are past observations. Xhist[0:n1,:] x observatinons. Xhist[n1:n2,:] w observations
###varianceObservations
###n1 dim of x, n2 dim of w
###muStart mean of the Gaussian process (constant)
###points are the points considered to estimate the VOI
###a,b limits where the points are
###A=self.An(n,X[0:n,:],W[0:n,:],varianceObservations), before
def graphAn(n,Xhist,y,varianceObservations,Fvalues,Vn,aN_grad,VarF,A,muN,n1,n2,muStart,points,alpha1,alpha2,variance0,a=-3,b=3):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}
    X=Xhist[:,0:n1]
    W=Whist[:,n1:n2+n1]
    L=np.linalg.cholesky(A)
    y2=np.array(y[0:n])-muStart
    m=points.shape[0]
    z=np.zeros(m)
    var=np.zeros(m)
    ###countours for F
    w1=np.linspace(a,b,m)
    points=points[:,0]
    X2,W2=np.meshgrid(points,w1)
    Z=Fvalues(X2,W2)
    Z2=muN(X2,W2,n)
    fig=plt.figure()
    CS=plt.contour(X2,W2,Z2)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('x',fontsize=26)
    plt.ylabel('w',fontsize=24)
    plt.savefig('%d'%n+"Contours of estimation of F.pdf")
    plt.close(fig)
    fig=plt.figure()
    CS=plt.contour(X2,W2,Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel('x',fontsize=26)
    plt.ylabel('w',fontsize=24)
    plt.savefig('%d'%n+"F.pdf")
    plt.close(fig)
    
    #####contors for VOI
    VOI=np.zeros((m,m))
    for i in xrange(m):
        for j in xrange(m):
            VOI[i,j]=Vn(np.array([[X2[i,j]]]),np.array([[W2[i,j]]]),X[0:n-1,:],W[0:n-1,:],n,y[0:n-1,:],muStart,k,variance0,alpha1,alpha2,n1,n2,points,varianceObservations,False)
    fig=plt.figure()
    CS=plt.contour(X2,W2,VOI)
    plt.clabel(CS, inline=1, fontsize=10)
    n1=n-1
    plt.xlabel('x',fontsize=26)
    plt.ylabel('w',fontsize=24)
    plt.savefig('%d'%n+"VOI.pdf")
    plt.close(fig)
    ######
    for i in xrange(m):
        z[i]=aN_grad(points[i,:],L,y2,X,W,n1,n2,variance0,alpha1,alpha2,muStart,n,gradient=False)
        var[i]=VarF(n,points[i,:])
    
    fig=plt.figure()
    plt.plot(points,-(points**2),label="G(x)")
    plt.plot(points,z,'--',label='$a_%d(x)$'%n)
    confidence=z+1.96*(var**.5)
    plt.plot(points,confidence,'--',color='r',label="95% CI")
    confidence2=z-1.96*(var**.5)
    plt.plot(points,confidence2,'--',color='r')
    plt.legend()
    plt.xlabel('x',fontsize=26)
    plt.savefig('%d'%n+"a_n.pdf")
    plt.close(fig)


##i is the seed; M number of samples to estimate
##T is the number of iterations
def runProgram (i,M,T,kernel,alpha1,alpha2,n1,n2,mu,muStart,f,graphAn,w,muW2,sigmaW2):
    np.random.seed(i)
    a=[-3]
    b=[3]
    l=50
    kernel=GPy.kern.RBF(input_dim=n1+n2, variance=sigma0**2, lengthscale=np.concatenate(((0.5/np.array(alpha1))**(.5),(0.5/np.array(alpha2))**(.5))),ARD=True)+GPy.kern.White(n1+n2,0.1)
   # def graphAn2(n,Xhist,y,varianceObservations):
    #    graphAn(n,Xhist,y,varianceObservations,Vn,aN_grad,VarF,A,muN,n1,n2,muStart,points,alpha1=alpha1,alpha2=alpha2,variance0,a=-3,b=3,Fvalues=Fvalues)
  #  B()
    alg=SBO(f,graphAn2,kernel,B,alpha1,alpha2,sigma0,sigma,mu,muStart,n1,n2,l,a,b,w,muW2,sigmaW2,M)
    opt=alg.optimize(T,i)
    return 0


    
if __name__ == '__main__':
    T=14
    M=15
    n1=1
    n2=1
    alpha1=[.5]
    alpha2=[.5]
    sigma0=10
    sigma=[1]
    mu=[0]
    muStart=0
    w=np.array([1]) ##start the steepest ascent
    muW2=np.array([0])
    sigmaW2=np.array([1])
    variance=np.array(sigma)**2
    variance0=sigma**2
    def B(x,X,W,alpha1,alpha2,variance0,mu=mu,variance=variance):
        tmp=-(((mu/variance)+2.0*(alpha2)*np.array(W))**2)/(4.0*(-alpha2-(1/(2.0*variance))))
        tmp2=-alpha2*(np.array(W)**2)
        tmp3=-(mu**2)/(2.0*variance)
        tmp=np.exp(tmp+tmp2+tmp3)
        tmp=tmp*(1/(sqrt(2.0*variance)))*(1/(sqrt((1/(2.0*variance))+alpha2)))
        x=np.array(x).reshape((x.size,n1))
        tmp1=variance0*np.exp(np.sum(-alpha1*((np.array(x)-np.array(X))**2),axis=1))
        return np.prod(tmp)*tmp1
 #   pool=multiprocessing.Pool()
 #   for i in range(1,300):
 #       pool.apply_async(runProgram,args=(i,M,T,kernel,alpha1,alpha2,sigma0,sigma,n1,n2,mu,muStart,g,w,muW2,sigmaW2,graphAn,))
 #   pool.close()
 #   pool.join()
    runProgram(15,M,T,kernel,alpha1,alpha2,sigma0,sigma,n1,n2,mu,muStart,g,w,muW2,sigmaW2,graphAn)


