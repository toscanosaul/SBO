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
from scipy.stats import poisson
from simulationPoissonProcess import *
from multiStartGradientForSBO import steepestAscent,SteepestAscent_aN
from VOI_SBO_logTrick import Vn,aN_grad
from estimateParametersKernel import estimate

logFactorial=[np.sum([log(i) for i in range(1,j+1)]) for j in range(1,501)]
logFactorial.insert(1,0)
logFactorial=np.array(logFactorial)

#Computes log*sum(exp(x)) for a vector x, but in numerically careful way
def logSumExp(x):
    xmax=np.max(np.abs(x))
    y=xmax+np.log(np.sum(np.exp(x-xmax)))
    return y

####WRITE THE DOCUMENTATION

#f is the function to optimize
#alpha1,alpha2,sigma0 are the parameters of the covariance matrix of the gaussian process
#alpha1,alpha2 should be vectors
#muStart is the mean of the gaussian process (assumed is constant)
#lamb vector parameters of the poisson processes
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

###A is the collection of sets of the configuration of the bike stations
###funtionY is used to estimate the noisy observations
###user should set self._Xprior, self._wPrior, to estimate the prior
class SBO:
    def __init__(self,f,alpha1,alpha2,sigma0,lamb,muStart,n1,n2,l,c,d,wStart,M,h,h2,A,functionY,mPrior=2,old=False,Xold=-1,Wold=-1,yold=-1):
        self._k=GPy.kern.RBF(input_dim=n1+n2, variance=sigma0**2, lengthscale=np.concatenate(((0.5/np.array(alpha1))**(.5),(0.5/np.array(alpha2))**(.5))),ARD=True)+GPy.kern.White(n1+n2,0.1)
        self._lambda=np.array(lamb)
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
        self._points=np.loadtxt("pointsPoisson.txt")
      #  self._points=grid(c,d,l)
        self._functionY=functionY
        self._wStart=wStart
        self._old=old
        self._mPrior=mPrior
        self._noise=0.1
        self._simulateW1=h
        self._simulateW2=h2
        self._B=np.zeros((self._points.shape[0],0)) ##save evaluations of B(i,points)
        self._set=A
        self._varianceObservations=np.zeros(0)
        ###old is true if we are including data to the model
        if old==False:
            self._X=np.zeros([0,n1]) #points used
            self._W=np.zeros([0,n2]) #points used
            self._y=np.zeros([0,1]) #evaluations of the function
            self._n=0 #stage of the algorithm
            self._Xprior=np.zeros((mPrior,self._n1))
            self._Wprior=np.zeros((mPrior,self._n1))
            for i in xrange(self._n1):
                self._Xprior[:,i]=np.random.random_integers(c[i],d[i],mPrior)
                self._Wprior[:,i]=np.random.random_integers(1000,size=mPrior)
            self._mPrior=mPrior
        else:
            self._X=Xold
            self._W=Wold
            self._y=yold
            self._n=yold.shape[0]

        self._M=M

        
    def kernel(self):
        return self._k
    
    def evaluatef(self,x,w1,w2):
        return self._f(x,w1,w2)
    
    
    #compute B(x,i). X=X[i,:],W[i,:]. x is a matrix of dimensions nxm where m is the dimension of an element of x.
    ##This function is used by update. Remember that B(x,i)=integral_(sigma(x,w,x_i,w_i))dp(w)
    ##n1 is the dimension of X[i,:]
    ##n2 is the dimension of W[i,:]
    ##variance0 is the parameter of kernel (the variance)
    ##alpha1 parameter of the kernel. It is related to x
    ##alpha2 parameter of the kernel. It is related to w
    ##poisson is true if we want to use it for the citibike problem
    ##lambdaParameter are the parameters of the Poisson processes for W. It can be empty if poisson=False
    def Bparameters(self,x,X,W,n1,n2,variance0,alpha1,alpha2,lambdaParameter,poisson2=True):
        x=np.array(x).reshape((x.shape[0],n1))
        results=np.zeros(x.shape[0])
        parameterLamb=np.zeros(n2)
        for j in xrange(n2):
            parameterLamb[j]=np.sum(lambdaParameter[j])
        quantil=int(poisson.ppf(.99999999,max(parameterLamb)))
        expec=np.array([i for i in xrange(quantil)])
        if poisson2==True:
            for i in xrange(x.shape[0]):
                temp=log(variance0)+logSumExp(-alpha1*((x[i,:]-X)**2) )
                temp2=0
                for j in xrange(n2):
                    temp2=temp2+logSumExp(-alpha2[j]*((expec-W[j])**2)-parameterLamb[j]+expec*log(parameterLamb[j])-logFactorial[expec])
                results[i]=temp2+temp
    #    print np.exp(results)
        return np.exp(results)

    ##X,W are the past observations
    ##An is the matrix of the paper
    def An(self,n,X,W):
        An=self._k.K(np.concatenate((X,W),1))+np.diag(self._varianceObservations[0:n])
        return An

        #only if X and W have been updated
        #Ainv is inv(An)
        #y2 is y-muStart
        ##past are all the observations
    def getMu(self, n,x,w,L,y2,X,W):
        past=np.concatenate((X,W),1)
        new=np.concatenate((x,w),1)
        bn=self._k.K(new,past)
        temp1=linalg.solve_triangular(L,np.array(y2),lower=True)
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
        if np.array_equal(new,new1):
            temp1=self._k.K(new)-np.dot(temp2.T,temp1)
        else:
            temp1=self._k.K(new,new1)-np.dot(temp2.T,temp1)
      #  print "prueba de simetria"
        #print Ainv-Ainv.T
       # print "up"
       # print self._k.K(new1,new)-np.dot(np.dot(self._k.K(new1,pas),Ainv),self._k.K(new,pas).T)
       # print temp1
       # return
        return temp1
        

    ##Using M samples for W2
    ###tal vez no usare self._M
    def updateY (self,n,X,W):
        simulations=np.zeros(self._M)
        for i in xrange(self._M):
            simulations[i]=self._f(4.0,W[n,:],X[n,:],4,self._lambda,self._set,"2014-05")
        return np.mean(simulations),float(np.var(simulations))/self._M

    


        



    def graphAn (self,n):
        A=self.An(n,self._X[0:n,:],self._W[0:n,:])
       # Ainv=linalg.inv(A)
        L=np.linalg.cholesky(A)
        y2=np.array(self._y[0:n])-self._muStart
        m=self._points.shape[0]
        z=np.zeros(m)
        for i in xrange(m):
            z[i]=self.aN_grad(self._points[i,:],L,y2,gradient=False)
   #     a,b,gamma,BN,L,B=self.update(self._points,n,self._y,self._X,self._W,xNew,wNew)
        
        fig=plt.figure()
        plt.plot(self._points,-(self._points**2))
        plt.plot(self._points,z,'--')
       # plt.plot(self._points,z+1.96*b,'--',color='r')
       # plt.plot(self._points,z-1.96*b,'--',color='r')
        plt.title("Graph of a_n vs real function")
        plt.savefig('%d'%n+"a_n")
        plt.close(fig)
       
        ####if old=true, this produces n more points
    def getNPoints(self,n,tol=1e-8,maxit=1000,maxtry=25):
        if self._old==False:
           # estimate(self._wStart,self._mPrior,self._functionY,self._n1,self._n2,self._c,self._d,self._k,100,self._f,self._lambda,self._set,self._points,poisson=True)
        #    print "ya"
            X2=np.loadtxt("ValuesForhyperparametersX_W.txt")
            y2=np.loadtxt("ValuesForhyperparametersY.txt")
            conditionalSigma=np.loadtxt("ValuesForhyperparametersSigmaConditional.txt")
            self._varianceObservations=conditionalSigma
            model = GPy.models.GPRegression(X2,y2.reshape((self._mPrior,1)),self._k)
            model.optimize()
            m=model
            n1=self._n1
            n2=self._n2
            alpha1=np.array(0.5/(np.array(m['.*rbf'].values()[1:n1+1]))**2)
            alpha2=np.array(0.5/(np.array(m['.*rbf'].values()[n1+1:n1+1+n2]))**2)
            variance0=np.array(m['.*rbf'].values()[0])
            muStart=np.array(m['.*var'].values()[1])
            self._alpha1=alpha1
            self._alpha2=alpha2
            self._variance0=variance0
            self._muStart=muStart
            f=open("hyperparametersPoisson.txt",'w')

            np.savetxt(f,self._alpha1)
            np.savetxt(f,self._alpha2)
            np.savetxt(f,np.array([self._variance0]))
            np.savetxt(f,np.array([self._muStart]))
            f.close()
            print "yea"
            ########change i to 1
            f=open("oldXpoisson3.txt",'w')
            g=open("oldWpoisson3.txt",'w')
            f2=open("oldYpoisson3.txt",'w')
            f3=open("samplesVSvaluePointsPoisson3.txt","w")
            f4=open("samplesVSvaluePoisson3.txt","w")
            f.close()
            g.close()
            f2.close()
            f3.close()
            f4.close()
            self._n=len(y2)
            self._y=y2.reshape((self._n,1))
            self._X=X2[:,0:self._n1]
            self._W=X2[:,self._n1:self._n1+self._n2]
            m=len(y2)
            for i in xrange(m):
                newB=self.Bparameters(self._points,self._X[i,:],self._W[i,:],self._n1,self._n2,self._variance0,self._alpha1,self._alpha2,self._lambda)
                self._B=np.insert(self._B,i,newB,axis=1)
           # print self._B
            f=open("oldBpoisson3.txt",'w')
            np.savetxt(f,self._B)
            f.close()
            m=len(y2)
            for i in xrange(1,n+1):
                print i
                x1,w1=steepestAscent(i+m,self._n1,self._n2,self._c,self._d,Vn,self._X,self._W,self._y,self._muStart,self._k,self._variance0,self._alpha1,self._alpha2,self._lambda,self._points,self._varianceObservations,self._B)
                self._X=np.vstack([self._X,x1])
                self._W=np.vstack([self._W,w1])
                newB=self.Bparameters(self._points,x1,w1,self._n1,self._n2,self._variance0,self._alpha1,self._alpha2,self._lambda)
                self._B=np.insert(self._B,i+m-1,newB,axis=1)
                ###change for a funtion that updates y
                temp34=self.updateY(self._n,self._X,self._W)
                self._y=np.vstack([self._y,temp34[0]])
                self._n=self._n+1
                self._varianceObservations=np.append(self._varianceObservations,temp34[1])
                x3,val=SteepestAscent_aN (i+m,self._n1,self._n2,self._X,self._W,aN_grad,self._c,self._d,self._y,self._muStart,self._variance0,self._alpha1,self._alpha2,self._lambda,self._k,self._varianceObservations,self._B)
                val2=np.zeros(1000)
                for i in xrange(1000):
                    w4=np.zeros(n2)
                    for j2 in xrange(n2):
                        w4[j2]=np.random.poisson(np.sum(self._lambda[j2]), 1)
                    val2[i]=self._f(4.0,w4,x3,4,self._lambda,self._set,"2014-05")
                print "value"
                print np.mean(val2)
                print "variance"
                print np.var(val2)/1000
                print "point"
                print x3
                temp5=np.mean(val2)
                temp6=np.var(val2)/1000
                with open("samplesVSvaluePoisson3.txt", "a") as f4:
                    np.savetxt(f4,np.array(temp5).reshape((1,1)))
                    np.savetxt(f4,np.array(temp6).reshape((1,1)))
                f4.close()
                with open("samplesVSvaluePointsPoisson3.txt", "a") as f4:
                    np.savetxt(f4,x3)
                f4.close()
                with open("oldYpoisson3.txt.txt", "a") as f:
                    np.savetxt(f,self._X)
                f.close()
                with open("oldXpoisson3.txt.txt", "a") as f:
                    np.savetxt(f,self._X)
                f.close()
                
                with open("oldWpoisson3.txt","a") as f:
                    np.savetxt(f,self._W)
                f.close()
                
                with open("oldBpoisson3.txt", "a") as f4:
                    np.savetxt(f4,self._B)
                f4.close()


                print i
                w=w1
                x=x1
                f2.close()
                f.close()
                g.close()
                f3.close()
                f4.close()
        else:
            n1=self._n
            for i in xrange(1,n+1):
                print i
                x1,w1=steepestAscent(n1+i)
                print i
                w=w1
                x=x1
                w=w1
                x=x1


    def optimize(self,n):
        self.getNPoints(n)
        x,val=SteepestAscent_aN (n+self._M,self._n1,self._n2,self._X,self._W,aN_grad,self._c,self._d,self._y,self._muStart,self._variance0,self._alpha1,self._alpha2,self._lambda,self._k,self._varianceObservations,self._B)
        return x,val
    

        
    def getPoints(self):
        A=self.An(self._n,self._X,self._W)
        return self._X,self._W,self._y,self._variance0,self._muStart,self._alpha1,self._alpha2, linalg.inv(A)
    


def g(x,w1,w2):
    temp=LA.norm(w1)**2
    temp2=LA.norm(w2)**2
    if temp!=0:
        val=temp2/temp
    else:
        val=Infinity
    return -(val)*(LA.norm(x)**2)

def functionY(x,w,M,f,lambdaParameters,setParameter):
    simulations=np.zeros(M)
    for i in xrange(M):
        simulations[i]=f(4.0,w,x,4,lambdaParameters,setParameter,"2014-05")
    return np.mean(simulations),float(np.var(simulations))/M

###n is the number of samples
##output is a nxm matrix where m is the dimension of one element simulated
def h(n):
    return np.random.multivariate_normal((0,0),np.ones((2,2)),n)

def h2(n,w1):
    return np.array(np.random.multivariate_normal(w1,np.ones((2,2)),n))


    
if __name__ == '__main__':
    np.random.seed(10)
    alpha1=.5*np.ones(4)
    alpha2=.5*np.ones(4)
    sigma0=10
    muStart=np.zeros(4)
    f=unhappyPeople
    a=100*np.ones(4)
    b=600*np.ones(4)
    n1=4
    n2=4
    l=5
    w=np.ones(4)
    M=100
    fil="2014-05PoissonParameters.txt"
    nSets=4
    A,lamb=generateSets(nSets,fil)
    alg=SBO(f,alpha1,alpha2,sigma0,lamb,muStart,n1,n2,l,a,b,w,M,h,h2,A,functionY)
    opt=alg.optimize(100)
    print opt[0]
    
    print "optimal value"
    print opt[1]
    #####Read Files
    X=np.loadtxt("oldX.txt",ndmin=2)
    W=np.loadtxt("oldW.txt",ndmin=2)
    Y=np.loadtxt("oldY.txt",ndmin=2)
    T=np.loadtxt("hyperparameters.txt",ndmin=2)
    alpha1=T[0,:]
    alpha2=T[1,:]
    sigma0=T[2,:][0]
    muStart=T[3,:][0]
