#!/usr/bin/env python

###See P.178 of Nocedal, Numerical Optimization

import numpy as np

class LBFGS:
    def __init__(self,m,f,gradF,dimension,alphaMax=10):
        self.dimension=dimension
        self.numberPoints=m
        self.f=f
        self.gradF=gradF
        self.s=np.zeros((m,dimension))
        self.y=np.zeros((m,dimension))
        self.points=np.zeros((m,dimension))
        self.gradient=np.zeros((m,dimension))
        self.ro=np.zeros(m)
        self.alphaMax=alphaMax
        

    ###gradient is the function to compute the gradient
    ###m is the number of points to keep
    ###k is the iteration
    def productHgradient(self,k,Hk):
        m=self.numberPoints
        ind=k%m
        q=self.gradient[ind,:]
        m=ind
        alpha=np.zeros(m)
        for i in range(m):
            indTemp=(k-1-i)%m
            alpha[m-1-i]=self.ro[indTemp]*np.dot(self.s[indTemp,:].T,q)
            q=q-alpha[m-1-i]*self.y[indTemp,:]
            
        r=np.dot(Hk,q)
        beta=0
        for i in range(m):
            indTemp=(k-m+i)%m
            beta=self.ro[indTemp]*np.dot(self.y[indTemp,:],r)
            r=r+self.s[indTemp,:]*(alpha[i]-beta)
        return r
    
    def zoom(self,a,b,g,dg,grad,c1,c2):
        conv=False
        ev3=np.dot(dg(0),grad)
        cont=0
        while conv==False:

           # print "zoom1"
            c=a+(b-a)*0.5
            ev=g(c)
           # print g(0)
            if ev>g(0)+c1*c*ev3 or ev>=g(a):
               # return c
                b=c
            else:
                ev2=np.dot(dg(c),grad)
                if np.abs(ev2)<=-c2*ev3:
                    conv=True
                    return c
                if ev2*(b-a)>=0:
                    b=a
                a=c
            cont+=1
    
    ##P.60 of Nocedal
    ###g(t)=f(x+t*grad)
    ###dg(t)=gradf(x+t*grad)
    ###grad is defined in the equation that is above
    def lineSearch(self,g,dg,grad,c1=0.3,c2=0.7):
        alphaMax=self.alphaMax
      #  print alphaMax
        alpha=np.zeros(2)
        alpha[0]=0.0
        alpha[1]=0.5*alphaMax
        i=1
        conv=False
        ev3=np.dot(dg(0),grad)
        while conv==False:
            ev=g(alpha[i])
            if g(alpha[i])>g(alpha[0])+c1*alpha[i]*ev3 or (g(alpha[i])>g(alpha[i-1]) and i>1):
                alphaAns=self.zoom(alpha[i-1],alpha[i],g,dg,grad,c1,c2)
                conv=True
                return alphaAns
            ev2=np.dot(dg(alpha[i]),grad)
            if np.abs(ev2)<=-c2*ev3:
                alphaAns=alpha[i]
                conv=True
                return alphaAns
            if ev2>=0:
                alphaAns=self.zoom(alpha[i],alpha[i-1],g,dg,grad,c1,c2)
                conv=True
                return alphaAns
            alpha=np.append(alpha,alpha[i]+(alphaMax-alpha[i])*0.5)
            i=i+1
        
    
    def BFGS(self,xStart,tol=10e-8):
        k=0
        conv=False
        Hk=np.identity(self.dimension)
        self.points[0,:]=xStart
        self.gradient[0,:]=self.gradF(xStart)
        m=self.numberPoints
        while conv==False:
            pk=-self.productHgradient(k,Hk)
            def g(t):
                return self.f(self.points[k%m,:]+t*pk)

            def dg(t):
                return self.gradF(self.points[k%m,:]+t*pk)
         #   print pk,self.points[k%m,:]
            alpha=self.lineSearch(g,dg,pk)
         #   print "bfgs"
            self.points[(k+1)%m,:]=self.points[k%m,:]+alpha*pk
            self.s[k%m,:]=self.points[(k+1)%m,:]-self.points[k%m,:]
            self.gradient[(k+1)%m,:]=self.gradF(self.points[(k+1)%m,:])
            self.y[k%m,:]=self.gradient[(k+1)%m,:]-self.gradient[(k)%m,:]
            k=k+1
            if (np.sqrt(np.sum(self.gradient[(k)%m,:]**2))<tol):
                conv=True
        return self.points[(k)%m,:]