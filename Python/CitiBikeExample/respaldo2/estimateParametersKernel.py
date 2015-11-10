#!/usr/bin/env python

###Estimate the parameters of the kernel using observations

import numpy as np
import GPy
import multiprocessing

###This works only for SBO algorithm
##wStar parameter of the kernel
##functionY is a function that estimates the noisy observations Y
##n1 is the dimension of X, and n2 is the dimension of W
##c,d, are the bounds of the compact set
##M samples to estimate Y
##f is the original function, f(x,w,z)
def estimate(wStart,numberSamples,functionY,n1,n2,c,d,kernel,M,f,lambdaParameters,setParameters,points,poisson=True):
    sampleSize=points.shape[0]
    randomIndexes=np.random.random_integers(0,sampleSize-1,numberSamples)
    w=wStart
    y=np.zeros([0,1])
    varianceObservations=np.zeros(0)
    xPrior=np.zeros((numberSamples,n1))
    wPrior=np.zeros((numberSamples,n2))
    xPrior=points[randomIndexes,:]
    if poisson==True:
        for j2 in xrange(n2):
            wPrior[:,j2]=np.random.random_integers(300,400,size=numberSamples)
    pool=multiprocessing.Pool()
    results_async = [pool.apply_async(functionY,args=(xPrior[i,:],wPrior[i,:],M,f,lambdaParameters,setParameters,)) for i in range(numberSamples)]
    output = [p.get() for p in results_async]
    for i in xrange(numberSamples):
        temp=output[i]
        varianceObservations=np.append(varianceObservations,temp[1])
        y=np.vstack([y,temp[0]])
    f=open("ValuesForhyperparametersX_W.txt",'w')
    np.savetxt(f,np.concatenate((xPrior,wPrior),1))
    f.close()
    f=open("ValuesForhyperparametersY.txt",'w')
    np.savetxt(f,y)
    f.close()
    f=open("ValuesForhyperparametersSigmaConditional.txt",'w')
    np.savetxt(f,varianceObservations)
    f.close()
    
    
    
    
