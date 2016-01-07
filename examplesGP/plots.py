#!/usr/bin/env python

import numpy as np
from math import *
import os

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

repetitions=300



samplesIteration=[1,2,4,8,16]
numberIterations=100
numberPrior=30
A=[2,4,8,16]
varianceb=[1.0/(2.0**k) for k in xrange(5)]

numberdifIteration=len(samplesIteration)
numberAs=len(A)
numberofvariance=len(varianceb)

directory="results"

meansSBO=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
varSBO=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
contSBO=np.zeros((numberofvariance,numberdifIteration,numberAs,1))

meansKG=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
varKG=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
contKG=np.zeros((numberofvariance,numberdifIteration,numberAs,1))

for r in xrange(numberofvariance):
    betah=varianceb[r]
    for s in xrange(numberdifIteration):
        n=samplesIteration[s]
        x=np.linspace(0,n*numberIterations,numberIterations+1)
        for j in xrange(numberAs):
            Aparam=1.0/(A[j]*n)
            
            y=np.zeros([0,numberIterations+1])
            for i in range(1,repetitions+1):
                try:
                    "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%nTemp3+".txt"
                    temp=np.loadtxt(os.path.join(directory,
                                                 "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+"Results"+
                                                 '%d'%n+"AveragingSamples"+'%d'%+numberPrior+"TrainingPoints"+".txt",
                                                 "SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
                    if len(temp)>=(numberIterations+1)*2 :
                        temp1=np.zeros(numberIterations+1)	    
                        for j in range(numberIterations+1):
                            temp1[j]=temp[2*j]
                        y=np.vstack((y,temp1))
                        contSBO[r,s,j,0]+=1
                except:
                    continue

            
            for i in xrange(numberIterations+1):
                meansKG[r,s,j,i]=np.mean(y[:,i])
                varKG[r,s,j,i]=np.var(y[:,i])
                
            y=np.zeros([0,numberIterations+1])
            for i in range(1,repetitions+1):
                try:
                    "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%nTemp3+".txt"
                    temp=np.loadtxt(os.path.join(directory,
                                                 "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+"Results"+
                                                 '%d'%n+"AveragingSamples"+'%d'%+numberPrior+"TrainingPoints"+".txt",
                                                 "SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
                    if len(temp)>=(numberIterations+1)*2 :
                        temp1=np.zeros(numberIterations+1)	    
                        for j in range(numberIterations+1):
                            temp1[j]=temp[2*j]
                        y=np.vstack((y,temp1))
                        contKG[r,s,j,0]+=1
                except:
                    continue

            print y
            for i in xrange(numberIterations+1):
                meansKG[r,s,j,i]=np.mean(y[:,i])
                varKG[r,s,j,i]=np.var(y[:,i])
        
            
            
            
varianceB=np.tile(varianceb,numberIterations+1)

BETA, N = np.meshgrid(varianceB, samplesIteration)

Z=np.zeros(BETA.shape)


indexofA=1

for i in xrange(BETA.shape[0]):
    for j in xrange(BETA.shape[1]):
        Z[i,j]=meansKG[j%numberofvariance,i,indexofA,j/(numberofvariance)]

print Z
print BETA
print N


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(BETA, N, Z, rstride=8, cstride=8, alpha=0.3)

cset = ax.contour(BETA, N, Z, zdir='z',  cmap=cm.coolwarm)


plt.savefig("contourPlot.pdf")
plt.close("all")
#cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)







#x=np.linspace(0,samplesIteration*numberIterations,numberIterations+1)
#y=np.zeros([0,numberIterations+1])



#plt.plot(x,means,color='g',linewidth=2.0,label='SBO')
#confidence=means+1.96*(var**.5)/np.sqrt(cont)
#plt.plot(x,confidence,'--',color='g',label="95% CI")
#confidence=means-1.96*(var**.5)/np.sqrt(cont)
#plt.plot(x,confidence,'--',color='g')






