#!/usr/bin/env python

import numpy as np
from math import *
import os

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

repetitions=500



samplesIteration=[1,2,4,8,16]
#samplesIteration=[1,2]
numberIterations=50
numberPrior=30
#A=[2,4,8,16]
A=[2,4]
varianceb=[1.0/(2.0**k) for k in xrange(5)]
#varianceb=[1.0/(2.0**k) for k in xrange(2)]

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
                    temp=np.loadtxt(os.path.join(directory,
                                                 "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+"Results"+
                                                 '%d'%n+"AveragingSamples"+'%d'%+numberPrior+"TrainingPoints",
                                                 "SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
                    if len(temp)>=(numberIterations+1)*2 :
                        temp1=np.zeros(numberIterations+1)	    
                        for l in range(numberIterations+1):
                            temp1[l]=temp[2*l]
                        y=np.vstack((y,temp1))
                        contSBO[r,s,j,0]+=1
                except:
                    continue


            for i in xrange(numberIterations+1):
                meansSBO[r,s,j,i]=np.mean(y[:,i])
                varSBO[r,s,j,i]=np.var(y[:,i])
                
            y=np.zeros([0,numberIterations+1])
            for i in range(1,repetitions+1):
                try:
                    temp=np.loadtxt(os.path.join(directory,
                                                 "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+"Results"+
                                                 '%d'%n+"AveragingSamples"+'%d'%+numberPrior+"TrainingPoints",
                                                 "SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
                    
                    if len(temp)>=(numberIterations+1)*2 :
                        temp1=np.zeros(numberIterations+1)	    
                        for l in range(numberIterations+1):
                            temp1[l]=temp[2*l]
             
                        y=np.vstack((y,temp1))
                        contKG[r,s,j,0]+=1
                except:
                    continue

            for i in xrange(numberIterations+1):
                meansKG[r,s,j,i]=np.mean(y[:,i])
                varKG[r,s,j,i]=np.var(y[:,i])
        
            
            
write=True

if write is True:
    np.savetxt("meansKG.txt",meansKG)
    np.savetxt("contKG.txt",contKG)
    np.savetxt("varKG.txt",varKG)
    np.savetxt("meansSBO.txt",meansSBO)
    np.savetxt("contSBO.txt",contSBO)

varianceB=np.tile(varianceb,numberIterations+1)

BETA, N = np.meshgrid(varianceB, samplesIteration)

Z=np.zeros(BETA.shape)
Z2=np.zeros(BETA.shape)

X=np.zeros(BETA.shape)
indexofA=1


for i in xrange(BETA.shape[0]):
    x=np.linspace(0,samplesIteration[i]*numberIterations,numberIterations+1)
    rang=ceil(100.0/samplesIteration[i])+1
    for j in xrange(BETA.shape[1]):
        temp1=(np.max(meansKG[j%numberofvariance,i,indexofA,0:rang])-np.min(meansKG[j%numberofvariance,i,indexofA,0:rang]))
        X[i,j]=x[j/(numberofvariance)]
        Z[i,j]=(meansKG[j%numberofvariance,i,indexofA,j/(numberofvariance)])/temp1
	Z[i,j]+=-(1.0/temp1)*np.min(meansKG[j%numberofvariance,i,indexofA,:])
        temp2=(np.max(meansKG[j%numberofvariance,i,indexofA,:])-np.min(meansKG[j%numberofvariance,i,indexofA,:]))
        
        Z2[i,j]=meansSBO[j%numberofvariance,i,indexofA,j/(numberofvariance)]/temp2
        Z2[i,j]+=-()



Z=Z


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(BETA, N, X, rstride=1, cstride=1, facecolors=cm.jet(Z),linewidth=0, antialiased=False, shade=False)

ax.set_xlabel('beta_h')
ax.set_ylabel('N')

m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(Z)
plt.colorbar(m)

plt.savefig("contourPlot.pdf")
plt.close("all")





#x=np.linspace(0,samplesIteration*numberIterations,numberIterations+1)
#y=np.zeros([0,numberIterations+1])



#plt.plot(x,means,color='g',linewidth=2.0,label='SBO')
#confidence=means+1.96*(var**.5)/np.sqrt(cont)
#plt.plot(x,confidence,'--',color='g',label="95% CI")
#confidence=means-1.96*(var**.5)/np.sqrt(cont)
#plt.plot(x,confidence,'--',color='g')






