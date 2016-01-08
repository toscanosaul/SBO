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

            
            
varianceB=np.tile(varianceb,numberIterations+1)

BETA, N = np.meshgrid(varianceB, samplesIteration)

Z=np.zeros(BETA.shape)
Z2=np.zeros(BETA.shape)

X=np.zeros(BETA.shape)
indexofA=1


for i in xrange(BETA.shape[0]):
    x=np.linspace(0,samplesIteration[i]*numberIterations,numberIterations+1)
    for j in xrange(BETA.shape[1]):
        X[i,j]=x[j/(numberofvariance)]

Z=np.loadtxt("Z.txt")

Z=Z


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(BETA, N, X, rstride=1, cstride=1, facecolors=cm.jet(Z),linewidth=0, antialiased=False, shade=False)

ax.set_xlabel('beta_h')
ax.set_ylabel('N')

m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(Z)
plt.colorbar(m)

plt.savefig("contourPlot2.pdf")
plt.close("all")





#x=np.linspace(0,samplesIteration*numberIterations,numberIterations+1)
#y=np.zeros([0,numberIterations+1])



#plt.plot(x,means,color='g',linewidth=2.0,label='SBO')
#confidence=means+1.96*(var**.5)/np.sqrt(cont)
#plt.plot(x,confidence,'--',color='g',label="95% CI")
#confidence=means-1.96*(var**.5)/np.sqrt(cont)
#plt.plot(x,confidence,'--',color='g')






