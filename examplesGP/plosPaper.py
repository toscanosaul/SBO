#!/usr/bin/env python

import numpy as np
from math import *
import os
import scipy.io
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
#samplesIteration=[1]
numberIterations=100
numberPrior=30
A=[2,4,8,16]
#A=[2]
varianceb=[1.0/(2.0**k) for k in xrange(5)]#+[2.0**r for r in range(1,5)]
varianceb=[np.log(2.0)*(k-4.0) for k in xrange(5)]+[np.log(2.0)*k for k in range(1,11)]
#varianceb=[1.0/(2.0**k) for k in xrange(2)]

numberdifIteration=len(samplesIteration)
numberAs=len(A)
numberofvariance=len(varianceb)
numberSamples=100

directory="results"

differences=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
varDifferences=np.zeros((numberofvariance,numberdifIteration,numberAs,numberIterations+1))
cont=np.zeros((numberofvariance,numberdifIteration,numberAs,1))

load=False



if load is True:
    for r in xrange(numberofvariance):
        betah=varianceb[r]
        for s in xrange(numberdifIteration):
            n=samplesIteration[s]
            for j in xrange(numberAs):
                Aparam=1.0/(A[j]*n)
                
                y=np.zeros([repetitions,numberIterations+2])
                for i in range(1,repetitions+1):
                    try:
                        temp=np.loadtxt(os.path.join(directory,
                                                     "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+"Results"+
                                                     '%d'%n+"AveragingSamples"+'%d'%+numberPrior+"TrainingPoints",
                                                     "KG","%d"%i+"run","%d"%i+"optimalValues.txt"))
                        if len(temp)>=(numberIterations+1)*2 :
                            for l in range(numberIterations+1):
                                y[i-1,l]=temp[2*l]
                            y[i-1,numberIterations+1]=1
                    except:
                        continue
    
    
                y2=np.zeros([repetitions,numberIterations+2])
                for i in range(1,repetitions+1):
                    try:
                        temp=np.loadtxt(os.path.join(directory,
                                                     "function"+"betah"+'%f'%betah+"Aparam"+'%f'%Aparam+'%d'%n+"Results"+
                                                     '%d'%n+"AveragingSamples"+'%d'%+numberPrior+"TrainingPoints",
                                                     "SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
                        
                        if len(temp)>=(numberIterations+1)*2 :
                            for l in range(numberIterations+1):
                                y2[i-1,l]=temp[2*l]
                            y2[i-1,numberIterations+1]=1
                 
                    except:
                        continue
                diff=np.zeros([0,numberIterations+1])
                for i in range(1,repetitions+1):
                    if (y2[i-1,numberIterations+1]==1 and y[i-1,numberIterations+1]==1):
                        temp=y2[i-1:i,0:numberIterations+1]-y[i-1:i,0:numberIterations+1]
			temp=temp/np.abs(y[i-1:i,0:numberIterations+1])
                        diff=np.concatenate((diff,temp),0)
                        cont[r,s,j,0]+=1
                
                for i in xrange(numberIterations+1):
                    differences[r,s,j,i]=np.mean(diff[:,i])
                    varDifferences[r,s,j,i]=np.var(diff[:,i])


write=load



if write is True:
    matfile = 'weighteddifferences.mat'
    scipy.io.savemat(matfile, mdict={'out': differences}, oned_as='row')
    
    matfile = 'varDiffweighted.mat'
    scipy.io.savemat(matfile, mdict={'out': varDifferences}, oned_as='row')
    
    matfile = 'contDiffweighted.mat'
    scipy.io.savemat(matfile, mdict={'out': cont}, oned_as='row')
    
    
if load is False:
    differences = scipy.io.loadmat('weighteddifferences.mat')['out']
    varDifferences=scipy.io.loadmat('varDiffweighted.mat')['out']
    contt=scipy.io.loadmat('contDiffweighted.mat')['out']
    
    differences2=scipy.io.loadmat('largebetaweighteddifferences.mat')['out']
    

minD=min(differences[:,np.array([0,3]),0,:].min(),differences2.min())
print differences2
t1=differences2.max()
maxD=max(differences[:,np.array([0,3]),0,:].max(),t1)
print minD,maxD
###betah

x=np.linspace(0,samplesIteration[0]*numberIterations,numberIterations+1)
BETA,ITERATIONS=np.meshgrid(varianceb,x)

Z=np.zeros(BETA.shape)
Z2=np.zeros(BETA.shape)

for i in xrange(BETA.shape[0]):
    for j in range(0,5):
	Z[i,j]=differences[j,0,0,i]
	Z2[i,j]=differences[j,3,0,i]
    for j in xrange(5,BETA.shape[1]):
	Z[i,j]=differences2[j-5,0,0,i]
	Z2[i,j]=differences2[j-5,1,0,i]

v2=np.arange(-10.0,10.0,0.1)
norm=cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cmap=cm.PRGn 
v = np.linspace(minD, maxD, 100, endpoint=True)
fig = plt.figure()
cset1 = plt.contourf(BETA, ITERATIONS, Z,v2, 
                 cmap=cm.get_cmap(cmap,len(v2)-1),norm=norm,
                 )
plt.colorbar() 
       #m = cm.ScalarMappable(cmap=cm.jet)
    #m.set_array(Z)
plt.title("Percent Increase between SBO and KG. N is 1 and A is 0.5")
plt.ylabel("Number of samples")
plt.xlabel("beta")
plt.savefig(os.path.join("plots","contourPlotbetahN1A2ver2"+".pdf"))
plt.close("all")
fig = plt.figure()
cset1 = plt.contourf(BETA, ITERATIONS, Z2,v2,
		   cmap=cm.get_cmap(cmap,len(v2)-1),norm=norm,                
                 )
plt.colorbar()
       #m = cm.ScalarMappable(cmap=cm.jet)
    #m.set_array(Z)
plt.title("Percent Increase between SBO and KG. N is 8 and A is 0.06")
plt.ylabel("Number of samples")
plt.xlabel("beta")
plt.savefig(os.path.join("plots","contourPlotbetahN8A2ver2"+".pdf"))
plt.close("all")


