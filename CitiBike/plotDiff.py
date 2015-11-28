#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import *
import os

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}


samplesIteration=15
numberIterations=14
numberPrior=5
directory="FinalNonHomogeneous112715Results15AveragingSamples5TrainingPoints"
startIteration=0
endIteration=100

x=np.linspace(startIteration*samplesIteration,samplesIteration*numberIterations,numberIterations-startIteration+1)
y=np.zeros([endIteration+1,numberIterations-startIteration+1+1])
varY=np.zeros([endIteration+1,numberIterations-startIteration+1+1])
#minSol=np.zeros([repetitions,numberIterations+1])
cont=0

for i in range(1,endIteration+1):
    try:
        temp=np.loadtxt(os.path.join(directory,"SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
        if len(temp)>=(numberIterations+1)*2:
	    temp=temp[2*(startIteration):]
    	    varTemp=np.zeros((1,numberIterations-startIteration+1))
 	    yTemp=np.zeros((1,numberIterations-startIteration+1))
            for j in range(numberIterations-startIteration+1):
                yTemp[0,j]=temp[2*j]
                varTemp[0,j]=temp[2*j+1]
	    y[i,0]=1
            y[i,1:]=yTemp
            varY[i,0]=1
            varY[i,1:]=varTemp
	   # varY=np.concatenate((varY,varTemp),0)
	   # y=np.concatenate((y,yTemp),0)
            cont+=1
    except:
	   continue
print cont

means=np.zeros(numberIterations-startIteration+1)
var=np.zeros(numberIterations-startIteration+1)



print len(means)

si=startIteration
y2=np.zeros([endIteration+1,numberIterations-si+1+1])
varY2=np.zeros([endIteration+1,numberIterations-si+1+1])

cont=0

for i in range(1,endIteration+1):
    try:
#        j=i
        temp=np.loadtxt(os.path.join(directory,"KG","%d"%i+"run","%d"%i+"optimalValues.txt"))
        if len(temp)>=(numberIterations+1)*2:
	    temp=temp[2*si:]
            varTemp=np.zeros((1,numberIterations-si+1))
            yTemp=np.zeros((1,numberIterations-si+1))
            for j in range(numberIterations-si+1):
                yTemp[0,j]=temp[2*j]
                varTemp[0,j]=temp[2*j+1]
	    y2[i,0]=1
            y2[i,1:]=yTemp
            varY2[i,0]=1
            varY2[i,1:]=varTemp
          #  varY=np.concatenate((varY,varTemp),0)
          #  y=np.concatenate((y,yTemp),0)
            cont+=1
    except:
	   continue
print "EI"
print cont

	
differences=np.zeros([0,numberIterations-si+1])

for i in range(1,endIteration+1):
    if (y[i,0]==1 and y2[i,0]==1):
	temp=y[i:i+1,1:]-y2[i:i+1,1:]
	differences=np.concatenate((differences,temp),0)
print differences
means=np.zeros(numberIterations-si+1)
var=np.zeros(numberIterations-si+1)

for i in xrange(numberIterations-si+1):
    means[i]=np.mean(differences[:,i])
    var[i]=np.var(differences[:,i])
print "differences mean"
print means
print var
meanDiff=np.mean(differences)
cont=len(differences[:,1])
print "replications"
print cont
plt.plot(x,means,color='b',linewidth=2.0,label='Differences')
confidence=means+1.96*(var**.5)/np.sqrt(cont)
plt.plot(x,confidence,'--',color='b',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(cont)
plt.plot(x,confidence,'--',color='b')
plt.axhline(y=0, xmin=0, xmax=450,color='r')
plt.xlabel('Number of Samples',fontsize=26)
plt.ylabel('Difference',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(os.path.join(directory,"differences.pdf"))
plt.close("all")

