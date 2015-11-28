#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import *
import os

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

repetitions=200

samplesIteration=15
numberIterations=40
numberPrior=5
directory="FinalNonHomogeneous112715Results15AveragingSamples5TrainingPoints"
#directory2=os.path.join("ExpectedImprovement","AnalyticExample","Results"+"%d"%samplesIteration+"AveragingSamples"+"%d"%numberPrior+"TrainingPoints")
#directory3=os.path.join("KnowledgeGradient","AnalyticExample","Results"+"%d"%samplesIteration+"AveragingSamples"+"%d"%numberPrior+"TrainingPoints")

cont=0


x=np.linspace(0,samplesIteration*numberIterations,numberIterations+1)
y=np.zeros([0,numberIterations+1])
for i in range(1,repetitions+1):
    try:
        temp=np.loadtxt(os.path.join(directory,"SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
        if len(temp)>=(numberIterations+1)*2 :
            temp1=np.zeros(numberIterations+1)	    
            for j in range(numberIterations+1):
                temp1[j]=temp[2*j]
            y=np.vstack((y,temp1))
            cont+=1
    except:
        continue


means=np.zeros(numberIterations+1)
var=np.zeros(numberIterations+1)

for i in xrange(numberIterations+1):
    means[i]=np.mean(y[:,i])
    var[i]=np.var(y[:,i])


plt.plot(x,means,color='g',linewidth=2.0,label='SBO')
confidence=means+1.96*(var**.5)/np.sqrt(cont)
plt.plot(x,confidence,'--',color='g',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(cont)
plt.plot(x,confidence,'--',color='g')

cont=0


x=np.linspace(0,samplesIteration*numberIterations,numberIterations+1)
y=np.zeros([0,numberIterations+1])
for i in range(1,repetitions+1):
    try:
        temp=np.loadtxt(os.path.join(directory,"KG","%d"%i+"run","%d"%i+"optimalValues.txt"))
        if len(temp)>=(numberIterations+1)*2 :
            temp1=np.zeros(numberIterations+1)	    
            for j in range(numberIterations+1):
                temp1[j]=temp[2*j]
            y=np.vstack((y,temp1))
            cont+=1
    except:
        continue


means=np.zeros(numberIterations+1)
var=np.zeros(numberIterations+1)

for i in xrange(numberIterations+1):
    means[i]=np.mean(y[:,i])
    var[i]=np.var(y[:,i])
print cont

plt.plot(x,means,color='r',linewidth=2.0,label='KG')
confidence=means+1.96*(var**.5)/np.sqrt(cont)
plt.plot(x,confidence,'--',color='r',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(cont)
plt.plot(x,confidence,'--',color='r')

plt.xlabel('Number of Samples',fontsize=26)
plt.ylabel('Optimum Value of G',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(os.path.join(directory,"comparisonDifferentMethodsIncludesEIKG.pdf"))
plt.close("all")




