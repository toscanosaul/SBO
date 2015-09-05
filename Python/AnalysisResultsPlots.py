#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import *
import os

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

repetitions=1000

samplesIteration=15
numberIterations=19
numberPrior=5
directory=os.path.join("AnalyticExample","Results"+"%d"%samplesIteration+"AveragingSamples"+"%d"%numberPrior+"TrainingPoints")
directory2=os.path.join("ExpectedImprovement","AnalyticExample","Results"+"%d"%samplesIteration+"AveragingSamples"+"%d"%numberPrior+"TrainingPoints")


x=np.linspace(0,samplesIteration*numberIterations,numberIterations+1)
y=np.zeros([repetitions,numberIterations+1])
for i in range(1,repetitions+1):
    temp=np.loadtxt(os.path.join(directory,"SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
    for j in range(numberIterations+1):
        y[i-1,j]=temp[2*j]


means=np.zeros(numberIterations+1)
var=np.zeros(numberIterations+1)

for i in xrange(numberIterations+1):
    means[i]=np.mean(y[:,i])
    var[i]=np.var(y[:,i])

plt.plot(x,means,color='r',linewidth=2.0,label='SBO')
confidence=means+1.96*(var**.5)/np.sqrt(repetitions)
plt.plot(x,confidence,'--',color='r',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(repetitions)
plt.plot(x,confidence,'--',color='r')

plt.axhline(y=0, xmin=0, xmax=samplesIteration*numberIterations,color='b',label='Optimal Solution')

y2=np.zeros([972,numberIterations+1])
cont=0
for i in range(1,repetitions+1):
    temp=np.loadtxt(os.path.join(directory2,"EI","%d"%i+"run","%d"%i+"optimalValues.txt"))
    if len(temp)>=(numberIterations+1)*2:
        for j in range(numberIterations+1):
            y2[cont,j]=temp[2*j]
        cont+=1
  #  y[i-1,:]=np.loadtxt("%d"%i+"G(xn)KGAnalytic.txt")
print cont

for i in xrange(numberIterations+1):
    means[i]=np.mean(y2[:,i])
    var[i]=np.var(y2[:,i])


plt.plot(x,means,color='g',linewidth=2.0,label='EI')
confidence=means+1.96*(var**.5)/np.sqrt(cont)
plt.plot(x,confidence,'--',color='g',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(cont)
plt.plot(x,confidence,'--',color='g')




plt.xlabel('Number of Samples',fontsize=26)
plt.ylabel('Optimum Value of G',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(os.path.join(directory,"comparisonDifferentMethodsIncludesEI.pdf"))
plt.close("all")




