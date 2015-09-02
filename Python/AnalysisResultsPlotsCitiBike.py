#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import *
import os

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

repetitions=100

samplesIteration=100
numberIterations=30
numberPrior=50
directory=os.path.join("CitiBikeExample","Results"+"%d"%samplesIteration+"AveragingSamples"+"%d"%numberPrior+"TrainingPoints")


x=np.linspace(0,samplesIteration*numberIterations,numberIterations+1)
y=np.zeros([repetitions,numberIterations+1])
varY=np.zeros([repetitions,numberIterations+1])
minSol=np.zeros([repetitions,numberIterations+1])
cont=0
for i in range(1,100+1):
    temp=np.loadtxt(os.path.join(directory,"SBO","%d"%i+"run","%d"%i+"optimalValues.txt"))
    temp2=np.loadtxt(os.path.join(directory,"SBO","%d"%i+"run","%d"%i+"optimalSolutions.txt"))
    if len(temp)==(numberIterations+1)*2:
        for j in range(numberIterations+1):
            y[cont,j]=temp[2*j]
            varY[cont,j]=temp[2*j+1]
            minSol[cont,j]=np.min(temp2[j,:])
        cont+=1

print cont

means=np.zeros(numberIterations+1)
var=np.zeros(numberIterations+1)

meansVar=np.zeros(numberIterations+1)
varVar=np.zeros(numberIterations+1)

meansMin=np.zeros(numberIterations+1)
varMin=np.zeros(numberIterations+1)


for i in xrange(numberIterations+1):
    means[i]=np.mean(y[:,i])
    var[i]=np.var(y[:,i])
    meansVar[i]=np.mean(varY[:,i])
    varVar[i]=np.var(varY[:,i])
    meansMin[i]=np.mean(minSol[:,i])
    varMin[i]=np.var(minSol[:,i])

plt.plot(x,means,color='r',linewidth=2.0,label='SBO')
confidence=means+1.96*(var**.5)/np.sqrt(repetitions)
plt.plot(x,confidence,'--',color='r',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(repetitions)
plt.plot(x,confidence,'--',color='r')

plt.axhline(y=0, xmin=0, xmax=samplesIteration*numberIterations,color='b',label='Optimal Solution')

#y=np.zeros([35,n])
#for i in range(1,36):

#    y[i-1,:]=np.loadtxt("%d"%i+"G(xn)KGAnalytic.txt")
    
#for i in xrange(n):
#    means[i]=np.mean(y[:,i])
#    var[i]=np.var(y[:,i])
#means+=.19
#plt.plot(x,means,color='g',linewidth=2.0,label='KG')
#confidence=means+1.96*(var**.5)/np.sqrt(35.0)
#plt.plot(x,confidence,'--',color='g',label="95% CI")
#confidence=means-1.96*(var**.5)/np.sqrt(35.0)
#plt.plot(x,confidence,'--',color='g')




plt.xlabel('Number of Samples',fontsize=26)
plt.ylabel('Optimum Value of G',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(os.path.join(directory,"comparisonDifferentMethods.pdf"))
plt.close("all")

plt.plot(x,meansVar,color='r',linewidth=2.0,label='Variances of the estimations of G')
confidence=meansVar+1.96*(varVar**.5)/np.sqrt(repetitions)
plt.plot(x,confidence,'--',color='r',label="95% CI")
confidence=meansVar-1.96*(varVar**.5)/np.sqrt(repetitions)
plt.plot(x,confidence,'--',color='r')

plt.xlabel('Number of Samples',fontsize=26)
plt.ylabel('Variance of the Value of G',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(os.path.join(directory,"VariancesComparisonDifferentMethods.pdf"))
plt.close("all")

plt.plot(x,meansMin,color='r',linewidth=2.0,label='Variances of the estimations of G')
confidence=meansMin+1.96*(varMin**.5)/np.sqrt(repetitions)
plt.plot(x,confidence,'--',color='r',label="95% CI")
confidence=meansMin-1.96*(varMin**.5)/np.sqrt(repetitions)
plt.plot(x,confidence,'--',color='r')

plt.xlabel('Number of Samples',fontsize=26)
plt.ylabel('Min Entry of the Solutions',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig(os.path.join(directory,"MinimumEntryComparisonDifferentMethods.pdf"))
plt.close("all")

