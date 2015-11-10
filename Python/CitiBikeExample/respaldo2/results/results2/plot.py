#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import *

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

s=9

numberIterations=100

x=np.linspace(100,n*nSamples,n)
y=np.zeros([77,n])
for i in range(1,21):
    y[i-1,:]=np.loadtxt("%d"%i+"G(xn)SBOAnalytic3.txt")

for i in range(23,80):
    y[i-3,:]=np.loadtxt("%d"%i+"G(xn)SBOAnalytic3.txt")

means=np.zeros(n)
var=np.zeros(n)

for i in xrange(n):
    means[i]=np.mean(y[:,i])
    var[i]=np.var(y[:,i])

means+=.19
plt.plot(x,means,color='r',linewidth=2.0,label='SBO')
confidence=means+1.96*(var**.5)/np.sqrt(77.0)
plt.plot(x,confidence,'--',color='r',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(77.0)
plt.plot(x,confidence,'--',color='r')


#y=np.zeros([51,n])
#for i in range(1,52):
#    y[i-1,:]=np.loadtxt("%d"%i+"G(xn)EIAnalytic.txt")

#for i in xrange(n):
#    means[i]=np.mean(y[:,i])
#    var[i]=np.var(y[:,i])
 
#means+=.19
#plt.plot(x,means,color='b',linewidth=2.0,label='EI')
#confidence=means+1.96*(var**.5)/np.sqrt(51.0)
#plt.plot(x,confidence,'--',color='b',label="95% CI")
#confidence=means-1.96*(var**.5)/np.sqrt(51.0)
#plt.plot(x,confidence,'--',color='b')

y=np.zeros([35,n])
for i in range(1,36):

    y[i-1,:]=np.loadtxt("%d"%i+"G(xn)KGAnalytic.txt")
    
for i in xrange(n):
    means[i]=np.mean(y[:,i])
    var[i]=np.var(y[:,i])
means+=.19
plt.plot(x,means,color='g',linewidth=2.0,label='KG')
confidence=means+1.96*(var**.5)/np.sqrt(35.0)
plt.plot(x,confidence,'--',color='g',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(35.0)
plt.plot(x,confidence,'--',color='g')




plt.xlabel('Number of Samples',fontsize=26)
plt.ylabel('Optimum Value of G',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig("comparisonDifferentMethods.pdf")
plt.close("all")#!/usr/bin/env python




