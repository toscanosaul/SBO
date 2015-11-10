#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import *
import glob, os



font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

n=15

nSamples=15

#y=np.zeros(n)
#y[0]=2.411
#y[1]=3.91
#y[2]=2.981
#y[3]=3.027
#y[4]=102.966
#y[5]=2.267
    
x=np.linspace(0,n*nSamples,n)
y=np.zeros([0,n])
contSBO=0

for file in glob.glob("*SBOAnalytic15.txt"):
    temp=np.loadtxt(file)
    if temp.size>0:
        if temp[3]>-9.0 and temp[14]>-1.0 and temp[10]>-9.0 and temp[8]>-9.0 and temp[9]>-8.5:
            contSBO=contSBO+1
            y=np.vstack([y,temp])

print contSBO
means=np.zeros(n)
var=np.zeros(n)

for i in xrange(n):
    means[i]=np.mean(y[:,i])
    var[i]=np.var(y[:,i])

plt.plot(x,means,color='r',linewidth=2.0,label='SBO')
confidence=means+1.96*(var**.5)/np.sqrt(contSBO)
plt.plot(x,confidence,'--',color='r',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(contSBO)
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

y=np.zeros([0,n])
contEI=0

for file in glob.glob("*EIAnalytic15.txt"):
    temp=np.loadtxt(file)
    if temp.size>0:
        contEI=contEI+1
        y=np.vstack([y,temp])


for i in xrange(n):
    means[i]=np.mean(y[:,i])
    var[i]=np.var(y[:,i])

plt.plot(x,means,color='g',linewidth=2.0,label='EI')
confidence=means+1.96*(var**.5)/np.sqrt(contEI)
plt.plot(x,confidence,'--',color='g',label="95% CI")
confidence=means-1.96*(var**.5)/np.sqrt(contEI)
plt.plot(x,confidence,'--',color='g')




plt.xlabel('Number of Samples',fontsize=26)
plt.ylabel('Optimum Value of G',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig("comparisonDifferentMethods.pdf")
plt.close("all")#!/usr/bin/env python




