#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import *

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 15}

n=8



#y=np.zeros(n)
#y[0]=2.411
#y[1]=3.91
#y[2]=2.981
#y[3]=3.027
#y[4]=102.966
#y[5]=2.267
    


#y=np.loadtxt("valuesCitibike.txt")
y2=np.loadtxt("G(xn)PIAnalytic.txt")
y=np.zeros(n)
y=y2[1:n]
#for i in range(n):
#    y[i]=-y2[2*i]
x=np.linspace(2,n,n-1)

#y=np.loadtxt("valuesCitibike.txt")
y3=np.loadtxt("G(xn)EIAnalytic.txt")
y4=np.zeros(n)
y4=y3[1:n]
#for i in range(n):
#    y[i]=-y2[2*i]

#y=np.loadtxt("valuesCitibike.txt")
y5=np.loadtxt("G(xn)UCBAnalytic.txt")
y6=np.zeros(n)
y6=y5[1:n]
#for i in range(n):
#    y[i]=-y2[2*i]

y7=np.loadtxt("G(xn)SBOAnalytic3.txt")
y8=np.zeros(n)
y8=y7[1:n]

y9=np.loadtxt("prior10G(xn)KGAnalytic100samples100.txt")
y10=np.zeros(n)
y10=y9[1:n]

plt.plot(x,y,linewidth=2.0,label='PI')
plt.plot(x,y4,linewidth=2.0,label='EI')
plt.plot(x,y6,linewidth=2.0,label='UCB')
plt.plot(x,y8,linewidth=2.0,label='SBO')
plt.plot(x,y10,linewidth=2.0,label='KG')
#plt.plot(x,y2,linewidth=2.0,label='SBO')
#plt.legend()
plt.xlabel('Iteration number',fontsize=26)
plt.ylabel('Optimum Value of G',fontsize=24)
plt.legend(loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig("comparisonDifferentMethods.pdf")
plt.close("all")