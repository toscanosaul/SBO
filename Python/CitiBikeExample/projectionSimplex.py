#!/usr/bin/env python
#####Porjection onto the simplex

import numpy as np

##Precondition: v isn't in the modified simplex (i.e. the sum is M)
def project(v,M):
    I=[]
    x=v
    n=len(v)
    if np.sum (v)==M:
        return v
    while True:
        nI=n-len(I)
      #  print I
        temp=np.sum(x)
        for i in range(n):
            if i in I:
                x[i]=100
            else:
                x[i]=x[i]-(temp-M)/nI
        if np.all(x>99):
            return x
        else:
            for i in range(n):
                if x[i]<100:
                    x[i]=100
                    I.append(i)
#    return x

if __name__ == '__main__':
    print project(np.array([712,200,300,150]),600)
